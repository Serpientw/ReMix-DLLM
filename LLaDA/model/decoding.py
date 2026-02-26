import torch
import torch.nn.functional as F
import numpy as np
import time

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

@ torch.no_grad()
def decoding_default(model, prompt, steps=256, gen_length=256, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, store=False):
    '''
    Default decoding function from LLaDA paper
    '''
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    
    if store:
        history = []
        
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            
            if store:
                history.append(x.clone().cpu())

    return (x, steps * num_blocks, history) if store else (x, steps * num_blocks)

@torch.no_grad()
def decoding_remix(model, prompt, gen_length=256, block_length=256, temperature=0.,
                        mask_id=126336, threshold=0.9, js_threshold=0.3, beta_mix=0.5, store=False):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature. (Currently not used.)
        mask_id: The toke id of [MASK].
        threshold: The confidence threshold for decoding.
        js_threshold: The JS divergence threshold for rejection.
        beta_mix: The mixing ratio for mixed embeddings. (Higher means more continuous embedding.)
        store: Whether to store the trajectory of decoding.
    """
    device = model.device
    W = model.model.transformer.wte.weight                         # [vocab, hidden]
    w_mask = W[mask_id].view(1, 1, -1)                             # [1, 1, hidden]

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    steps = 0
    history = [] if store else None
    prob_history = None
    
    for num_block in range(num_blocks):
        mask_index_block = (x == mask_id)
        mask_index_block[:, prompt.shape[1] + (num_block + 1) * block_length:] = False

        block_steps = 0
        prob_history_ready = False

        while mask_index_block.any():
            inputs_embeds_curr = W[x]  # [b,l,h]

            if prob_history is not None and prob_history_ready:
                _, _, vocab_size = prob_history.shape
                k = min(vocab_size, 50)

                top_k_probs, top_k_indices = torch.topk(prob_history, k=k, dim=-1)
                cum_probs = torch.cumsum(top_k_probs, dim=-1)

                alpha_min, alpha_max = 0.2, 0.9
                max_p = top_k_probs.max(dim=-1).values
                alpha = (2 * max_p).clamp(min=alpha_min, max=alpha_max)

                alpha_expanded = alpha.unsqueeze(-1).expand(-1, -1, k)
                valid_mask = cum_probs <= alpha_expanded
                valid_mask[..., 0] = True  

                selected_probs = torch.zeros_like(prob_history)
                selected_probs.scatter_(-1, top_k_indices, 
                                    torch.where(valid_mask, top_k_probs, torch.tensor(0.0, device=prob_history.device)))

                p_sel = selected_probs.sum(dim=-1, keepdim=True).clamp(min=0.0, max=1.0)
                
                soft_emb = selected_probs @ W + (1.0 - p_sel) * w_mask

                mixed_embeddings = torch.where(
                    mask_index_block.unsqueeze(-1),
                    (1 - beta_mix) * inputs_embeds_curr + beta_mix * soft_emb,
                    inputs_embeds_curr
                )
                inputs_embeds_curr = mixed_embeddings
                
            logits = model(None, inputs_embeds=inputs_embeds_curr).logits
            
            # Decode tokens exceeding the threshold
            p = F.softmax(logits.to(W.dtype), dim=-1)                                     # [b,l,v]
            max_accept = min(max(int(mask_index_block.sum() * 0.7), 5), 20)
            x0 = torch.argmax(logits, dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # [b, l]

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index_block, x0, x)
            confidence = torch.where(mask_index_block, x0_p, -np.inf)

            transfer_index = confidence > threshold
            if transfer_index.sum() > max_accept:
                # Get top max_accept tokens
                _, indices = torch.topk(confidence, k=max_accept, largest=True)
                transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
                transfer_index.view(-1)[indices] = True
            else:
                if not transfer_index.any():
                    max_confidence_index = torch.argmax(confidence)
                    transfer_index.view(-1)[max_confidence_index] = True

            x[transfer_index] = x0[transfer_index]
            mask_index_block[transfer_index] = False

            # Reset prob_history to all zero at block_step=0
            if prob_history is None:
                prob_history = p.detach()
                js_reset_mask = None
            else:
                if mask_index_block.any():
                    eps = torch.finfo(torch.float32).eps

                    # Calculate JS divergence
                    prev = prob_history[mask_index_block].clamp_min(eps).to(torch.float32)
                    curr = p[mask_index_block].clamp_min(eps).to(torch.float32)
                    m = 0.5 * (prev + curr)

                    kl_pm = (prev * (prev.log() - m.log())).sum(dim=-1)            # [N]
                    kl_qm = (curr * (curr.log() - m.log())).sum(dim=-1)            # [N]
                    js = 0.5 * (kl_pm + kl_qm)

                    js_reset_mask = torch.zeros_like(mask_index_block, dtype=torch.bool)
                    js_reset_mask[mask_index_block] = js > js_threshold 
                else:
                    js_reset_mask = None

                next_prob_history = p.detach()
                if js_reset_mask is not None and js_reset_mask.any():
                    next_prob_history[js_reset_mask] = 0.0
                prob_history = next_prob_history
                prob_history_ready = True
                
            if store:
                history.append(x.clone().cpu())

            block_steps += 1

        steps += block_steps

    return (x, steps, history) if store else (x, steps)
