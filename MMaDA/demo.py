import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from models import MAGVITv2, MMadaConfig, MMadaModelLM
from models.prompting_utils import UniversalPrompting
from models.utils import get_config, flatten_omega_conf, image_transform
from transformers import AutoTokenizer, AutoConfig

reasoning_prompt = "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here\n"

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("Gen-Verse/MMaDA-8B-MixCoT", padding_side="left")
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=512,
                                       special_tokens=(
                                       "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>",
                                       "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=0.1,
                                       use_reserved_token=True)

    vq_model = MAGVITv2.from_pretrained("showlab/magvitv2").to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    model = MMadaModelLM.from_pretrained("Gen-Verse/MMaDA-8B-MixCoT", trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.to(device)
    mask_token_id = model.config.mask_token_id

    question='Please describe this image in detail.'
    image_path = "./demo.png"
    image_ori = Image.open(image_path).convert("RGB")
    image = image_transform(image_ori, resolution=512).to(device)
    image = image.unsqueeze(0)
    image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
    batch_size = 1
    question = reasoning_prompt + question
    input_ids = uni_prompting.text_tokenizer(['<|start_header_id|>user<|end_header_id|>\n' + question + '<eot_id><|start_header_id|>assistant<|end_header_id|>\n'], padding=True, padding_side="left")[
        'input_ids']
    input_ids = torch.tensor(input_ids).to(device)

    input_ids = torch.cat([
        (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
        (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
        image_tokens,
        (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
        (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
        input_ids
    ], dim=1).long()

    gen_length = 256
    decoding_step = 256
    block_length = 128

    print(f'Default generation on the demo image:')
    output_ids, history = model.mmu_generate(input_ids, max_new_tokens=gen_length, steps=decoding_step, block_length=block_length, store=True)
    with open("case_default.txt", 'w') as f:
        for i in range(len(history)):
            step_output = uni_prompting.text_tokenizer.batch_decode(history[i][:, input_ids.shape[1]:], skip_special_tokens=True)
            f.write(f'Step {i}: \n{step_output}\n\n')
            
    text = uni_prompting.text_tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
    response= f'\nAnswer: ' + text[0] + f'\nDecoding step:{decoding_step}\n'
    print(response)

    # print("Remix generation on the demo image:")
    # output_ids, inference_step, history = model.mmu_generate_remix(input_ids, gen_length=gen_length, block_length=block_length, threshold=0.8, beta_mix=0.4, js_threshold=0.3, store=True)
    # with open("case_remix.txt", 'w') as f:
    #     for i in range(len(history)):
    #         step_output = uni_prompting.text_tokenizer.batch_decode(history[i][:, input_ids.shape[1]:], skip_special_tokens=True)
    #         f.write(f'Step {i}: \n{step_output}\n\n')
            
    # text = uni_prompting.text_tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
    # response = f'\nAnswer: ' + text[0] + f'\nDecoding step:{inference_step}\n'
    # print(response)

