import torch
import numpy as np
import time

import torch.nn.functional as F
from transformers import AutoTokenizer
from model import LLaDAModelLM, decoding_default, decoding_remix


def main():
    device = 'cuda:0'
    model_path = 'GSAI-ML/LLaDA-8B-Instruct'

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(model_path,torch_dtype=torch.bfloat16).to(device).eval()
    
    prompt = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    m = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    
    # out, step, history = decoding_default(model, input_ids, steps=256, gen_length=256, block_length=128, temperature=0., cfg_scale=0., remasking='low_confidence', store=True)
    out, step, history = decoding_remix(model, input_ids, gen_length=256, block_length=128, threshold=0.8, js_threshold=0.3, beta_mix=0.5, store=True) 

    print(f'Total steps: {step}\n')
    for i in range(len(history)):
        step_output = tokenizer.batch_decode(history[i][:, input_ids.shape[1]:], skip_special_tokens=False)[0]
        print(f'Step {i}: \n{step_output}\n\n')
    
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=False))

if __name__ == '__main__':
    main()
