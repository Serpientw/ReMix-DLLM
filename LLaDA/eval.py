import os
import sys
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules["flash_attn"] = None
sys.modules["flash_attn_2_cuda"] = None
sys.modules["flash_attn_3"] = None
import json
import argparse
import tqdm
import time
from datasets import load_dataset
import torch
import torch.distributed as dist  # NEW
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import LLaDAModelLM, decoding_default, decoding_remix
import dataset_utils
from human_eval.evaluation import evaluate_functional_correctness
import tempfile
from dataset_utils.eval_correctness_mbpp.evaluation import evaluate_functional_correctness as evaluate_functional_correctness_mbpp

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def ddp_is_enabled():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def ddp_setup():
    if ddp_is_enabled() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def ddp_cleanup():
    if ddp_is_enabled() and dist.is_initialized():
        dist.destroy_process_group()

def get_rank():
    return dist.get_rank() if (ddp_is_enabled() and dist.is_initialized()) else 0

def get_world_size():
    return dist.get_world_size() if (ddp_is_enabled() and dist.is_initialized()) else 1

def parse_gen_kwargs(s: str):
    out = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if "=" not in kv:
            raise argparse.ArgumentTypeError(f"Invalid item: {kv!r}, expected key=value")
        k, v = kv.split("=", 1)
        vl = v.lower()
        if vl in {"true", "false"}:
            val = (vl == "true")
        else:
            try:
                val = int(v)
            except ValueError:
                try:
                    val = float(v)
                except ValueError:
                    val = v
        out[k] = val
    return out

def get_generation_function(method_name):
    if method_name == 'default': return decoding_default
    elif method_name == 'remix': return decoding_remix
    else: raise ValueError(f"Unknown method: {method_name}")

def save_results(config, method_name, raw_outputs, correct_count, total_count, total_steps, total_time, total_len, total_tokens, external_metrics):
    summary = {
        "config": config,        
        "method": method_name,   
        "metrics": {},
        "raw_outputs": raw_outputs
    }
    eval_mode = config['dataset_config']['eval_mode']
    if external_metrics:
        summary['metrics'].update(external_metrics)
        accuracy = external_metrics['pass@1']
    else:
        accuracy = "N/A"
        if eval_mode == 'binary':
            accuracy = correct_count / total_len if total_len > 0 else 0
        elif eval_mode == 'partial_credit':
            accuracy = correct_count / total_count if total_count > 0 else 0
        summary['metrics']['accuracy'] = accuracy
    
    avg_steps = total_steps / total_len if total_len > 0 else 0
    avg_time = total_time / total_len if total_len > 0 else 0
    summary['metrics']['average_steps'] = avg_steps
    summary['metrics']['average_latency'] = avg_time
    summary["metrics"]["throughput_tps"] = total_tokens / total_time if total_time > 0 else 0 
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"./results/{config['dataset_name']}_{method_name}_{timestamp}.json"
    if config.get('num_samples') is not None:
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_samples{config['num_samples']}{ext}"
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Results saved in .jsonl format to {output_path}")
    print("\n--- Evaluation Summary ---")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Method: {method_name}")
    print(f"Accuracy: {accuracy:.4f}" if isinstance(accuracy, float) else f"Accuracy: {accuracy}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Latency: {avg_time:.2f}")
    print(f"Throughput: {summary['metrics']['throughput_tps']:.2f}")

def run_single_task_evaluation(config, model, tokenizer):
    dataset_name = config['dataset_name']
    if get_rank() == 0:
        print(f"==> Running Single-Task Evaluation for {dataset_name}")

    dataset_cfg = config['dataset_config']
    doc_to_text_fn = getattr(dataset_utils, dataset_cfg['doc_to_text_fn'])
    eval_fn = getattr(dataset_utils, dataset_cfg.get('eval_fn')) if dataset_cfg.get('eval_fn') else None
    extract_answer_fn = getattr(dataset_utils, dataset_cfg.get('extract_answer_fn')) if dataset_cfg.get('extract_answer_fn') else None
    is_correct_fn = getattr(dataset_utils, dataset_cfg.get('is_correct_fn')) if dataset_cfg.get('is_correct_fn') else None
    
    gen_cfg = config['generation_args']
    method_name = config['method']
    method_params = config.get('method_args', {}).get(method_name, {})
    generation_fn = get_generation_function(method_name)

    if get_rank() == 0:
        print(f"==> Loading dataset...")
    loader_type = dataset_cfg.get('data_loader', 'huggingface')
    dataset_path = os.path.join(config['data_root'], dataset_cfg['load_dataset_args']['path'])
    
    if loader_type == 'huggingface':
        dataset_name_hf = dataset_cfg['load_dataset_args'].get('name')
        dataset = load_dataset(dataset_path, dataset_name_hf, trust_remote_code=True)[dataset_cfg['split']]
    elif dataset_name == 'mbpp':
        loader_fn = getattr(dataset_utils, dataset_cfg['loader_fn'])
        dataset = list(loader_fn(dataset_path))
    else:
        loader_fn = getattr(dataset_utils, dataset_cfg['loader_fn'])
        dataset = loader_fn(dataset_path)

    num_samples = config.get('num_samples')
    if num_samples is not None:
        if hasattr(dataset, 'select'):
            dataset = dataset.select(range(num_samples))
        else:
            dataset = dataset[:num_samples]

    world_size = get_world_size()
    rank = get_rank()

    total_len_all = len(dataset)
    global_indices = list(range(total_len_all))
    shard_indices = global_indices[rank::world_size]

    if get_rank() == 0:
        print(f"==> Total samples: {total_len_all} | world_size={world_size}")

    # --- Warm-up ---
    if len(shard_indices) > 0:
        warmup_doc = dataset[shard_indices[0]]
        trailing_prompt = "" 
        if dataset_cfg['eval_mode'] in ['save_only', 'code_execution']:
            context, _, trailing_prompt = doc_to_text_fn(warmup_doc)
        else:
            context, _ = doc_to_text_fn(warmup_doc)
        prompt = tokenizer.apply_chat_template(context, add_generation_prompt=True, tokenize=False) + trailing_prompt + dataset_cfg.get('prompt_suffix', '')
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        _ = generation_fn(model, input_ids, **gen_cfg, **method_params)
    if get_rank() == 0:
        print("==> Warm-up complete.")

    # === Main Loop ===
    raw_outputs_local, correct_count_local, total_count_local, total_steps_local, total_tokens = [], 0, 0, 0, 0

    # Show progress bar only in rank0
    prog_iter = shard_indices
    if rank == 0:
        prog_iter = tqdm.tqdm(shard_indices, desc=f"Evaluating {dataset_name} with method '{method_name}' (rank0)")

    total_time_local = 0.0
    for i in prog_iter:
        doc = dataset[i]
        
        gt_doc, gt_for_eval, trailing_prompt = None, None, ""
        if dataset_cfg['eval_mode'] in ['save_only', 'code_execution']:
            context, gt_doc, trailing_prompt = doc_to_text_fn(doc)
        else:
            context, gt_for_eval = doc_to_text_fn(doc)
        
        prompt = tokenizer.apply_chat_template(context, add_generation_prompt=True, tokenize=False) + trailing_prompt + dataset_cfg.get('prompt_suffix', '')
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        
        start_time = time.time()
        gen_output, steps = generation_fn(model, input_ids, **gen_cfg, **method_params)
        end_time = time.time()
        
        generation_time = end_time - start_time 
        total_time_local += generation_time
        gen_str = tokenizer.batch_decode(gen_output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        total_steps_local += steps
        total_tokens += gen_output[:, input_ids.shape[1]:].shape[1]
        
        pred, is_correct = None, "N/A"
        result_item = {'index': i, 'full_response': gen_str, 'steps': steps}

        if dataset_cfg['eval_mode'] == 'binary':
            pred = extract_answer_fn(gen_str)
            if dataset_name == 'countdown':
                target = int(doc['output'])
                num_str = doc['input']
                numbers = [int(i) for i in num_str.split(',')]
                is_correct = is_correct_fn(pred, target, numbers)
            else:
                is_correct = is_correct_fn(pred, gt_for_eval)
            if is_correct: correct_count_local += 1
            result_item.update({'prediction': pred, 'is_correct': is_correct})
        elif dataset_cfg['eval_mode'] == 'partial_credit':
            target = doc['Solution']
            correct, total, pred = eval_fn(gen_str, target, doc['Puzzle'])
            correct_count_local += correct
            total_count_local += total
            result_item.update({'prediction': pred, 'correct_cells': correct, 'total_cells': total})
        elif dataset_cfg['eval_mode'] == 'save_only':
            if dataset_name == 'mbpp':
                gen_str = f"```python\n" + gen_str
                gen_code = extract_answer_fn(gen_str, doc['entry_point'])
                result_item['completion'] = gen_code
            else:
                pred = extract_answer_fn(gen_str, doc)
                result_item['completion'] = pred
            if 'task_id' in gt_doc: result_item['task_id'] = gt_doc['task_id']
            
        raw_outputs_local.append(result_item)
    
    # === Gather ===
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    t_correct = torch.tensor(correct_count_local, device=device, dtype=torch.int)
    t_total = torch.tensor(total_count_local, device=device, dtype=torch.int)
    t_steps = torch.tensor(total_steps_local, device=device, dtype=torch.int)
    t_time = torch.tensor(total_time_local, device=device, dtype=torch.float64)
    t_tokens = torch.tensor(total_tokens, device=device, dtype=torch.int)
    dist.all_reduce(t_correct, op=dist.ReduceOp.SUM) if ddp_is_enabled() else None
    dist.all_reduce(t_total, op=dist.ReduceOp.SUM) if ddp_is_enabled() else None
    dist.all_reduce(t_steps, op=dist.ReduceOp.SUM) if ddp_is_enabled() else None
    dist.all_reduce(t_time, op=dist.ReduceOp.SUM) if ddp_is_enabled() else None
    dist.all_reduce(t_tokens, op=dist.ReduceOp.SUM) if ddp_is_enabled() else None

    t_correct = t_correct.cpu()
    t_total = t_total.cpu()
    t_steps = t_steps.cpu()
    t_time = t_time.cpu()
    t_tokens = t_tokens.cpu()

    correct_count_all = t_correct.item()
    total_count_all = t_total.item()
    total_steps_all = t_steps.item()
    total_time_all = t_time.item()
    total_tokens_all = t_tokens.item()

    if ddp_is_enabled():
        gathered = [None for _ in range(world_size)]
        if rank == 0:
            dist.gather_object(raw_outputs_local, gathered, dst=0)
        else:
            dist.gather_object(raw_outputs_local, None, dst=0)
        if rank == 0:
            raw_outputs = []
            for part in gathered:
                if part is not None:
                    raw_outputs.extend(part)
            raw_outputs.sort(key=lambda x: x['index'])
    else:
        raw_outputs = raw_outputs_local

    if rank == 0:
        total_len = total_len_all
        final_metrics = {}
        output_formatter = config['dataset_config'].get('output_formatter', 'default')

        if output_formatter == 'humaneval':
            if evaluate_functional_correctness is None:
                print("Warning: 'human-eval' library not found. Skipping functional correctness evaluation.")
            else:
                print("\n==> Generations complete. Calling official evaluation script...")
                with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".jsonl") as temp_f:
                    for item in raw_outputs:
                        temp_f.write(json.dumps(item) + "\n")
                        temp_file_path = temp_f.name
                if dataset_name == 'humaneval':
                    final_metrics = evaluate_functional_correctness(temp_file_path)
                elif dataset_name == 'mbpp':
                    problem_file = config['dataset_config'].get('problem_file', 'default')
                    final_metrics = evaluate_functional_correctness_mbpp(temp_file_path, problem_file=problem_file, is_mbpp=True)
                os.unlink(temp_file_path)

        correct_count, total_count, total_steps, total_time, total_tokens = correct_count_all, total_count_all, total_steps_all, total_time_all, total_tokens_all
        save_results(config, method_name, raw_outputs, correct_count, total_count, total_steps, total_time, total_len, total_tokens, external_metrics=final_metrics)

def main():
    parser = argparse.ArgumentParser(description="Unified Config-driven Evaluation Script for Language Models")
    parser.add_argument("--config", type=str, required=True, help="Path to the dataset config YAML file (e.g., configs/gsm8k.yaml)")
    parser.add_argument("--method", type=str, required=True, default="default")
    parser.add_argument("--gen-kwargs", type=parse_gen_kwargs, required=False)
    args = parser.parse_args()

    ddp_setup() 

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    generation_args = dict(config.get("generation_args") or {})
    method_in_cfg = config.get("method", "default")
    method = args.method or method_in_cfg
    method_args_all = dict(config.get("method_args") or {})
    method_args_cur = dict(method_args_all.get(method) or {})
    cli = args.gen_kwargs or {}
    for k, v in cli.items():
        if k in generation_args:
            generation_args[k] = v
        else:
            method_args_cur[k] = v
    method_args_all[method] = method_args_cur
    config["generation_args"] = generation_args
    config["method_args"] = method_args_all
    config["method"] = method

    if get_rank() == 0:
        print(f"==> Loaded config for: {config['dataset_name']}")
        print(f"==> Method: {config['method']}")
        print(f"==> Final generation_args: {config['generation_args']}")
        print(f"==> Final method_args[{method}]: {config['method_args'][method]}")

    model_path = config['model_path']
    if get_rank() == 0:
        print(f"==> Loading model: {model_path}")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    model = LLaDAModelLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    try:
        with torch.no_grad():
            run_single_task_evaluation(config, model, tokenizer)
    finally:
        ddp_cleanup()  

if __name__ == "__main__":
    main()