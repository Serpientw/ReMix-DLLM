# ReMix <a href="https://arxiv.org/abs/2602.22868" alt="arXiv"> <img src="https://img.shields.io/badge/arXiv-2602.22868-b31b1b.svg?style=flat" /></a> <a href="https://arxiv.org/pdf/2602.22868" alt="arXiv"> <img src="https://img.shields.io/badge/Paper-PDF-b31b1b.svg?style=flat" /></a>

Official Implementation of "Rejection Mixing: Fast Semantic Propagation of Mask Tokens for Efficient DLLM Inference" 

## Evaluation
This repository offers the codebase for **ReMix**, including comprehensive scripts for reproducing demos and evaluations on [LLaDA](https://github.com/ML-GSAI/LLaDA) and [MMaDA](https://github.com/Gen-Verse/MMaDA). 

### Environment Management
We recommend using [uv](https://github.com/astral-sh/uv) for dependency and virtual environment management.

For LLaDA:
```
pip install uv
cd LLaDA
uv venv --python 3.11 dev
source dev/bin/activate
uv pip install -r requirements.txt
```
For MMaDA:
```
cd MMaDA
pip install uv
uv venv --python 3.11 dev
source dev/bin/activate
uv pip install -r requirements.txt
### For lmms-eval
cd lmms_eval
uv pip install -e .
```

### Evaluation on LLaDA
1. Prepare Model and Datasets

Before running inference or evaluation, please download the following models and datasets from [Hugging Face](https://huggingface.co/) into the specified local directories (e.g., `./LLaDA/models/` and `./LLaDA/data/`). 

You may use either `huggingface-cli` or the Python `datasets` library to complete the download.

| Model Name         | Hugging Face Repo                                               | Local Path                     |
|--------------------|------------------------------------------------------------------|--------------------------------|
| LLaDA-8B-Instruct  | [GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) | `./LLaDA/models/LLaDA-8B-Instruct/`  |

| Dataset Name  | Hugging Face Repo                                                                 | Local Path          |
|---------------|------------------------------------------------------------------------------------|---------------------|
| GSM8K         | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)                    | `./LLaDA/data/gsm8k/`     |
| MATH-500      | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | `./LLaDA/data/math500/`   |
| HumanEval     | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) | `./LLaDA/data/humaneval/` |
| ai2_arc     | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc)              | `./LLaDA/data/ai2_arc/`       |

Datasets not listed above are already included in the [`./LLaDA/data/`](./LLaDA/data/) directory

2. Demo

We have provided a quick demo to run our method on LLaDA, make sure to set `model_path` to your model path.
```
python demo.py
```
