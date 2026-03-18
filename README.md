# ReMix <a href="https://arxiv.org/abs/2602.22868" alt="arXiv"> <img src="https://img.shields.io/badge/arXiv-2602.22868-b31b1b.svg?style=flat" /></a> <a href="https://arxiv.org/pdf/2602.22868" alt="arXiv"> <img src="https://img.shields.io/badge/Paper-PDF-b31b1b.svg?style=flat" /></a>

Official Implementation of "Rejection Mixing: Fast Semantic Propagation of Mask Tokens for Efficient DLLM Inference" 

## Evaluation
This repository offers the codebase for **ReMix**, including comprehensive scripts for reproducing demos and evaluations on [LLaDA](https://github.com/ML-GSAI/LLaDA) and [MMaDA](https://github.com/Gen-Verse/MMaDA). 

### Environment Management
We recommend using [uv](https://github.com/astral-sh/uv) for dependency and virtual environment management.

For LLaDA:
```bash
pip install uv
cd LLaDA
uv venv --python 3.11 dev
source dev/bin/activate
uv pip install -r requirements.txt
```
For MMaDA:
```bash
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
