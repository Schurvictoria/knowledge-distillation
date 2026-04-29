# Bidirectional Knowledge Distillation between LLM and Sequence Models for Event Sequences

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/Lightning-2.1-792EE5.svg?logo=lightning&logoColor=white)](https://lightning.ai/)
[![pytorch-lifestream](https://img.shields.io/badge/pytorch--lifestream-0.7-4B8BBE.svg)](https://github.com/dllllb/pytorch-lifestream)
[![Datasets on HF](https://img.shields.io/badge/%F0%9F%A4%97%20datasets-pytorch--lifestream-FFD21E.svg)](https://huggingface.co/pytorch-lifestream)

This project explores knowledge transfer between LLMs (large language models) and specialized models for structured (transactional) data. LLMs can reason over serialized event logs and generate rich explanations, but are slow and costly at inference. Structured models are efficient and accurate, but lack interpretive depth.

We study two directions of knowledge transfer: distilling LLM outputs (pseudo-labels, rationales) into structured models, and enriching LLM prompts with embeddings and statistics from structured models as chain-of-thought context.

We assume the general research questions:

**RQ1:** Does bidirectional knowledge transfer between LLMs and structured models improve transaction classification performance compared to either approach alone?

**RQ2:** Which teacher signal and transfer method works best in each direction?

**RQ3:** How do LLM size and CoT reasoning ability affect transfer quality in both directions?



<p align="center">
  <img src="images/method.png" width="50%" alt="LATTE + Symmetric KL" />
</p>


## Datasets

| Dataset  | Task                | Metric    | Clients | Transactions |
|----------|---------------------|-----------|---------|--------------|
| Gender   | Binary (gender)     | ROC-AUC   | 8 400   | 6.85M        |
| Rosbank  | Binary (churn)      | ROC-AUC   | 5 000   | 1.0M         |
| Age      | 4-class (age group) | Accuracy  | 30 000  | 27M          |


## Quick Start

Experiments were run on RTX 3090 24GB

```bash
pip install -e .

# Pre-download all three datasets
bash scripts/download_data.sh
```

The bidirectional method uses checkpoints from earlier stages, so the four stages must be run in order.

```bash
# CoLES baseline
python experiments/rq1_bidirectional/coles/run_gender_coles.py
python experiments/rq1_bidirectional/coles/run_rosbank_coles.py
python experiments/rq1_bidirectional/coles/run_age_coles.py

# LLM4ES
python experiments/rq2_d1_teacher_signals/feature_based/gender_llm4es.py
python experiments/rq2_d1_teacher_signals/feature_based/rosbank_llm4es.py
python experiments/rq2_d1_teacher_signals/feature_based/age_llm4es.py

# LATTE
python experiments/rq1_bidirectional/latte/E1_2_gender_latte.py
python experiments/rq1_bidirectional/latte/E1_2_rosbank_latte.py
python experiments/rq1_bidirectional/latte/E1_2_age_latte.py

# Bidirectional LATTE + Symmetric KL without LoRA
python experiments/rq1_bidirectional/latte_mutual_kl/gender_mutual_kl.py
python experiments/rq1_bidirectional/latte_mutual_kl/rosbank_mutual_kl.py
python experiments/rq1_bidirectional/latte_mutual_kl/age_mutual_kl.py
```

## Main results

Comparison of knowledge transfer directions

| Method                              | Direction      | Gender     | Rosbank    | Age        |
|-------------------------------------|----------------|------------|------------|------------|
| CoLES baseline                      | Unidirectional | 0.8626     | 0.8054     | 0.6345     |
| LATTE                               | Unidirectional | 0.8674     | 0.8057     | **0.6429** |
| LATTE + Symmetric KL (with LoRA)    | Bidirectional  | 0.8713     | 0.8122     | 0.6313     |
| LATTE + Symmetric KL (without LoRA) | Bidirectional  | **0.8774** | **0.8192** | 0.6397     |

