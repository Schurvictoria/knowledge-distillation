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


You can see the results in [REPORT.md](./REPORT.md)

As a result we suggest RAMD


## Architecture

## Quick Start

## Datasets

## Main results

## Project Structure

## Experiment Variants

## Datasets

## Requirements

## Results
