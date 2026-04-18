# Bidirectional Knowledge Distillation between LLM and Sequence Models for Event Sequences

This project explores knowledge transfer between LLMs (large language models) and specialized models for structured (transactional) data. LLMs can reason over serialized event logs and generate rich explanations, but are slow and costly at inference. Structured models are efficient and accurate, but lack interpretive depth.

We study two directions of knowledge transfer: distilling LLM outputs (pseudo-labels, rationales) into structured models, and enriching LLM prompts with embeddings and statistics from structured models as chain-of-thought context.

We assume the general research questions:

- Does the best way to transfer knowledge depend on how the LLM is prompted (zero-shot, few-shot, CoT)?
- Does LLM size affect transfer quality in both directions?
- Does chain-of-thought reasoning improve the teacher signal?

And we assume other questions:
## 1. LLMs -> Structure models
- Which approach works better for transferring LLM knowledge — training-time distillation (soft labels, contrastive alignment) or inference-time feature augmentation (LLM embeddings as extra input features)?
- Which student architecture learns best from LLM supervision — gradient boosting (XGBoost/CatBoost), sequence encoder (CoLES), or tabular foundation model (TabPFN)?
 
## 1. Structure models -> LLMs
- Which approach works better for transferring structured model knowledge to the LLM — fine-tuning on transaction sequences, or enriching prompts with SHAP explanations and kNN retrieval context?

As a result

## Architecture

## Quick Start

## Datasets

## Main results

## Project Structure

## Experiment Variants

## Datasets

## Requirements

## Results
