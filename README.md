# Bidirectional Knowledge Distillation between LLM and Sequence Models for Event Sequences

This project explores knowledge transfer between LLMs (large language models) and specialized models for structured (transactional) data. LLMs can reason over serialized event logs and generate rich explanations, but are slow and costly at inference. Structured models are efficient and accurate, but lack interpretive depth.

We study two directions of knowledge transfer: distilling LLM outputs (pseudo-labels, rationales) into structured models, and enriching LLM prompts with embeddings and statistics from structured models as chain-of-thought context.

We assume the general research questions:

- Зависит ли оптимальный способ инъекции знаний от стратегии промптинга (zero-shot, few-shot, CoT) и существует ли взаимодействие между типом инъекции и стратегией?
- Как размер LLM влияет на качество knowledge transfer в обоих направлениях и существует ли порог размера, ниже которого дистилляция не даёт прироста?
- Улучшает ли CoT reasoning качество учительского сигнала при дистилляции и при каких условиях CoT-rationales полезны как дополнительный supervisory сигнал?

And we assume other questions:
## 1. LLMs -> Structure models
- Какой метод передачи знаний от LLM в structured models эффективнее — training-time (дистилляция: soft labels, contrastive alignment) или inference-time (feature augmentation: LLM embeddings как доп. признаки)?
- Какая архитектура ученика лучше усваивает знания от LLM — gradient boosting (XGBoost), sequence encoder (CoLES) или tabular foundation model (TabPFN)?
 
## 1. Structure models -> LLMs
- Какой метод передачи знаний от structured model в LLM эффективнее — training-time (fine-tune LLM на транзакциях) или inference-time (обогащение промптов: SHAP, kNN retrieval)?


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
