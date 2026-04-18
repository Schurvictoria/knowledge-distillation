# Bidirectional Knowledge Distillation between LLM and Sequence Models for Event Sequences

This project explores knowledge transfer between LLMs (large language models) and specialized models for structured (transactional) data. LLMs can reason over serialized event logs and generate rich explanations, but are slow and costly at inference. Structured models are efficient and accurate, but lack interpretive depth.

We study two directions of knowledge transfer: distilling LLM outputs (pseudo-labels, rationales) into structured models, and enriching LLM prompts with embeddings and statistics from structured models as chain-of-thought context.

Задачи проекта:
"- Провести обзор работ по knowledge distillation и hybrid LLM+tabular моделям.
 - Разработать pipeline дистилляции знаний LLM → XGBoost/CoLES/LATTE.
 - Разработать подход обогащения промптов LLM сигналами от структурированных моделей (структурированная CoT-подсказка).
 - Оценить полученные методы на задаче предсказания churn/классификации пользователей по транзакциям.
 - [опц.] Сравнить эффективность в zero-shot и fine-tuning режимах.
 - [опц.] Подготовить научную публикацию.

 - 
## 1. Structure models -> LLMs

Here we assume the following research questions:

RQ5: Какой тип сигнала от structured model лучше помогает LLM?

| Тип сигнала | Что LLM получает | Gender | Rosbank | Age |
|-------------|-----------------|--------|---------|-----|
| Нет | Только профиль клиента | 0.498 | 0.499 | 0.249 |
| Prediction | "ML model says: male (73%)" | ? | ? | ? |
| Explanation | "Top factors: Retail high" (SHAP) | 0.606 | 0.637 | ? |
| Retrieval | "7/10 neighbors = male" (kNN) | 0.762 | 0.766 | 0.250 |
| Все вместе | Prediction + SHAP + kNN | 0.745 | 0.751 | ? |

---

RQ6: Какая стратегия промптинга лучше работает с обогащением?

| Стратегия | Без обогащения | + SHAP | + kNN | + оба |
|-----------|---------------|--------|-------|-------|
| Zero-shot | 0.498 | ? | ? | ? |
| Few-shot | 0.578 | ? | ? | ? |
| CoT | ? | 0.606 | 0.762 | 0.745 |

---

RQ7: Влияет ли сила LLM на эффективность обогащения промптов?

| LLM | Размер | Без обогащения | + kNN CoT | Δ |
|-----|--------|---------------|-----------|---|
| Gemma 3n E2B | 2B | ? | ? | ? |
| Qwen2.5-7B | 7B | 0.498 | 0.762 | +26 пп |
| Qwen3.6-35B | 35B | ? | ? | ? |
| DeepSeek-R1 | ~70B | ? | ? | ? |
| GPT-4o | ~200B | ? | ? | ? |

---

RQ8: Влияет ли CoT reasoning LLM на качество при обогащении?

| LLM | Thinking | Без обогащения | + kNN CoT |
|-----|----------|---------------|-----------|
| Qwen3.6-35B | off | ? | ? |
| Qwen3.6-35B | on | ? | ? |
| DeepSeek-R1 | off | ? | ? |
| DeepSeek-R1 | on | ? | ? |


We ass

## Architecture

## Quick Start

## Datasets

## Main results


## Project Structure

## Experiment Variants

## Datasets

## Requirements

## Results
