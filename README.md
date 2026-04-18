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

RQ1: Какой метод передачи знаний из LLM в structured models наиболее эффективен — knowledge distillation (response/feature/relation-based), feature augmentation (LLM embeddings как фичи), CoT distillation или их комбинация?

RQ2: Как размер и качество LLM-teacher'а влияют на результат передачи знаний в табличные модели?

RQ3: Какой тип учительского сигнала наиболее эффективен при дистилляции: response-based (soft labels), feature-based (embedding augmentation) или relation-based (contrastive alignment)?

RQ4: Какая архитектура ученика лучше усваивает знания от LLM — gradient boosting или sequence encoder?

### Direction 2: Structured Models → LLM

RQ5: Какой тип знания от structured model наиболее эффективен для улучшения предсказаний LLM — model predictions, feature-level explanations, instance-level retrieval или structured reasoning?

RQ6: Как стратегия промптинга (zero-shot, few-shot, CoT) взаимодействует с типом обогащения от structured model?

RQ7: Как масштаб и архитектура LLM влияют на эффективность обогащения промптов сигналами от structured models?



We assume the general research questions:
RQ8: Помогает ли CoT reasoning в обоих направлениях knowledge transfer?

And we assume following research questions in this project:

## 1. LLMs -> Structure models
RQ1: Улучшает ли дистилляция из LLM качество structured models?
RQ2: Влияет ли сила LLM-teacher'а на качество дистилляции?
RQ3: Важен ли CoT reasoning teacher'а при дистилляции?
RQ4: Какой ученик лучше усваивает — бустинг или sequence encoder?

 
## 1. Structure models -> LLMs

RQ5: Какой тип сигнала от structured model лучше помогает LLM?
RQ6: Какая стратегия промптинга лучше при обогащении?
RQ7: Зависит ли эффект обогащения от силы LLM?


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
