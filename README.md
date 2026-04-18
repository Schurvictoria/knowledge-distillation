# Bidirectional Knowledge Distillation between LLM and Sequence Models for Event Sequences

This project explores knowledge transfer between LLMs (large language models) and specialized models for structured (transactional) data. LLMs can reason over serialized event logs and generate rich explanations, but are slow and costly at inference. Structured models are efficient and accurate, but lack interpretive depth.

We study two directions of knowledge transfer: distilling LLM outputs (pseudo-labels, rationales) into structured models, and enriching LLM prompts with embeddings and statistics from structured models as chain-of-thought context.

We assume the general research questions:

RQ1: Does bidirectional knowledge transfer between LLMs and structured models improve transaction classification performance compared to either approach alone?

RQ2: Which teacher signal and transfer method works best in each direction?

RQ3: How do LLM size and CoT reasoning ability affect transfer quality in both directions?


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
