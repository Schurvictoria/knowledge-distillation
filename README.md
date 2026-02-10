# Knowledge Distillation

Pseudo-label distillation from LLM to XGBoost/CatBoost on bank transaction datasets.

## Quick Start

```bash
# Install dependencies
uv venv && uv pip install -e ".[dev]"

# Run with mock LLM (no API key needed)
uv run python scripts/run_experiment.py --dataset gender --use-mock

# Run with local vLLM
uv run python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ --port 8000
uv run python scripts/run_experiment.py \
    --dataset gender \
    --llm-model "openai/Qwen/Qwen2.5-7B-Instruct-AWQ" \
    --api-base "http://localhost:8000/v1"

# Run with OpenAI API
export OPENAI_API_KEY=sk-...
uv run python scripts/run_experiment.py --dataset gender --llm-model gpt-4o-mini
```

## Project Structure

```
src/distil/
    data.py            # Dataset loading (Gender, Rosbank) from HuggingFace
    text_convert.py    # Transactions -> text for LLM (MCC code mapping)
    pseudo_labels.py   # LLM API calls with caching, guided JSON, CoT prompting
    train.py           # XGBoost/CatBoost training (variants a/b/c/d)
    evaluate.py        # ROC-AUC with bootstrap CI, comparison tables
scripts/
    run_experiment.py  # Full pipeline CLI
results/               # CSV results per dataset
REPORT.md              # Detailed experiment report
```

## CLI Options

```
--dataset       gender|rosbank          Dataset to use
--model-type    xgboost|catboost|both   Student model type
--llm-model     gpt-4o-mini             LLM model name (litellm format)
--api-base      http://localhost:8000/v1 Local vLLM server URL
--use-mock                              Use mock LLM (for testing)
--max-clients   N                       Limit clients (faster runs)
--output-dir    results                 Output directory
--seed          42                      Random seed
```

## Experiment Variants

| Variant | Description |
|---------|-------------|
| (a) baseline | Aggregated transaction features only |
| (b) with_pseudo | Features + LLM pseudo-label + probability |
| (c) with_tfidf | Features + TF-IDF of LLM text explanations |
| (d) pseudo_only | Only LLM predictions (measures LLM quality) |
| (e) llm_raw | Raw LLM probability (no model training) |

## Datasets

Downloaded automatically from HuggingFace on first run:
- `pytorch-lifestream/transactions-gender` (72MB)
- `pytorch-lifestream/rosbank-churn` (9MB)

## Requirements

- Python 3.10+
- GPU with 16GB VRAM (for local vLLM inference)
- Or OpenAI/Anthropic API key

## Results

See [REPORT.md](REPORT.md) for detailed results and analysis.
