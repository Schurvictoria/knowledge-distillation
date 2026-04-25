# Bidirectional Knowledge Distillation between LLM and Sequence Models for Event Sequences

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/Lightning-2.1-792EE5.svg?logo=lightning&logoColor=white)](https://lightning.ai/)
[![pytorch-lifestream](https://img.shields.io/badge/pytorch--lifestream-0.7-4B8BBE.svg)](https://github.com/dllllb/pytorch-lifestream)
[![Datasets on HF](https://img.shields.io/badge/%F0%9F%A4%97%20datasets-pytorch--lifestream-FFD21E.svg)](https://huggingface.co/pytorch-lifestream)

This project explores knowledge transfer between LLMs (large language models) and specialized models for structured (transactional) data. LLMs can reason over serialized event logs and generate rich explanations, but are slow and costly at inference. Structured models are efficient and accurate, but lack interpretive depth.

We study two directions of knowledge transfer: distilling LLM outputs (pseudo-labels, rationales) into structured models, and enriching LLM prompts with embeddings and statistics from structured models as chain-of-thought context.

## Research Questions

**RQ1:** Does bidirectional knowledge transfer between LLMs and structured models improve transaction classification performance compared to either approach alone?

**RQ2:** Which teacher signal and transfer method works best in each direction?

**RQ3:** How do LLM size and CoT reasoning ability affect transfer quality in both directions?

Final results: [REPORT.md](./REPORT.md). Number → script mapping: [EXPERIMENTS_MAP.md](./EXPERIMENTS_MAP.md).

---

## Project Structure

Думай о репозитории как о кухне:

```
src/distil/            ингредиенты на кухне (мука, яйца, сахар)
experiments/           рецепты (E1.2 gender = торт, E2.4 combined = пирог)
results/               готовые блюда (мерим успех)
data/                  сырые продукты с рынка (HuggingFace)
embeddings/            заготовки которые используются в нескольких рецептах
scripts/               кухонные инструменты (миксер, весы)
logs/                  записи "как готовил, что пошло не так"
```

### Зачем три уровня (`src/`, `experiments/`, `results/`)

| Слой | Что внутри | Запускается? |
|---|---|---|
| `src/distil/` | **Библиотека** — переиспользуемые функции (`load_gender_dataset`, `train_coles_baseline`, `evaluate_lgbm`). Не запускается сама по себе — только импортируется. | нет |
| `experiments/` | **Точки входа** — каждый файл соответствует одному эксперименту из REPORT.md (E1.1, E1.2 …). Импортирует блоки из `src/distil/`. | да: `python experiments/.../run_*.py` |
| `results/` | **Артефакты** — `result.json`, чекпоинты `.pt`, эмбеддинги `.npy`. Создаются скриптами при запуске. | нет |

Без `src/distil/` каждый из 44 экспериментов имел бы свою копию `build_records()`, `evaluate_downstream()` (~200 строк копипасты × 44 = 8800 строк дубликатов). Один баг → во всех 44.
Без `experiments/` нужен был бы монстр-скрипт `run.py` с 30+ if-веток по экспериментам.

---

## Detailed Layout

```
knowledge-distillation/
│
├── src/distil/                              # БИБЛИОТЕКА (импортируется из experiments/)
│   ├── data/                                # Загрузчики датасетов
│   │   ├── gender.py                        # load_gender_dataset(seed=42) → GenderDataset
│   │   ├── rosbank.py                       # load_rosbank_dataset(seed=42)
│   │   ├── age.py                           # load_age_dataset(seed=42)
│   │   ├── _downloads.py                    # auto-download с HuggingFace
│   │   └── embeddings_io.py                 # save/load CoLES embeddings
│   ├── models/
│   │   ├── coles_baseline.py                # ColesConfig, build_encoder, train, extract
│   │   ├── latte.py                         # LATTE Phase 2 finetune (contrastive)
│   │   ├── mutual_kl.py                     # KL-divergence helpers (E1.3)
│   │   └── ramd.py                          # RAMD step1+step2 helpers
│   ├── llm/
│   │   ├── transaction_serializer.py        # transactions → text
│   │   ├── openrouter_client.py             # API + budget tracker
│   │   ├── local_inference.py               # 4-bit Qwen/Gemma loader
│   │   └── prompt_templates.py              # zero-shot / few-shot / CoT
│   ├── downstream/
│   │   └── classifiers.py                   # evaluate_{lgbm,xgboost,logreg}
│   ├── reproducibility/
│   │   ├── seeding.py                       # seed_everything(42)
│   │   └── input_checks.py                  # require_raw_data(...) и т.д.
│   └── results/
│       ├── schema.py                        # ExperimentResult dataclass
│       └── writer.py                        # save_experiment_result(...) → result.json
│
├── experiments/                             # СКРИПТЫ ЗАПУСКА (entry points)
│   ├── rq1_bidirectional/
│   │   ├── coles/                           # E1.1 — CoLES baseline (3 datasets)
│   │   ├── latte/                           # E1.2 — LATTE
│   │   │   └── ablations/                   # alpha sweep + variants
│   │   ├── latte_mutual_kl/                 # E1.3 — bidirectional (CoLES ↔ LLM)
│   │   │   └── ablations/                   # early-stop variant
│   │   └── ramd/                            # E1.4 + E1.5 — RAMD loop
│   ├── rq2_d1_teacher_signals/
│   │   ├── response_based/                  # E2.1 — soft labels (Reverse KL)
│   │   ├── feature_based/                   # E2.2 — LLM4ES embedding concat
│   │   └── combined/                        # E2.4 — all three signals
│   ├── rq2_d2_prompt_enrichment/
│   │   ├── by_enrichment_type/              # E3.1-E3.5 — None/Pred/SHAP/kNN/Both
│   │   ├── by_strategy/                     # E4.1-E4.3 — zero/few/CoT × enrichment
│   │   └── by_model/                        # E6.x + E7.x — runners для разных LLM
│   ├── rq3_llm_size_effect/                 # E5.3 + E6.1 — размер LLM влияет?
│   └── shared/                              # отключённые скрипты (disabled)
│
├── results/                                 # АРТЕФАКТЫ (создаются скриптами)
│   ├── rq1/                                 # result.json для всех E1.x
│   ├── rq2_d1/                              # result.json для всех E2.x
│   ├── rq2_d2/                              # result.json для всех E3.x, E4.x, E6.x, E7.x
│   ├── rq3/                                 # result.json для E5.3, E6.1
│   ├── gender_coles/, age_coles/, rosbank_coles/                # checkpoints (E1.1)
│   ├── gender_true_latte/, age_true_latte/, rosbank_true_latte/ # checkpoints (E1.2)
│   └── ...
│
├── scripts/                                 # КУХОННЫЕ ИНСТРУМЕНТЫ
│   ├── aggregate_results.py                 # results/**/*.json → REPORT.md tables
│   └── verify_environment.py                # pre-flight check (data/embeddings/GPU)
│
├── data/                                    # СЫРЫЕ ДАННЫЕ (gitignored, скачиваются)
├── embeddings/                              # ЗАГОТОВКИ — CoLES embeddings per dataset
├── logs/                                    # ЗАПИСИ — overnight run logs
│
├── README.md                                # этот файл
├── REPORT.md                                # canonical results
├── EXPERIMENTS_MAP.md                       # E#.# → script
├── Makefile                                 # one-command-per-RQ
├── environment.yml                          # conda env с фикcированными версиями
└── pyproject.toml                           # установка `pip install -e .` для distil
```

---

## Quick Start

```bash
# 1. Окружение
conda env create -f environment.yml
conda activate knowledge-distillation
pip install -e .                            # делает distil библиотекой

# 2. Один эксперимент
python experiments/rq1_bidirectional/coles/run_gender_coles.py
# Auto-скачает data/transactions.csv с HF, обучит CoLES, сохранит embeddings

# 3. Серия по RQ
make rq1                                    # все E1.x подряд
make rq2-d1                                 # все E2.x
make aggregate                              # обновляет REPORT_GENERATED.md
```

---

## Datasets

| Dataset | Task | Metric | Train | Test | Source |
|---|---|---|---|---|---|
| Gender | Binary (M/F) | ROC-AUC | 7397 | 818 | [pytorch-lifestream/transactions-gender](https://huggingface.co/datasets/pytorch-lifestream/transactions-gender) |
| Rosbank | Binary (churn) | ROC-AUC | 3967 | 437 | [pytorch-lifestream/rosbank-churn](https://huggingface.co/datasets/pytorch-lifestream/rosbank-churn) |
| Age | 4-class | Accuracy | 27000 | 3000 | [pytorch-lifestream/age-group-prediction](https://huggingface.co/datasets/pytorch-lifestream/age-group-prediction) |

Train/test split: 90/10 stratified, seed=42 (везде одинаково для совместимости checkpoints).

---

## Main Results (single seed=42)

### Direction 1: LLM → Structured Model (RQ1)

| Method | Gender (AUC) | Rosbank (AUC) | Age (Acc) |
|---|---|---|---|
| CoLES baseline (E1.1) | 0.8626 | 0.8054 | 0.6345 |
| LATTE (E1.2) | **0.8674** | 0.8057 | **0.6429** |
| LATTE + mutual KL (E1.3) | **0.8676** | **0.8142** | 0.6363 |
| RAMD Qwen2.5-7B (E1.4) | 0.8630 | 0.8074 | — |
| RAMD DeepSeek (E1.5) | 0.8630 ± 0.0006 | 0.8072 ± 0.0034 | OOF running |

### Direction 2: Structured → LLM (RQ2 D2)

| Strategy | Gender (AUC) | Rosbank (AUC) | Δ vs zero-shot |
|---|---|---|---|
| LLM zero-shot (E3.1) | 0.498 | 0.499 | — |
| **kNN CoT (E3.4)** | **0.762** | **0.766** | **+26-27 pp** |

**Главный finding:** structured→LLM (kNN CoT, +26pp) >>> LLM→structured (LATTE, +0.5pp). Асимметрия.

---

## Reproducibility

- **Seed=42** везде: `seed_everything()` сидирует random + numpy + torch + cuda + Lightning + cudnn.deterministic
- **Train/test split** одинаковый во всех скриптах (test_size=0.1, random_state=42, stratified)
- **API calls** к OpenRouter: `seed=42` в payload, температура 0 для детерминированных моделей
- **Checkpoints** сохраняются в `results/{dataset}_*/` — можно посмотреть finetuned encoder
- **`result.json`** для каждого эксперимента содержит git_commit + torch_version + ptls_version

См. подробности в [`EXPERIMENTS_MAP.md`](./EXPERIMENTS_MAP.md).

---

## Tech Stack

- **pytorch-lifestream** — CoLES, TrxEncoder, RnnSeqEncoder
- **Qwen2.5-3B/7B** — LLM inference (4-bit NF4) + LLM4ES fine-tuning (QLoRA)
- **LightGBM / XGBoost** — downstream classifiers на embeddings
- **OpenRouter** — DeepSeek-V3.2, Qwen3.6-35B, GLM-4.7 для RQ3
- **RTX 3090 / vast.ai** — GPU training
