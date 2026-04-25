# Results

Сюда пишутся результаты экспериментов. Структура простая — четыре папки по RQ:

```
results/
├── rq1/        # E1.x — bidirectional vs unidirectional
├── rq2_d1/     # E2.x — какой teacher signal лучше (LLM → Structured)
├── rq2_d2/     # E3.x + E4.x — какой prompt enrichment лучше (Structured → LLM)
└── rq3/        # E5.x + E6.x + E7.x — размер LLM влияет?
```

Внутри каждой RQ-папки скрипт создаёт свою подпапку `<experiment_id>/` и пишет туда:

- `result.json` — единая схема (метрики, конфиг, git_commit, runtime). Коммитится в git.
- `*.pt` — checkpoints PyTorch. **Не** коммитятся (gitignored).
- `*.npy` / `*.npz` — pre-computed embeddings/predictions. Не коммитятся.
- `*.csv` — детали по сидам, если эксперимент multi-seed.

Пример после запуска E1.2 на gender:

```
results/rq1/E1_2_gender/
├── result.json                         # коммитится
├── coles_baseline.pt                   # gitignored
├── coles_finetuned_alpha0.1.pt         # gitignored
└── true_latte_results.csv              # коммитится
```

## Как обновить REPORT.md из result.json

```bash
make aggregate
# создаст REPORT_GENERATED.md со всеми таблицами по RQ
```

Скрипт `scripts/aggregate_results.py` обходит `results/**/result.json`, группирует по RQ и рендерит markdown-таблицы.

## Что было раньше (legacy)

В предыдущих версиях repo результаты лежали в плоских папках типа `results/gender_true_latte/`, `results/age_llm4es/` и т.д. (по одной на каждый эксперимент). Существующие скрипты пишут туда checkpoints — папки создаются на лету через `Path(...).mkdir(parents=True, exist_ok=True)`. Они появятся при первом запуске любого эксперимента.
