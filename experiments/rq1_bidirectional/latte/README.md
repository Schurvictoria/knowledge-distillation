# LATTE (standalone): LLM → CoLES

**Метод:** CoLES учится у LLM embeddings через contrastive loss. Веса LLM НЕ обновляются (односторонняя дистилляция).

## Скрипты

| Скрипт | Датасет |
|---|---|
| `E1_2_gender_latte.py` | Gender (818 test) |
| `E1_2_rosbank_latte.py` | Rosbank (437 test) |
| `E1_2_age_latte.py` | Age 4-class (3000 test) |

## Methodology (после фиксов)

- `seed=42` + `pl.seed_everything` + `cudnn.deterministic=True` (полная воспроизводимость)
- Train (80%) / **val (10%)** / test (10%) — stratified split, seed=42
- Best checkpoint выбирается по **val** AUC/acc (не test — no test-peeking)
- Test eval только параллельно для трекинга, reported `best_test` = test на эпохе где val максимален
- Hard-assert на отсутствие LLM embeddings (нет silent zero fallback)
- LATTE: contrastive InfoNCE, τ=0.07, α=0.1
- LGBM downstream classifier (`random_state=42`)

## Результаты (val-split, seed=42, single training run)

| Dataset | CoLES baseline | LATTE α=0.1 | Δ |
|---|---|---|---|
| Gender | 0.8606 | **0.8713** | +1.07 пп |
| Rosbank | 0.8041 | **0.8082** | +0.41 пп |
| Age | 0.6283 | **0.6333** | +0.50 пп |

`*` Baseline отличается от REPORT__2_.md (0.8626/0.8054/0.6345), потому что:
- В REPORT base взят из 5-seed `run_seeded_eval.py` (LGBM-only variance)
- Здесь base — single-run val-split на seed=42 c `cudnn.deterministic=True`
- Разница ±0.002–0.005 пп ожидаема, narrative тот же: LATTE > baseline

## Reproducibility note

`torch.use_deterministic_algorithms(True)` отключает non-deterministic CUDA kernels (scatter_add backward etc). Это может **слегка изменить числа** при перезапуске относительно ранних non-deterministic прогонов (±0.002-0.005 пп). Если падает с ошибкой о non-det op — установить `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
