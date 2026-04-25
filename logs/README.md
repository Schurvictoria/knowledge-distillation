# Logs

Лог-файлы overnight runs и долгих экспериментов.

## Naming convention

```
logs/
├── overnight_2026-04-26.log           # полный stdout overnight скрипта
├── overnight_2026-04-26.err           # stderr (если разделено)
├── E1_2_gender_seed42.log             # лог одного эксперимента
├── E5_x_qwen25_3b_extract_age.log     # E5.3 extract на Age
└── ...
```

## Что писать в логи

- timestamp + experiment_id в начале
- GPU info (`nvidia-smi`)
- pytorch + ptls версии
- Прогресс по эпохам (для больших exp)
- Финальные метрики

## Idempotency

Overnight скрипт пишет лог в режиме `>>` (append) чтобы при повторных запусках видеть полную историю. Каждая итерация маркируется `=== <timestamp> START <experiment> ===`.

## Ignored в git

`.log`/`.out`/`.err` → в `.gitignore`. README + структура коммитятся.
