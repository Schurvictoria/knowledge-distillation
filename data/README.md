# Data

Сырые транзакции и labels для трёх датасетов. Скачиваются автоматически из HuggingFace при запуске CoLES baseline скриптов.

## После загрузки должны лежать

```
data/
├── transactions.csv                # Gender — 6.85M транзакций
├── gender_train.csv                # Gender labels (8.4K клиентов)
├── transactions_train.csv          # Age — 27M транзакций
├── train_target.csv                # Age labels (30K клиентов)
└── rosbank_train.csv               # Rosbank — 1M транзакций + labels (5K клиентов)
```

## Источники

- Gender: `huggingface.co/datasets/pytorch-lifestream/transactions-gender`
- Age: `huggingface.co/datasets/pytorch-lifestream/age-group-prediction`
- Rosbank: `huggingface.co/datasets/pytorch-lifestream/rosbank-churn`

## Как скачать

Запуск любого CoLES baseline тянет данные автоматически:
```bash
python experiments/rq1_bidirectional/coles/run_gender_coles.py    # → transactions.csv + gender_train.csv
python experiments/rq1_bidirectional/coles/run_rosbank_coles.py   # → rosbank_train.csv
python experiments/rq1_bidirectional/coles/run_age_coles.py       # → transactions_train.csv + train_target.csv
```

## Ignored в git

Все CSV-файлы → в `.gitignore` (по 100MB-1GB каждый). Структура коммитится через `.gitkeep`.
