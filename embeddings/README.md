# Embeddings

CoLES baseline embeddings — продукт E1.1. Используются всеми последующими RQ как input.

## Структура

```
embeddings/
├── gender/
│   ├── emb_train_seed42.npy        # CoLES embeddings train
│   ├── emb_test_seed42.npy         # CoLES embeddings test
│   ├── cids_train_seed42.npy       # client_id порядок (для alignment с LLM4ES)
│   ├── cids_test_seed42.npy
│   ├── y_train_seed42.npy          # ground-truth labels
│   └── y_test_seed42.npy
├── rosbank/
│   └── (same structure)
└── age/
    └── (same structure)
```

## Кто пишет

`experiments/rq1_bidirectional/coles/run_{gender,rosbank,age}_coles.py`

## Кто читает

Практически все скрипты в RQ2 / RQ3:
- LATTE (E1.2) — для contrastive alignment
- mutual_KL (E1.3) — то же
- RAMD step1 (E1.5) — для kNN
- LLM4ES concat (E2.2) — для concatenation с LLM embeddings
- response-based (E2.1) — для downstream
- by_enrichment_type (E3.x) — для kNN retrieval
- E5_x extract — для alignment

## Ignored в git

`.npy` файлы крупные → в `.gitignore`. Структура папок (с `.gitkeep`) коммитится.
