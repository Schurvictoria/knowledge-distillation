# CoLES baseline — reference outputs

Layout after a successful run (per dataset):

```
results/
  gender/
    gender_summary.json         # config + emb_dims + lgbm_config + seeds + elapsed + date
    gender_coles_per_seed.csv   # columns: seed, model, roc_auc
  age/
    age_summary.json
    age_coles_per_seed.csv      # columns: seed, model, accuracy
  rosbank/
    rosbank_summary.json
    rosbank_coles_per_seed.csv  # columns: dataset, model, variant, seed, roc_auc, accuracy, f1
    rosbank_coles_aggregated.csv
```

The `*_summary.json` files checked in here are committed as **format references** — config values match the scripts, metrics are not included (they land in the CSV after the run). Per-seed CSVs appear only after running the corresponding script.

Reference lgbm metrics reported in [../../../../REPORT.md](../../../../REPORT.md) (table E1.1):

| Dataset  | Metric   | CoLES baseline |
|----------|----------|----------------|
| Gender   | ROC-AUC  | 0.8626         |
| Rosbank  | ROC-AUC  | 0.8054         |
| Age      | Accuracy | 0.6345         |
