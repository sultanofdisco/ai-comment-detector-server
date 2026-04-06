# Saved Models

This repo expects two saved model folders.

## 1. Stage 1

Path:

- `saved_models/stage1_kcbert_binary/`

Required files:

- `artifacts.json`
- `label_map.json`
- `metrics.json`
- `bert_classifier/`

## 2. Stage 2

Path:

- `saved_models/stage2_llm_hybrid/`

Required files:

- `artifacts.json`
- `label_map.json`
- `metrics.json`
- `xgb_model.joblib`
- `bert_classifier/`

The server loads stage 1 first, then conditionally uses stage 2.
