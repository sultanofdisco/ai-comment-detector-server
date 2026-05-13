# Saved Models

This repo expects two saved model folders.

## 1. Stage 1

Path:

- `saved_models/stage1_kcbert_binary/`
- `saved_models/hybrid_current/` (preferred when trained)

Required files:

- `artifacts.json`
- `label_map.json`
- `metrics.json`
- `bert_classifier/`

Additional required file for hybrid stage 1:

- `xgb_model.joblib`

Notes:

- The application prefers `saved_models/hybrid_current/` for stage 1 when that
  folder contains trained artifacts.
- Large trained artifact folders are expected to be generated locally rather
  than committed on every branch. Recent stage-1 hybrid folders are around
  `1.6 GB`.
- To reproduce the main-server stage-1 artifact, run
  `training/train_stage1_hybrid.py` with `--output-dir saved_models/hybrid_current`
  and `--force-text-weight 0.8`.

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
