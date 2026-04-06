# AI Comment Detector Server

This repo now follows a two-stage pipeline.

## Pipeline

1. Stage 1:
   KC-BERT binary classifier
   input: transformed comment text
   output: `human` or `ai`

2. Stage 2:
   AI-only hybrid classifier
   input: comment text
   run only when stage-1 `ai_score >= 0.85`
   output: one of `gpt`, `claude`, `gemini`, `deepseek`

The previous all-in-one `human + llm` 5-class path has been removed.

## Request format

```json
{
  "comment_id": "c1",
  "author_id": "user123",
  "text": "comment text",
  "url": "https://x.com/...",
  "timestamp": "2026-03-30T12:00:00Z"
}
```

## Response format

If stage 2 runs:

```json
{
  "comment_id": "c1",
  "pred_label": "gpt",
  "confidence": 0.91,
  "ai_score": 0.93,
  "top2": [
    ["gpt", 0.91],
    ["gemini", 0.06]
  ],
  "risk_level": "high",
  "reason": "stage1 ai probability 0.93 passed threshold 0.85; stage2 predicted gpt"
}
```

If stage 2 does not run:

```json
{
  "comment_id": "c1",
  "pred_label": "ai",
  "confidence": 0.73,
  "ai_score": 0.73,
  "top2": [
    ["ai", 0.73],
    ["human", 0.27]
  ],
  "risk_level": "medium",
  "reason": "stage1 kcbert predicted ai with probability 0.73, below llm threshold 0.85"
}
```

## Stage-1 training

Default dataset columns:

- text: `transformed_text`
- label: `model_source`
- split: `split`

Command:

```bash
python training/train_kcbert_binary.py ^
  --data-path "C:\Users\user\Downloads\dataset2_team_all_merged.csv" ^
  --output-dir saved_models/stage1_kcbert_binary ^
  --text-col transformed_text ^
  --label-col model_source ^
  --split-col split
```

## Stage-2 training

Default dataset columns:

- text: `reply_text`
- label: `model_source`
- split: `split`

Command:

```bash
python training/train_hybrid.py ^
  --data-path "C:\Users\user\Downloads\dataset2_team_all_merged.csv" ^
  --output-dir saved_models/stage2_llm_hybrid ^
  --text-col reply_text ^
  --label-col model_source ^
  --split-col split
```

Stage 2 trains only on:

- `gpt`
- `claude`
- `gemini`
- `deepseek`

## Server run

```bash
uvicorn server.app:app --reload
```

## Main files

- `training/train_kcbert_binary.py`: stage-1 human vs ai trainer
- `training/train_hybrid.py`: stage-2 AI-family hybrid trainer
- `server/preprocess.py`: transform rules and stats feature extraction
- `server/inference.py`: two-stage inference pipeline
- `server/app.py`: FastAPI routes
- `docs/api-spec-draft.md`: API draft
