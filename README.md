# AI Comment Detector Server

This repo now follows a two-stage pipeline.

## Pipeline

1. Stage 1:
   KC-BERT + stats hybrid binary classifier
   input: comment text (+ optional parent post text for consistency stats)
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
  "post_text": "original post text",
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

## Stage-1 hybrid training

Preferred command for the plugin server:

```bash
python training/train_stage1_hybrid.py ^
  --data-path "C:\Users\user\Downloads\dataset2_team_all_merged.csv" ^
  --output-dir saved_models/hybrid_current ^
  --model-name saved_models/stage1_kcbert_binary/bert_classifier ^
  --text-col transformed_text ^
  --raw-text-col reply_text ^
  --post-col post_text ^
  --label-col model_source ^
  --split-col split
```

Main-server deployment target (leading reply mentions stripped, body mentions
normalized to `<MENTION>`, consistency feature enabled, and stage-1 hybrid
weights fixed to `text 0.8 / stats 0.2`):

```bash
python training/train_stage1_hybrid.py ^
  --data-path "C:\Users\user\Downloads\dataset2_team_all_merged.csv" ^
  --output-dir saved_models/hybrid_current ^
  --model-name saved_models/stage1_kcbert_binary/bert_classifier ^
  --text-col transformed_text ^
  --raw-text-col reply_text ^
  --post-col post_text ^
  --label-col model_source ^
  --split-col split ^
  --force-text-weight 0.8
```

Notes:

- `saved_models/hybrid_current` is the directory the server prefers by default.
- `--force-text-weight 0.8` reuses the validation-selected threshold for the
  `0.8 / 0.2` hybrid setting and writes that weight into `artifacts.json`.
- The generated model directory is intentionally not committed here because one
  trained artifact directory is about `1.6 GB`.
- After training, verify `saved_models/hybrid_current/artifacts.json` contains:
  - `model_text_transform = body_mentions_special_tokens_v1`
  - `strip_leading_mentions = true`
  - `text_weight = 0.8`
  - `stats_weight = 0.2`
  - `post_reply_cosine_similarity` inside `stats_features`

## Stage-2 training

Default dataset columns:

- text: `reply_text`
- parent post text: `post_text` (optional, used for `post_reply_cosine_similarity`)
- label: `model_source`
- split: `split`

Command:

```bash
python training/train_hybrid.py ^
  --data-path "C:\Users\user\Downloads\dataset2_team_all_merged.csv" ^
  --output-dir saved_models/stage2_llm_hybrid ^
  --text-col reply_text ^
  --post-col post_text ^
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

To run the plugin-facing server with the newer stage-2 model
`stage2_llm_hybrid_consistency_20260504_text07`, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_plugin_server.ps1
```

This script prefers stage 1 on `saved_models/hybrid_current` when that folder
contains trained artifacts, otherwise it falls back to
`saved_models/stage1_kcbert_binary`. It switches stage 2 via `STAGE2_MODEL_DIR`
and starts the same FastAPI app on
`http://127.0.0.1:8000`.

For the main-server stage-1 setup described above, retrain into
`saved_models/hybrid_current` first and then start the server with the same
script. No config-file change is needed if the trained artifacts exist in that
folder.

After startup, verify the loaded model with:

```text
GET /health
```

You should see:

- `stage2_model_dir`: `.../saved_models/stage2_llm_hybrid_consistency_20260504_text07`
- `stage2_model_version`: `stage2_llm_hybrid_consistency_20260504_text07`

## Tokenizer Special Tokens / 토크나이저 특수 토큰

### English

This branch adds explicit tokenizer special-token plumbing for both training and inference.

Supported transform names:

- `transformed_text_or_raw_to_special_tokens`
- `raw_to_special_tokens`
- `x_text_special_tokens_v1`

Legacy special tokens:

- `<SPACE>`
- `<ENTER>`
- `<REP>`
- `</REP>`

X-style special tokens:

- `<URL>`
- `<MENTION>`
- `<HASHTAG>`
- `<ENTER>`
- `<REP>`
- `</REP>`
- `<EMOJI>`

X-style preprocessing (`x_text_special_tokens_v1`) rewrites raw text before tokenization:

- URL -> `<URL>`
- mention -> `<MENTION>`
- hashtag -> `<HASHTAG> tag_text`
- newline -> `<ENTER>`
- emoji -> `<EMOJI>`
- repeated character 3+ times -> `<REP> char count </REP>`

Code paths:

- `server/preprocess.py`
  - defines transform names
  - defines token lists
  - exposes `add_model_special_tokens(...)`
  - applies `prepare_model_text(...)`
- `training/train_kcbert_binary.py`
  - accepts `--model-text-transform`
  - adds special tokens to the tokenizer
  - resizes token embeddings
  - stores the transform name in `artifacts.json`
- `training/train_hybrid.py`
  - applies the same tokenizer flow for stage-2 training
- `server/inference.py`
  - reads `model_text_transform` from model metadata
  - applies the same preprocessing path at runtime

Notes:

- legacy models still default to the legacy transform when `artifacts.json` does not specify `model_text_transform`
- `<SPACE>` is legacy-only and is not used by `x_text_special_tokens_v1`

### 한국어

이 브랜치는 학습과 추론 양쪽에서 사용할 수 있도록 토크나이저 special token 처리 경로를 명시적으로 추가합니다.

지원하는 transform 이름:

- `transformed_text_or_raw_to_special_tokens`
- `raw_to_special_tokens`
- `x_text_special_tokens_v1`

기존(legacy) special token:

- `<SPACE>`
- `<ENTER>`
- `<REP>`
- `</REP>`

X 스타일 special token:

- `<URL>`
- `<MENTION>`
- `<HASHTAG>`
- `<ENTER>`
- `<REP>`
- `</REP>`
- `<EMOJI>`

X 스타일 전처리(`x_text_special_tokens_v1`)는 토크나이징 전에 raw text를 아래처럼 변환합니다.

- URL -> `<URL>`
- 멘션 -> `<MENTION>`
- 해시태그 -> `<HASHTAG> 태그본문`
- 줄바꿈 -> `<ENTER>`
- 이모지 -> `<EMOJI>`
- 같은 문자가 3번 이상 반복되면 -> `<REP> 문자 개수 </REP>`

관련 코드 위치:

- `server/preprocess.py`
  - transform 이름 정의
  - token 목록 정의
  - `add_model_special_tokens(...)` 제공
  - `prepare_model_text(...)`에서 전처리 적용
- `training/train_kcbert_binary.py`
  - `--model-text-transform` 인자 지원
  - tokenizer에 special token 추가
  - embedding resize 수행
  - 사용한 transform 이름을 `artifacts.json`에 저장
- `training/train_hybrid.py`
  - stage-2 학습에도 같은 tokenizer 처리 흐름 적용
- `server/inference.py`
  - 모델 metadata에서 `model_text_transform`을 읽고
  - 추론 시에도 같은 전처리 경로를 사용

메모:

- `artifacts.json`에 `model_text_transform`이 없으면 기존 모델은 legacy transform을 기본값으로 사용합니다
- `<SPACE>`는 legacy 전용이며 `x_text_special_tokens_v1`에서는 사용하지 않습니다

## Main files

- `training/train_kcbert_binary.py`: stage-1 human vs ai trainer
- `training/train_hybrid.py`: stage-2 AI-family hybrid trainer
- `server/preprocess.py`: transform rules and stats feature extraction
- `server/inference.py`: two-stage inference pipeline
- `server/app.py`: FastAPI routes
- `docs/api-spec-draft.md`: API draft
