# AI 댓글 탐지 서버 API 명세서 초안

## 1. 문서 목적

이 문서는 크롬 확장 프로그램과 서버 간의 입출력 형식을 통일하기 위한 API 명세서이다.

플러그인 팀은 이 문서를 기준으로

- 댓글 데이터를 어떤 형식으로 서버에 보낼지
- 서버 응답을 어떤 방식으로 화면에 표시할지
- 관리자 페이지에서 어떤 집계 값을 사용할지

를 정리할 수 있다.

---

## 2. 시스템 개요

본 서버는 댓글 1건에 대해 2단계 분류를 수행한다.

### 2.1 1차 분류

- 모델: KC-BERT binary classifier
- 목적: 댓글이 `human`인지 `ai`인지 판별
- 입력: transformed된 텍스트
- 출력: `human` 또는 `ai`

### 2.2 2차 분류

- 모델: AI-only hybrid classifier
- 목적: AI 댓글로 판단된 경우 어떤 LLM 계열인지 분류
- 입력: 댓글 텍스트
- 출력: `gpt`, `claude`, `gemini`, `deepseek`

### 2.3 분기 조건

- 1차 분류 결과의 `ai_score < 0.85`
  - 2차 분류를 수행하지 않음
  - 최종 라벨은 `human` 또는 `ai`
- 1차 분류 결과의 `ai_score >= 0.85`
  - 2차 분류 수행
  - 최종 라벨은 LLM 계열 라벨

즉, 모든 댓글이 LLM 분류로 넘어가는 것이 아니라,
먼저 인간/AI 여부를 판별한 뒤 일정 기준을 넘는 경우에만 2차 분류를 수행한다.

---

## 3. 기본 정보

### 3.1 Base URL

로컬 개발 기준:

```text
http://127.0.0.1:8000
```

### 3.2 Content-Type

```text
Content-Type: application/json
```

### 3.3 시간 형식

`timestamp`는 ISO 8601 UTC 문자열을 권장한다.

예시:

```text
2026-03-30T12:00:00Z
```

---

## 4. 응답 해석 규칙

플러그인 팀이 가장 헷갈리기 쉬운 부분이라 별도로 정리한다.

### 4.1 `pred_label`

최종 결과 라벨이다.

가능한 값은 다음과 같다.

| 값 | 의미 |
| --- | --- |
| `human` | 1차 모델이 인간 댓글로 판단 |
| `ai` | 1차 모델이 AI로 판단했지만 2차 분류 기준 미만 |
| `gpt` | 2차 모델이 GPT 계열로 분류 |
| `claude` | 2차 모델이 Claude 계열로 분류 |
| `gemini` | 2차 모델이 Gemini 계열로 분류 |
| `deepseek` | 2차 모델이 DeepSeek 계열로 분류 |

### 4.2 `confidence`

최종 `pred_label`에 대한 confidence이다.

- `pred_label = human` 또는 `ai`
  - 1차 모델 confidence
- `pred_label = gpt`, `claude`, `gemini`, `deepseek`
  - 2차 모델 confidence

### 4.3 `ai_score`

항상 1차 모델 기준 AI 확률이다.

즉, `pred_label`이 `gpt`여도 `ai_score`는 “GPT 확률”이 아니라
“1차 모델이 AI라고 본 확률”이다.

### 4.4 `top2`

top-2 예측 결과이다.

- 2차 분류가 실행되지 않은 경우
  - 1차 모델 top-2
  - 예: `["ai", 0.73]`, `["human", 0.27]`
- 2차 분류가 실행된 경우
  - 2차 모델 top-2
  - 예: `["gpt", 0.91]`, `["gemini", 0.06]`

### 4.5 `risk_level`

`ai_score`를 기준으로 정한 위험도이다.

| 조건 | 값 |
| --- | --- |
| `ai_score < 0.5` | `low` |
| `0.5 <= ai_score < 0.85` | `medium` |
| `ai_score >= 0.85` | `high` |

---

## 5. API 목록

| Method | Path | 설명 |
| --- | --- | --- |
| `GET` | `/health` | 서버 및 모델 상태 확인 |
| `POST` | `/predict` | 댓글 1건 예측 |
| `GET` | `/admin/model-usage` | 관리자용 LLM 사용 빈도 집계 |

---

## 6. GET /health

### 6.1 설명

서버가 정상 동작 중인지,
1차/2차 모델이 어느 경로에 로드되는지,
현재 threshold가 얼마인지 확인하는 API이다.

### 6.2 Response `200 OK`

```json
{
  "status": "ok",
  "stage1_model_dir": "saved_models/stage1_kcbert_binary",
  "stage2_model_dir": "saved_models/stage2_llm_hybrid",
  "device": "cpu",
  "ai_threshold": 0.85,
  "stage1_model_version": "stage1_kcbert_binary",
  "stage2_model_version": "stage2_llm_hybrid"
}
```

### 6.3 필드 정의

| 필드명 | 타입 | 설명 |
| --- | --- | --- |
| `status` | string | 서버 상태 |
| `stage1_model_dir` | string | 1차 KC-BERT 모델 경로 |
| `stage2_model_dir` | string | 2차 hybrid 모델 경로 |
| `device` | string | 추론 장치, 예: `cpu`, `cuda` |
| `ai_threshold` | number | 2차 분류 진입 기준값 |
| `stage1_model_version` | string \| null | 1차 모델 버전 |
| `stage2_model_version` | string \| null | 2차 모델 버전 |

---

## 7. POST /predict

### 7.1 설명

플러그인에서 댓글 1건을 서버로 보내면,
서버는 1차 인간/AI 판별을 수행하고 필요하면 2차 LLM 분류까지 수행한 뒤 결과를 반환한다.

### 7.2 Request Body

```json
{
  "comment_id": "c1",
  "author_id": "user123",
  "text": "댓글 내용",
  "url": "https://x.com/example/status/1",
  "timestamp": "2026-03-30T12:00:00Z"
}
```

### 7.3 Request 필드 정의

| 필드명 | 타입 | 필수 여부 | 설명 |
| --- | --- | --- | --- |
| `comment_id` | string \| null | 선택 | 댓글 식별자 |
| `author_id` | string \| null | 선택 | 작성자 식별자 |
| `text` | string | 필수 | 댓글 원문 |
| `url` | string \| null | 선택 | 댓글이 포함된 페이지 URL |
| `timestamp` | string \| null | 선택 | 프론트에서 전달하는 댓글 시각 |

### 7.4 Response `200 OK`

#### Case A. 인간 댓글

```json
{
  "comment_id": "c1",
  "pred_label": "human",
  "confidence": 0.89,
  "ai_score": 0.11,
  "top2": [
    ["human", 0.89],
    ["ai", 0.11]
  ],
  "risk_level": "low",
  "reason": "stage1 kcbert predicted human with ai probability 0.11"
}
```

#### Case B. AI 의심이지만 2차 분류 기준 미만

```json
{
  "comment_id": "c2",
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

#### Case C. 2차 LLM 분류 수행

```json
{
  "comment_id": "c3",
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

### 7.5 Response 필드 정의

| 필드명 | 타입 | 설명 |
| --- | --- | --- |
| `comment_id` | string \| null | 요청에서 들어온 댓글 ID |
| `pred_label` | string | 최종 예측 라벨 |
| `confidence` | number | 최종 라벨 confidence |
| `ai_score` | number | 1차 KC-BERT 기준 AI 확률 |
| `top2` | array | top-2 결과 배열 |
| `risk_level` | string | `low`, `medium`, `high` |
| `reason` | string | 결과 설명 문장 |

### 7.6 에러 응답

#### Response `400 Bad Request`

```json
{
  "detail": "Text must not be empty."
}
```

#### Response `503 Service Unavailable`

```json
{
  "detail": "Model metadata not found at ... Run training first."
}
```

#### Response `500 Internal Server Error`

```json
{
  "detail": "Inference failed: ..."
}
```

---

## 8. GET /admin/model-usage

### 8.1 설명

관리자 화면에서
“현재 AI 댓글 중 어떤 LLM 계열이 얼마나 검출되었는지”
표로 보여주기 위한 집계 API이다.

이 집계는 2차 분류가 실제로 수행된 경우만 포함한다.

즉, 아래 두 경우는 `by_label` 집계에 포함되지 않는다.

- `pred_label = human`
- `pred_label = ai` 이지만 2차 분류 threshold 미만

### 8.2 Response `200 OK`

```json
{
  "total_predictions": 128,
  "llm_classified_predictions": 74,
  "by_label": [
    { "label": "gpt", "count": 29, "ratio": 0.3919 },
    { "label": "claude", "count": 18, "ratio": 0.2432 },
    { "label": "gemini", "count": 15, "ratio": 0.2027 },
    { "label": "deepseek", "count": 12, "ratio": 0.1622 }
  ],
  "stage2_model_version": "stage2_llm_hybrid"
}
```

### 8.3 Response 필드 정의

| 필드명 | 타입 | 설명 |
| --- | --- | --- |
| `total_predictions` | integer | 전체 예측 요청 수 |
| `llm_classified_predictions` | integer | 실제 2차 LLM 분류가 수행된 요청 수 |
| `by_label` | array | LLM 계열별 빈도 목록 |
| `stage2_model_version` | string \| null | 2차 모델 버전 |

### 8.4 `by_label` 내부 객체 정의

| 필드명 | 타입 | 설명 |
| --- | --- | --- |
| `label` | string | LLM 계열 이름 |
| `count` | integer | 해당 라벨로 분류된 건수 |
| `ratio` | number | `count / llm_classified_predictions` |

---

## 9. 플러그인 팀 참고 사항

### 9.1 댓글 수집 시 최소 필요값

플러그인에서는 아래 값만 보내도 된다.

```json
{
  "text": "댓글 내용"
}
```

다만 추후 로그 분석과 계정 분석을 고려하면
가능하면 `comment_id`, `author_id`, `url`, `timestamp`도 함께 보내는 것이 좋다.

### 9.2 화면 표시 권장 방식

- `pred_label = human`
  - 별도 표시 없음 또는 저위험 표시
- `pred_label = ai`
  - “AI 의심” 배지 표시
- `pred_label = gpt`, `claude`, `gemini`, `deepseek`
  - “AI 의심 / GPT 추정”처럼 표시 가능

### 9.3 해석 시 주의점

- `ai_score`는 “AI 확률”이지 “특정 LLM 확률”이 아니다.
- `confidence`는 최종 라벨 confidence이다.
- `top2`는 경우에 따라 1차 top-2일 수도 있고 2차 top-2일 수도 있다.
- 관리자 표는 2차 분류 결과만 집계한다.

---

## 10. 비고

- 1주차 기준 필수 API는 `/predict`, `/health`, `/admin/model-usage` 이다.
- 계정 단위 분석 API는 아직 포함하지 않았다.
- 현재 threshold는 `0.85`이며 서버 설정으로 변경 가능하다.
- 1차 모델은 transformed text 기반으로 학습되며,
  서버는 raw text를 내부 전처리 규칙으로 변환한 뒤 추론한다.
