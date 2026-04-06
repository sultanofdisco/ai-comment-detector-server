# Colab 학습 가이드

## 1. 목적

이 문서는 로컬 PC 대신 Google Colab에서 학습을 돌려서 모델 산출물을 확보하는 방법을 정리한 가이드입니다.

현재 프로젝트 기준 학습 구조는 다음과 같습니다.

1. 1차 모델
   - `KC-BERT binary`
   - 입력: `transformed_text`
   - 출력: `human` / `ai`
2. 2차 모델
   - `Hybrid LLM classifier`
   - 입력: `reply_text`
   - 대상: 1차에서 AI로 판정된 댓글
   - 출력: `gpt`, `claude`, `gemini`, `deepseek`

1주차 기준으로는 **1차 모델부터 먼저 학습해서 모델 폴더를 확보하는 것**을 권장합니다.

---

## 2. Colab에서 먼저 할 일

### 런타임 설정

- 메뉴에서 `런타임 > 런타임 유형 변경`
- `하드웨어 가속기: GPU` 선택

무료 Colab은 GPU 종류와 사용 가능 시간이 고정이 아니므로, 출력 폴더는 반드시 Google Drive에 저장하는 것이 안전합니다.

---

## 3. 셀 순서대로 실행하기

### 셀 1. Google Drive 마운트

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### 셀 2. GPU 확인

```bash
!nvidia-smi
```

```python
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("device_name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
```

정상이라면 `cuda_available: True`가 나와야 합니다.

---

### 셀 3. 작업 폴더 준비

```bash
%cd /content
!rm -rf ai-comment-detector-server
```

---

### 셀 4-A. GitHub에서 repo 가져오기

레포가 GitHub에 올라가 있다면 이 방법이 가장 편합니다.

```bash
!git clone <레포_주소> /content/ai-comment-detector-server
%cd /content/ai-comment-detector-server
```

예:

```bash
!git clone https://github.com/your-org/ai-comment-detector-server.git /content/ai-comment-detector-server
%cd /content/ai-comment-detector-server
```

---

### 셀 4-B. GitHub를 쓰지 않는 경우

현재 로컬 프로젝트 폴더를 zip으로 압축해서 Colab에 업로드한 뒤 푸는 방식도 가능합니다.

```python
from google.colab import files
uploaded = files.upload()
```

업로드한 zip 파일명이 `ai-comment-detector-server.zip`이라면:

```bash
!unzip -q ai-comment-detector-server.zip -d /content
%cd /content/ai-comment-detector-server
```

둘 중 하나만 선택해서 진행하면 됩니다.

---

### 셀 5. 패키지 설치

```bash
!python -m pip install --upgrade pip
!pip install -r requirements.txt
```

Colab 환경에서 `requirements.txt` 설치가 너무 오래 걸리면 아래처럼 최소 패키지만 설치해도 됩니다.

```bash
!pip install pandas numpy scikit-learn torch transformers accelerate xgboost joblib
```

---

### 셀 6. CSV 파일 확인

CSV를 Google Drive에 올려뒀다면 경로를 먼저 확인합니다.

```bash
!ls "/content/drive/MyDrive"
```

예시 경로:

```text
/content/drive/MyDrive/dataset2_team_all_merged.csv
```

---

## 4. 1차 모델 학습

### 셀 7. 1차 모델 빠른 검증용 실행

먼저 1 epoch로 돌아가는지 확인하는 것을 권장합니다.

```bash
!python training/train_kcbert_binary.py \
  --data-path "/content/drive/MyDrive/dataset2_team_all_merged.csv" \
  --output-dir "/content/drive/MyDrive/ai-comment-detector-models/stage1_kcbert_binary" \
  --text-col transformed_text \
  --label-col model_source \
  --split-col split \
  --train-batch-size 8 \
  --eval-batch-size 8 \
  --epochs 1
```

### 셀 8. 1차 모델 본학습

검증이 끝나면 epoch를 늘려 다시 실행합니다.

```bash
!python training/train_kcbert_binary.py \
  --data-path "/content/drive/MyDrive/dataset2_team_all_merged.csv" \
  --output-dir "/content/drive/MyDrive/ai-comment-detector-models/stage1_kcbert_binary" \
  --text-col transformed_text \
  --label-col model_source \
  --split-col split \
  --train-batch-size 8 \
  --eval-batch-size 8 \
  --epochs 3
```

### 1차 모델 산출물 위치

학습이 끝나면 아래 폴더에 결과가 저장됩니다.

```text
/content/drive/MyDrive/ai-comment-detector-models/stage1_kcbert_binary
```

이 안에는 보통 아래 파일들이 들어갑니다.

- `artifacts.json`
- `metrics.json`
- `label_map.json`
- `bert_classifier/`

---

## 5. 2차 모델 학습

2차 모델은 선택 사항입니다. 1주차 데모가 급하면 stage1만 먼저 확보해도 됩니다.

### 셀 9. 2차 모델 빠른 검증용 실행

```bash
!python training/train_hybrid.py \
  --data-path "/content/drive/MyDrive/dataset2_team_all_merged.csv" \
  --output-dir "/content/drive/MyDrive/ai-comment-detector-models/stage2_llm_hybrid" \
  --text-col reply_text \
  --label-col model_source \
  --split-col split \
  --train-batch-size 8 \
  --eval-batch-size 8 \
  --epochs 1
```

### 셀 10. 2차 모델 본학습

```bash
!python training/train_hybrid.py \
  --data-path "/content/drive/MyDrive/dataset2_team_all_merged.csv" \
  --output-dir "/content/drive/MyDrive/ai-comment-detector-models/stage2_llm_hybrid" \
  --text-col reply_text \
  --label-col model_source \
  --split-col split \
  --train-batch-size 8 \
  --eval-batch-size 8 \
  --epochs 3
```

### 2차 모델 산출물 위치

```text
/content/drive/MyDrive/ai-comment-detector-models/stage2_llm_hybrid
```

이 안에는 보통 아래 파일들이 들어갑니다.

- `artifacts.json`
- `metrics.json`
- `label_map.json`
- `xgb_model.joblib`
- `bert_classifier/`

---

## 6. 학습 결과 간단 확인

### 셀 11. 출력 폴더 확인

```bash
!find "/content/drive/MyDrive/ai-comment-detector-models" -maxdepth 3 -type f | sort
```

### 셀 12. metrics 확인

```python
import json
from pathlib import Path

stage1_metrics = Path("/content/drive/MyDrive/ai-comment-detector-models/stage1_kcbert_binary/metrics.json")
if stage1_metrics.exists():
    print("STAGE1")
    print(json.loads(stage1_metrics.read_text(encoding="utf-8")))

stage2_metrics = Path("/content/drive/MyDrive/ai-comment-detector-models/stage2_llm_hybrid/metrics.json")
if stage2_metrics.exists():
    print("STAGE2")
    print(json.loads(stage2_metrics.read_text(encoding="utf-8")))
```

---

## 7. 로컬 서버로 옮기기

학습이 끝났으면 Drive에 저장된 모델 폴더를 로컬 프로젝트의 `saved_models` 아래로 복사하면 됩니다.

로컬 기준 목표 구조:

```text
ai-comment-detector-server/
├─ saved_models/
│  ├─ stage1_kcbert_binary/
│  └─ stage2_llm_hybrid/
```

즉, Colab에서 만든 아래 두 폴더를 그대로 가져오면 됩니다.

- `/content/drive/MyDrive/ai-comment-detector-models/stage1_kcbert_binary`
- `/content/drive/MyDrive/ai-comment-detector-models/stage2_llm_hybrid`

---

## 8. 로컬 서버 실행

모델 폴더를 로컬 저장소에 옮긴 뒤 서버를 실행합니다.

```powershell
cd C:\Users\user\ai-comment-detector-server
.\.venv\Scripts\Activate.ps1
python -m uvicorn server.app:app --reload
```

### 상태 확인

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
```

### 예측 테스트

```powershell
$body = @{
  comment_id = "c1"
  author_id = "user123"
  text = "댓글 내용"
  url = "https://x.com/example/status/1"
  timestamp = "2026-03-30T12:00:00Z"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

---

## 9. 추천 순서

가장 안전한 순서는 아래와 같습니다.

1. Colab에서 stage1을 `epochs=1`로 먼저 실행
2. stage1 산출물 생성 확인
3. 필요하면 stage1을 `epochs=3`으로 재학습
4. 여유가 있으면 stage2 실행
5. 모델 폴더를 로컬 `saved_models`로 복사
6. 로컬 FastAPI 서버 실행
7. `/health`와 `/predict` 확인

---

## 10. 문제 생겼을 때 먼저 볼 것

- `cuda_available: False`
  - 런타임이 GPU로 안 바뀌었을 가능성
- `No module named server`
  - 현재 스크립트는 수정되어 있어야 정상 동작
  - 최신 repo를 다시 clone 했는지 확인
- Drive 저장 중 끊김
  - 출력 경로를 Drive로 지정했는지 확인
- 런타임 재시작
  - 다시 Drive mount 후 설치 셀부터 재실행
- 메모리 부족
  - `--train-batch-size 4`
  - `--eval-batch-size 4`
  - `--epochs 1`로 먼저 확인

