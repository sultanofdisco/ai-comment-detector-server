# Hard Negative 가이드

## 1. Hard Negative란?

Hard negative는 **사람이 쓴 댓글인데 현재 모델이 AI처럼 착각하기 쉬운 댓글**입니다.

지금 프로젝트에서는 아래 유형이 대표적인 hard negative입니다.

- 짧은 축하 댓글
- 공감/위로 댓글
- `ㅋㅋ`, `ㅠㅠ`, 이모지 중심 댓글
- 공손한 존댓말 답글
- 한두 문장짜리 정돈된 반응
- 질문형 마무리 댓글

즉, 그냥 인간 댓글을 아무거나 많이 넣는 것보다 **오탐이 실제로 많이 나는 인간 댓글**을 넣는 것이 훨씬 중요합니다.

---

## 2. 왜 필요한가?

현재 stage1은 `human` vs `ai`를 나누는데, 실제 X 댓글에서 사람이 쓰는 짧은 반응형 문장이 AI 댓글과 비슷하게 보이는 경우가 많습니다.

예를 들면 이런 댓글들입니다.

- `축하드립니다!!`
- `얼마나 노력하셨을까요... 축하드립니다`
- `수고 많으셨어요 ㅠㅠ`
- `그럴 수도 있겠네요`
- `ㅋㅋㅋ 진짜 웃기네`

이런 댓글을 `human`으로 충분히 보여줘야, 모델이 “공손함 / 짧음 / 감탄 / 이모지”만으로 AI라고 단정하지 않게 됩니다.

---

## 3. 몇 개 정도 필요할까?

권장 기준은 아래와 같습니다.

- 최소 체감 개선: `300~500개`
- 꽤 의미 있는 개선: `1,000~2,000개`
- 현재 편향이 큰 상황에서 추천: `2,000개+`

중요한 건 단순 총개수보다 **패턴 다양성**입니다.

추천 수집 비율:

- 축하/응원: `200+`
- 공감/위로: `200+`
- 짧은 감탄/리액션: `200+`
- `ㅋㅋ / ㅠㅠ / 이모지` 중심: `200+`
- 공손한 존댓말 답글: `200+`
- 욕설/반말/커뮤 톤: `200+`

---

## 4. 어떻게 수집하나?

가장 좋은 방법은 **현재 모델이 AI로 잘못 의심한 댓글만 따로 모으는 것**입니다.

실전 추천 흐름:

1. 익스텐션으로 실제 X 댓글 페이지를 본다.
2. `AI 의심` 또는 `판별 보류`로 뜬 댓글 중, 사람이 쓴 것이 분명한 댓글을 수집한다.
3. 수집한 댓글을 CSV로 정리한다.
4. `review_label=human`으로 검수한다.
5. 학습용 형식으로 변환해서 기존 dataset에 합친다.

---

## 5. 수집 템플릿

템플릿 파일:

- [hard_negative_template.csv](C:/Users/user/ai-comment-detector-server/samples/hard_negative_template.csv)

컬럼 예시:

- `reply_text`: 댓글 본문
- `post_text`: 원문 게시글 텍스트, 없으면 빈칸 가능
- `source_url`: 출처 URL
- `review_label`: 검수 결과, `human`
- `bucket`: 댓글 유형
- `notes`: 메모

---

## 6. 학습용 CSV로 변환하는 방법

수집이 끝나면 아래 스크립트로 현재 dataset 형식으로 바꿀 수 있습니다.

- [prepare_hard_negatives.py](C:/Users/user/ai-comment-detector-server/training/prepare_hard_negatives.py)

예시:

```powershell
cd C:\Users\user\ai-comment-detector-server
.\.venv\Scripts\Activate.ps1
python training\prepare_hard_negatives.py `
  --input-path samples\hard_negative_template.csv `
  --output-path samples\hard_negatives_prepared.csv `
  --base-dataset-path "C:\Users\user\Downloads\dataset2_team_all_merged.csv" `
  --merged-output-path samples\dataset2_team_all_merged_with_hard_negatives.csv
```

이 스크립트는 아래를 자동으로 처리합니다.

- `transformed_text` 생성
- stats feature 생성
- `model_source=human`
- `label=0`
- `split=train`
- 기존 데이터셋과 merge 시 `reply_text` 기준 dedup

---

## 7. 재학습 추천 순서

1. hard negative `300~500개`로 1차 실험
2. stage1 재학습
3. 실제 X 댓글 페이지에서 오탐 감소 확인
4. 부족하면 `1,000개+`까지 확장
5. 최종적으로 threshold 재선택

---

## 8. 수집할 때 주의할 점

- “사람이 쓴 게 확실한 댓글”만 넣기
- 같은 댓글 반복 수집하지 않기
- 너무 긴 설명형 댓글만 모으지 말기
- 짧은 댓글과 공손한 댓글을 적극적으로 모으기
- 실제 오탐 패턴을 중심으로 모으기

좋은 hard negative는 “모델이 틀리기 쉬운 인간 댓글”입니다.
즉, 쉬운 인간 댓글보다 **현재 모델이 가장 헷갈리는 인간 댓글**을 우선적으로 추가하는 게 맞습니다.

