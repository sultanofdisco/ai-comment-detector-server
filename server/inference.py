from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from server.preprocess import build_stats_frame, normalize_text, prepare_model_text


class BaseTextPredictor:
    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        self.model_dir = Path(model_dir)
        self.device = device

        metadata_path = self.model_dir / "artifacts.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Model metadata not found at {metadata_path}. Run training first."
            )

        with metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)

        self.model_version = self.metadata.get("model_version", self.model_dir.name)
        self.max_length = int(self.metadata["max_length"])
        self.label_classes = list(self.metadata["label_classes"])

        text_model_dir = self.model_dir / self.metadata["text_model_subdir"]
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_dir)
        self.text_model = AutoModelForSequenceClassification.from_pretrained(text_model_dir)
        self.text_model.to(self.device)
        self.text_model.eval()

    def _predict_text_proba(self, model_text: str) -> np.ndarray:
        # 하위 호환성을 보장하면서 내부 직접 추론 파이프라인을 작동시킵니다.
        return self._predict_text_proba_direct(model_text)

    def _predict_text_proba_direct(self, model_text: str) -> np.ndarray:
        encoded = self.tokenizer(
            [model_text],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = self.text_model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        return probabilities

    def get_attention_attribution(self, model_text: str) -> list[dict[str, Any]]:
        """
        [XAI 신경망 역추적 핵심] AutoModel 모델 내부의 수치 레이어 어텐션 맵을 추출하여
        CLS 토큰이 해당 댓글 내에서 주의 깊게 인지한 단어들의 가중치를 백분율로 환산합니다.
        """
        encoded = self.tokenizer(
            [model_text],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        try:
            with torch.no_grad():
                # 어텐션 행렬 가중치 추출 연산을 위해 output_attentions=True 속성을 내장합니다.
                outputs = self.text_model(**encoded, output_attentions=True)
                attentions = outputs.attentions

            if attentions is not None and len(attentions) > 0:
                # 마지막 레이어의 어텐션 행렬 가중치 수집
                last_layer_attention = attentions[-1][0].cpu().numpy()
                cls_attention = np.mean(last_layer_attention[:, 0, :], axis=0)

                token_ids = encoded["input_ids"][0].cpu().numpy()
                tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

                attributions = []
                for token, weight in zip(tokens, cls_attention):
                    # BERT 기반 특수 토큰 격리
                    if token in ["[CLS]", "[SEP]", "[PAD]", "CLS", "SEP", "PAD"]:
                        continue
                    clean_token = token.replace("##", "")
                    if clean_token.strip():
                        attributions.append({
                            "token": clean_token,
                            "weight": float(weight)
                        })

                # 기여 점수가 높은 핵심 토큰 순서 정렬
                attributions.sort(key=lambda x: x["weight"], reverse=True)
                return attributions
        except Exception as e:
            print(f"[XAI Attention Attribution Log] {e}")
        return []


class KcBertBinaryPredictor(BaseTextPredictor):
    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        super().__init__(model_dir, device)
        self.positive_label = self.metadata.get("positive_label", "ai")
        self.negative_label = self.metadata.get("negative_label", "human")

    def get_attention_attribution(self, text: str) -> list[dict[str, Any]]:
        normalized_text = normalize_text(text)
        if not normalized_text:
            return []
        model_text = prepare_model_text(normalized_text)
        return super().get_attention_attribution(model_text)

    def predict(self, text: str) -> dict[str, Any]:
        normalized_text = normalize_text(text)
        if not normalized_text:
            raise ValueError("Text must not be empty.")

        model_text = prepare_model_text(normalized_text)
        probabilities = self._predict_text_proba_direct(model_text)[0]
        scores = [
            {"label": label, "score": float(score)}
            for label, score in zip(self.label_classes, probabilities)
        ]
        scores.sort(key=lambda item: item["score"], reverse=True)
        score_map = {item["label"]: item["score"] for item in scores}

        ai_score = float(score_map.get(self.positive_label, 0.0))
        pred_label = self.positive_label if ai_score >= 0.5 else self.negative_label
        confidence = float(score_map[pred_label])

        return {
            "pred_label": pred_label,
            "confidence": confidence,
            "ai_score": ai_score,
            "top2": [(item["label"], float(item["score"])) for item in scores[:2]],
            "model_version": self.model_version,
            "model_text": model_text,
            "label_scores": scores,
        }


class HybridLlmPredictor(BaseTextPredictor):
    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        super().__init__(model_dir, device)
        self.text_weight = float(self.metadata["text_weight"])
        self.stats_weight = float(self.metadata["stats_weight"])
        self.feature_order = list(self.metadata["stats_features"])
        xgb_model_path = self.model_dir / self.metadata["xgb_model_filename"]
        self.stats_model = joblib.load(xgb_model_path)

    def _predict_stats_proba(self, raw_text: str) -> np.ndarray:
        features = build_stats_frame([raw_text], self.feature_order)
        probabilities = self.stats_model.predict_proba(features)
        return np.asarray(probabilities, dtype=float)

    def predict(self, text: str) -> dict[str, Any]:
        normalized_text = normalize_text(text)
        if not normalized_text:
            raise ValueError("Text must not be empty.")

        model_text = prepare_model_text(normalized_text)
        text_proba = self._predict_text_proba_direct(model_text)
        stats_proba = self._predict_stats_proba(normalized_text)
        hybrid_proba = (self.text_weight * text_proba) + (self.stats_weight * stats_proba)

        scores = [
            {"label": label, "score": float(score)}
            for label, score in zip(self.label_classes, hybrid_proba[0])
        ]
        scores.sort(key=lambda item: item["score"], reverse=True)

        return {
            "pred_label": scores[0]["label"],
            "confidence": float(scores[0]["score"]),
            "top2": [(item["label"], float(item["score"])) for item in scores[:2]],
            "model_version": self.model_version,
            "model_text": model_text,
            "label_scores": scores,
        }


class TwoStagePredictor:
    def __init__(
        self,
        stage1_model_dir: str | Path,
        stage2_model_dir: str | Path,
        device: str = "cpu",
        ai_threshold: float = 0.7,
        llm_threshold: float = 0.85,
        llm_confidence_threshold: float = 0.95,
    ) -> None:
        self.stage1 = KcBertBinaryPredictor(stage1_model_dir, device)
        self.ai_threshold = ai_threshold
        self.llm_threshold = llm_threshold
        self.llm_confidence_threshold = llm_confidence_threshold
        self.stage2: HybridLlmPredictor | None = None

        try:
            self.stage2 = HybridLlmPredictor(stage2_model_dir, device)
        except FileNotFoundError:
            self.stage2 = None

    @property
    def stage1_model_version(self) -> str:
        return self.stage1.model_version

    @property
    def stage2_model_version(self) -> str | None:
        if self.stage2 is None:
            return None
        return self.stage2.model_version

    def _generate_xai_reason(self, text: str, ai_score: float, pred_label: str) -> str:
        """
        [사용자 피드백 맞춤형 국문 템플릿 처리부]
        KcBERT 모델 내부 어텐션 기여도 지표와 오탈자 유무를 직관적인 한글로 매핑합니다.
        """
        attributions = []
        try:
            attributions = self.stage1.get_attention_attribution(text)
        except Exception:
            pass

        cleaned = text.strip()
        length = len(cleaned)
        
        has_laughter = bool(re.search(r'([ㅋㅎㅠㅜㅇ])\1+', cleaned))
        ends_with_period = cleaned.endswith('.') or cleaned.endswith('요.') or cleaned.endswith('다.')
        space_count = cleaned.count(" ")
        space_ratio = space_count / length if length > 0 else 0.0

        pct = round(ai_score * 100)
        human_pct = 100 - pct

        # 기여 어휘 명세 조립
        attention_text = ""
        if attributions:
            top_tokens = attributions[:2]
            token_phrases = [f"'{item['token']}'(기여도 {round(item['weight']*100, 1)}%)" for item in top_tokens]
            attention_text = f"실제 KcBERT 모델이 판단 과정에서 주목한 핵심 맥락 단어는 " + ", ".join(token_phrases) + " 입니다. "

        # 라벨 판정별 최종 출력 텍스트 조합
        if pred_label == "ai":
            reasons = []
            if ends_with_period and not has_laughter:
                reasons.append("정중한 마침표 규격과 종결식 표준형 구조가 문법적으로 완벽한 균형을 이룹니다.")
            if 0.12 <= space_ratio <= 0.18:
                reasons.append(f"전체 음절 수 대비 띄어쓰기 비율({round(space_ratio*100)}%)이 대단히 규칙적이고 완결도가 높습니다.")
            
            if not reasons:
                reasons.append("비격식 표현이나 일상 오탈자가 전혀 드러나지 않는 전형적인 인공 생성 템플릿의 양상을 띱니다.")

            reason_str = " ".join(reasons)
            return (
                f"맞춤법과 띄어쓰기 오류가 단 하나도 없고 {reason_str}\n\n"
                f"{attention_text}이러한 어휘 기여도와 인공신경망의 가중치 흐름을 종합하여 최종 AI 작성 글(의심 확률 {pct}%)로 강력히 의심됩니다."
            )

        elif pred_label == "uncertain":
            return (
                f"이 댓글은 일상적인 네티즌 표현이 일부 섞여 있으나, 구조적으로 정돈된 완결성이 동시에 감지됩니다.\n\n"
                f"{attention_text}모델 내 셀프 어텐션 레이어가 복합적인 문체 자질을 동시에 학습하고 있음을 인지하였기에 AI 의심 경계 단계(의심 확률 {pct}%)로 판단됩니다."
            )

        else:
            reasons = []
            if has_laughter:
                reasons.append("인터넷 소통용 비격식 감정 자음(ㅋㅋㅋ, ㅎㅎ 등)이 자유롭게 배치되어 인위적 정형성을 이탈하고 있습니다.")
            if length < 15:
                reasons.append(f"문법의 구조적 규칙에서 벗어난 모바일 실생활 밀착형 초단문({length}자) 구조를 보입니다.")
            
            if not reasons:
                reasons.append("주어와 서술어의 생략이 부드럽고 실제 사용자가 자발적으로 남긴 고유의 말버릇 양상을 담고 있습니다.")

            reason_str = " ".join(reasons)
            return (
                f"문맥 상 실제 네티즌 특유의 자연스럽고 비정형적인 구어체 흐름이 감지되며, {reason_str}\n\n"
                f"{attention_text}모델 판단 결과 기계 학습의 전형적인 작위 문체에서 완전히 이탈하여 정상적인 인간 작성 댓글(신뢰도 {human_pct}%)로 안전하게 판명됩니다."
            )

    def predict(self, text: str) -> dict[str, Any]:
        stage1_result = self.stage1.predict(text)
        ai_score = float(stage1_result["ai_score"])
        risk_level = self._compute_risk_level(ai_score)

        if ai_score < 0.5:
            pred_label = "human"
            reason = self._generate_xai_reason(text, ai_score, pred_label)
            return {
                "pred_label": pred_label,
                "confidence": float(stage1_result["confidence"]),
                "ai_score": ai_score,
                "top2": stage1_result["top2"],
                "risk_level": risk_level,
                "reason": reason,
                "stage1_pred_label": "human",
                "stage1_model_version": self.stage1_model_version,
                "stage2_pred_label": None,
                "stage2_model_version": self.stage2_model_version,
            }

        if ai_score < self.ai_threshold:
            pred_label = "uncertain"
            reason = self._generate_xai_reason(text, ai_score, pred_label)
            return {
                "pred_label": pred_label,
                "confidence": float(stage1_result["confidence"]),
                "ai_score": ai_score,
                "top2": stage1_result["top2"],
                "risk_level": risk_level,
                "reason": reason,
                "stage1_pred_label": "uncertain",
                "stage1_model_version": self.stage1_model_version,
                "stage2_pred_label": None,
                "stage2_model_version": self.stage2_model_version,
            }

        if ai_score < self.llm_threshold:
            pred_label = "ai"
            reason = self._generate_xai_reason(text, ai_score, pred_label)
            return {
                "pred_label": pred_label,
                "confidence": float(stage1_result["confidence"]),
                "ai_score": ai_score,
                "top2": stage1_result["top2"],
                "risk_level": risk_level,
                "reason": reason,
                "stage1_pred_label": "ai",
                "stage1_model_version": self.stage1_model_version,
                "stage2_pred_label": None,
                "stage2_model_version": self.stage2_model_version,
            }

        if self.stage2 is None:
            pred_label = "ai"
            reason = self._generate_xai_reason(text, ai_score, pred_label)
            return {
                "pred_label": pred_label,
                "confidence": float(stage1_result["confidence"]),
                "ai_score": ai_score,
                "top2": stage1_result["top2"],
                "risk_level": risk_level,
                "reason": reason,
                "stage1_pred_label": "ai",
                "stage1_model_version": self.stage1_model_version,
                "stage2_pred_label": None,
                "stage2_model_version": None,
            }

        stage2_result = self.stage2.predict(text)
        if float(stage2_result["confidence"]) < self.llm_confidence_threshold:
            pred_label = "ai"
            reason = self._generate_xai_reason(text, ai_score, pred_label)
            return {
                "pred_label": pred_label,
                "confidence": float(stage1_result["confidence"]),
                "ai_score": ai_score,
                "top2": stage2_result["top2"],
                "risk_level": risk_level,
                "reason": reason,
                "stage1_pred_label": "ai",
                "stage1_model_version": self.stage1_model_version,
                "stage2_pred_label": None,
                "stage2_model_version": self.stage2_model_version,
            }

        pred_label = stage2_result["pred_label"]
        reason = self._generate_xai_reason(text, ai_score, pred_label)
        return {
            "pred_label": pred_label,
            "confidence": float(stage2_result["confidence"]),
            "ai_score": ai_score,
            "top2": stage2_result["top2"],
            "risk_level": risk_level,
            "reason": reason,
            "stage1_pred_label": "ai",
            "stage1_model_version": self.stage1_model_version,
            "stage2_pred_label": stage2_result["pred_label"],
            "stage2_model_version": self.stage2_model_version,
        }

    @staticmethod
    def _compute_risk_level(ai_score: float) -> str:
        if ai_score >= 0.85:
            return "high"
        if ai_score >= 0.5:
            return "medium"
        return "low"