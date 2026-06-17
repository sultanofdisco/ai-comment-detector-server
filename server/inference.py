from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from server.preprocess import (
    LEGACY_MODEL_TEXT_TRANSFORM,
    build_stats_frame,
    normalize_text,
    prepare_model_text,
    strip_leading_reply_mentions,
)


def _read_model_metadata(model_dir: str | Path) -> dict[str, Any]:
    metadata_path = Path(model_dir) / "artifacts.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found at {metadata_path}. Run training first."
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_sorted_scores(
    label_classes: list[str],
    probabilities: np.ndarray,
) -> list[dict[str, float]]:
    scores = [
        {"label": label, "score": float(score)}
        for label, score in zip(label_classes, probabilities)
    ]
    scores.sort(key=lambda item: item["score"], reverse=True)
    return scores


def _build_binary_prediction_result(
    label_classes: list[str],
    probabilities: np.ndarray,
    positive_label: str,
    negative_label: str,
    decision_threshold: float,
    model_version: str,
    model_text: str,
) -> dict[str, Any]:
    scores = _build_sorted_scores(label_classes, probabilities)
    score_map = {item["label"]: item["score"] for item in scores}

    ai_score = float(score_map.get(positive_label, 0.0))
    pred_label = positive_label if ai_score >= decision_threshold else negative_label
    confidence = float(score_map[pred_label])

    return {
        "pred_label": pred_label,
        "confidence": confidence,
        "ai_score": ai_score,
        "top2": [(item["label"], float(item["score"])) for item in scores[:2]],
        "model_version": model_version,
        "model_text": model_text,
        "label_scores": scores,
    }


class BaseTextPredictor:
    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        self.model_dir = Path(model_dir)
        self.device = device

        self.metadata = _read_model_metadata(self.model_dir)
        self.model_version = self.metadata.get("model_version", self.model_dir.name)
        self.max_length = int(self.metadata["max_length"])
        self.label_classes = list(self.metadata["label_classes"])
        self.model_text_transform = str(
            self.metadata.get("model_text_transform", LEGACY_MODEL_TEXT_TRANSFORM)
        )
        self.uses_stats_model = False

        text_model_dir = self.model_dir / self.metadata["text_model_subdir"]
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_dir)
        self.text_model = AutoModelForSequenceClassification.from_pretrained(text_model_dir)
        self.text_model.to(self.device)
        self.text_model.eval()

    def _predict_text_proba(self, model_text: str) -> np.ndarray:
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
                outputs = self.text_model(**encoded, output_attentions=True)
                attentions = outputs.attentions

            if attentions is not None and len(attentions) > 0:
                last_layer_attention = attentions[-1][0].cpu().numpy()
                cls_attention = np.mean(last_layer_attention[:, 0, :], axis=0)

                token_ids = encoded["input_ids"][0].cpu().numpy()
                tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

                attributions = []
                for token, weight in zip(tokens, cls_attention):
                    if token in ["[CLS]", "[SEP]", "[PAD]", "CLS", "SEP", "PAD"]:
                        continue
                    clean_token = token.replace("##", "")
                    if clean_token.strip():
                        attributions.append(
                            {
                                "token": clean_token,
                                "weight": float(weight),
                            }
                        )

                attributions.sort(key=lambda item: item["weight"], reverse=True)
                return attributions
        except Exception as exc:
            print(f"[XAI Attention Attribution Log] {exc}")
        return []

    def get_attention_attribution_for_text(self, text: str) -> list[dict[str, Any]]:
        normalized_text = normalize_text(text)
        if not normalized_text:
            return []
        model_text = prepare_model_text(normalized_text, self.model_text_transform)
        return self.get_attention_attribution(model_text)


class KcBertBinaryPredictor(BaseTextPredictor):
    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        super().__init__(model_dir, device)
        self.positive_label = self.metadata.get("positive_label", "ai")
        self.negative_label = self.metadata.get("negative_label", "human")
        self.decision_threshold = float(self.metadata.get("recommended_ai_threshold", 0.5))
        if len(self.label_classes) != 2:
            raise ValueError(
                f"Stage-1 binary model must have exactly 2 labels, got {self.label_classes}."
            )
        if self.positive_label not in self.label_classes or self.negative_label not in self.label_classes:
            raise ValueError(
                "Stage-1 binary model metadata must include positive/negative labels present in label_classes."
            )

    def predict(self, text: str, post_text: str | None = None) -> dict[str, Any]:
        normalized_text = normalize_text(text)
        if not normalized_text:
            raise ValueError("Text must not be empty.")

        model_text = prepare_model_text(normalized_text, self.model_text_transform)
        probabilities = self._predict_text_proba(model_text)[0]
        return _build_binary_prediction_result(
            self.label_classes,
            probabilities,
            positive_label=self.positive_label,
            negative_label=self.negative_label,
            decision_threshold=self.decision_threshold,
            model_version=self.model_version,
            model_text=model_text,
        )


class HybridTextStatsPredictor(BaseTextPredictor):
    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        super().__init__(model_dir, device)
        self.text_weight = float(self.metadata["text_weight"])
        self.stats_weight = float(self.metadata["stats_weight"])
        self.feature_order = list(self.metadata["stats_features"])
        xgb_model_path = self.model_dir / self.metadata["xgb_model_filename"]
        self.stats_model = joblib.load(xgb_model_path)
        self.uses_stats_model = True

    def _predict_stats_proba(self, raw_text: str, post_text: str | None = None) -> np.ndarray:
        features = build_stats_frame([raw_text], self.feature_order, post_texts=[post_text])
        probabilities = self.stats_model.predict_proba(features)
        return np.asarray(probabilities, dtype=float)

    def _predict_hybrid_proba(self, raw_text: str, model_text: str, post_text: str | None = None) -> np.ndarray:
        text_proba = self._predict_text_proba(model_text)
        stats_proba = self._predict_stats_proba(raw_text, post_text=post_text)
        return (self.text_weight * text_proba) + (self.stats_weight * stats_proba)


class HybridBinaryPredictor(HybridTextStatsPredictor):
    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        super().__init__(model_dir, device)
        self.positive_label = self.metadata.get("positive_label", "ai")
        self.negative_label = self.metadata.get("negative_label", "human")
        self.decision_threshold = float(self.metadata.get("recommended_ai_threshold", 0.5))
        if len(self.label_classes) != 2:
            raise ValueError(
                f"Stage-1 hybrid model must have exactly 2 labels, got {self.label_classes}."
            )
        if self.positive_label not in self.label_classes or self.negative_label not in self.label_classes:
            raise ValueError(
                "Stage-1 hybrid model metadata must include positive/negative labels present in label_classes."
            )

    def predict(self, text: str, post_text: str | None = None) -> dict[str, Any]:
        normalized_text = normalize_text(text)
        if not normalized_text:
            raise ValueError("Text must not be empty.")

        model_text = prepare_model_text(normalized_text, self.model_text_transform)
        hybrid_proba = self._predict_hybrid_proba(normalized_text, model_text, post_text=post_text)
        return _build_binary_prediction_result(
            self.label_classes,
            hybrid_proba[0],
            positive_label=self.positive_label,
            negative_label=self.negative_label,
            decision_threshold=self.decision_threshold,
            model_version=self.model_version,
            model_text=model_text,
        )


class HybridLlmPredictor(HybridTextStatsPredictor):
    def predict(self, text: str, post_text: str | None = None) -> dict[str, Any]:
        normalized_text = normalize_text(text)
        if not normalized_text:
            raise ValueError("Text must not be empty.")

        model_text = prepare_model_text(normalized_text, self.model_text_transform)
        hybrid_proba = self._predict_hybrid_proba(normalized_text, model_text, post_text=post_text)
        scores = _build_sorted_scores(self.label_classes, hybrid_proba[0])

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
        stage1_metadata = _read_model_metadata(stage1_model_dir)
        if stage1_metadata.get("xgb_model_filename"):
            self.stage1 = HybridBinaryPredictor(stage1_model_dir, device)
        else:
            self.stage1 = KcBertBinaryPredictor(stage1_model_dir, device)
        self.ai_threshold = ai_threshold
        self.llm_threshold = llm_threshold
        self.llm_confidence_threshold = llm_confidence_threshold
        self.stage1_human_cutoff = float(
            getattr(self.stage1, "decision_threshold", stage1_metadata.get("recommended_ai_threshold", 0.5))
        )
        self.stage1_reason_name = (
            "stage1 kcbert+stats hybrid" if self.stage1.uses_stats_model else "stage1 kcbert"
        )
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

    def _resolve_stage1_input_text(
        self,
        text: str,
        stage1_text: str | None = None,
    ) -> str:
        stage1_input_text = normalize_text(stage1_text) or text
        if self.stage1.metadata.get("strip_leading_mentions", True):
            stripped = strip_leading_reply_mentions(stage1_input_text)
            if stripped:
                stage1_input_text = stripped
        return stage1_input_text

    def _generate_xai_reason(self, text: str, ai_score: float, pred_label: str) -> str:
        attributions = []
        try:
            attributions = self.stage1.get_attention_attribution_for_text(text)
        except Exception:
            pass

        cleaned = text.strip()
        length = len(cleaned)

        has_laughter = bool(re.search(r"([ㅋㅎㅠㅜㅇ])\1+", cleaned))
        ends_with_period = cleaned.endswith(".") or cleaned.endswith("요.") or cleaned.endswith("다.")
        space_count = cleaned.count(" ")
        space_ratio = space_count / length if length > 0 else 0.0

        pct = round(ai_score * 100)
        human_pct = 100 - pct

        attention_text = ""
        if attributions:
            top_tokens = attributions[:2]
            token_phrases = [
                f"'{item['token']}'(기여도 {round(item['weight'] * 100, 1)}%)"
                for item in top_tokens
            ]
            attention_text = (
                "실제 KcBERT 모델이 판단 과정에서 주목한 핵심 맥락 단어는 "
                + ", ".join(token_phrases)
                + " 입니다. "
            )

        if pred_label == "ai":
            reasons = []
            if ends_with_period and not has_laughter:
                reasons.append(
                    "정중한 마침표 규격과 종결식 표준형 구조가 문법적으로 완벽한 균형을 이룹니다."
                )
            if 0.12 <= space_ratio <= 0.18:
                reasons.append(
                    f"전체 음절 수 대비 띄어쓰기 비율({round(space_ratio * 100)}%)이 "
                    "대단히 규칙적이고 완결도가 높습니다."
                )

            if not reasons:
                reasons.append(
                    "비격식 표현이나 일상 오탈자가 전혀 드러나지 않는 "
                    "전형적인 인공 생성 템플릿의 양상을 띱니다."
                )

            reason_str = " ".join(reasons)
            return (
                f"맞춤법과 띄어쓰기 오류가 단 하나도 없고 {reason_str}\n\n"
                f"{attention_text}이러한 어휘 기여도와 인공신경망의 가중치 흐름을 종합하여 "
                f"최종 AI 작성 글(의심 확률 {pct}%)로 강력히 의심됩니다."
            )

        if pred_label == "uncertain":
            return (
                "이 댓글은 일상적인 네티즌 표현이 일부 섞여 있으나, "
                "구조적으로 정돈된 완결성이 동시에 감지됩니다.\n\n"
                f"{attention_text}모델 내 셀프 어텐션 레이어가 복합적인 문체 자질을 "
                f"동시에 학습하고 있음을 인지하였기에 AI 의심 경계 단계(의심 확률 {pct}%)로 판단됩니다."
            )

        reasons = []
        if has_laughter:
            reasons.append(
                "인터넷 소통용 비격식 감정 자음(ㅋㅋㅋ, ㅎㅎ 등)이 자유롭게 배치되어 "
                "인위적 정형성을 이탈하고 있습니다."
            )
        if length < 15:
            reasons.append(
                f"문법의 구조적 규칙에서 벗어난 모바일 실생활 밀착형 초단문({length}자) 구조를 보입니다."
            )

        if not reasons:
            reasons.append(
                "주어와 서술어의 생략이 부드럽고 실제 사용자가 자발적으로 남긴 "
                "고유의 말버릇 양상을 담고 있습니다."
            )

        reason_str = " ".join(reasons)
        return (
            f"문맥 상 실제 네티즌 특유의 자연스럽고 비정형적인 구어체 흐름이 감지되며, {reason_str}\n\n"
            f"{attention_text}모델 판단 결과 기계 학습의 전형적인 작위 문체에서 완전히 이탈하여 "
            f"정상적인 인간 작성 댓글(신뢰도 {human_pct}%)로 안전하게 판명됩니다."
        )

    def predict(
        self,
        text: str,
        post_text: str | None = None,
        stage1_text: str | None = None,
    ) -> dict[str, Any]:
        stage1_input_text = self._resolve_stage1_input_text(text, stage1_text)
        stage1_result = self.stage1.predict(stage1_input_text, post_text=post_text)
        ai_score = float(stage1_result["ai_score"])
        risk_level = self._compute_risk_level(ai_score)
        xai_text = stage1_input_text or text

        if ai_score < self.stage1_human_cutoff:
            pred_label = "human"
            return {
                "pred_label": pred_label,
                "confidence": float(stage1_result["confidence"]),
                "ai_score": ai_score,
                "top2": stage1_result["top2"],
                "risk_level": risk_level,
                "reason": self._generate_xai_reason(xai_text, ai_score, pred_label),
                "stage1_pred_label": "human",
                "stage1_model_version": self.stage1_model_version,
                "stage2_pred_label": None,
                "stage2_model_version": self.stage2_model_version,
            }

        if ai_score < self.ai_threshold:
            pred_label = "uncertain"
            return {
                "pred_label": pred_label,
                "confidence": float(stage1_result["confidence"]),
                "ai_score": ai_score,
                "top2": stage1_result["top2"],
                "risk_level": risk_level,
                "reason": self._generate_xai_reason(xai_text, ai_score, pred_label),
                "stage1_pred_label": "uncertain",
                "stage1_model_version": self.stage1_model_version,
                "stage2_pred_label": None,
                "stage2_model_version": self.stage2_model_version,
            }

        if ai_score < self.llm_threshold:
            pred_label = "ai"
            return {
                "pred_label": pred_label,
                "confidence": float(stage1_result["confidence"]),
                "ai_score": ai_score,
                "top2": stage1_result["top2"],
                "risk_level": risk_level,
                "reason": self._generate_xai_reason(xai_text, ai_score, pred_label),
                "stage1_pred_label": "ai",
                "stage1_model_version": self.stage1_model_version,
                "stage2_pred_label": None,
                "stage2_model_version": self.stage2_model_version,
            }

        if self.stage2 is None:
            pred_label = "ai"
            return {
                "pred_label": pred_label,
                "confidence": float(stage1_result["confidence"]),
                "ai_score": ai_score,
                "top2": stage1_result["top2"],
                "risk_level": risk_level,
                "reason": self._generate_xai_reason(xai_text, ai_score, pred_label),
                "stage1_pred_label": "ai",
                "stage1_model_version": self.stage1_model_version,
                "stage2_pred_label": None,
                "stage2_model_version": None,
            }

        stage2_result = self.stage2.predict(text, post_text=post_text)
        if float(stage2_result["confidence"]) < self.llm_confidence_threshold:
            pred_label = "ai"
            return {
                "pred_label": pred_label,
                "confidence": float(stage1_result["confidence"]),
                "ai_score": ai_score,
                "top2": stage2_result["top2"],
                "risk_level": risk_level,
                "reason": self._generate_xai_reason(xai_text, ai_score, pred_label),
                "stage1_pred_label": "ai",
                "stage1_model_version": self.stage1_model_version,
                "stage2_pred_label": None,
                "stage2_model_version": self.stage2_model_version,
            }

        pred_label = stage2_result["pred_label"]
        return {
            "pred_label": pred_label,
            "confidence": float(stage2_result["confidence"]),
            "ai_score": ai_score,
            "top2": stage2_result["top2"],
            "risk_level": risk_level,
            "reason": self._generate_xai_reason(xai_text, ai_score, pred_label),
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
