from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


class KcBertBinaryPredictor(BaseTextPredictor):
    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        super().__init__(model_dir, device)
        self.positive_label = self.metadata.get("positive_label", "ai")
        self.negative_label = self.metadata.get("negative_label", "human")

    def predict(self, text: str) -> dict[str, Any]:
        normalized_text = normalize_text(text)
        if not normalized_text:
            raise ValueError("Text must not be empty.")

        model_text = prepare_model_text(normalized_text)
        probabilities = self._predict_text_proba(model_text)[0]
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
        text_proba = self._predict_text_proba(model_text)
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

    def predict(self, text: str) -> dict[str, Any]:
        stage1_result = self.stage1.predict(text)
        ai_score = float(stage1_result["ai_score"])
        risk_level = self._compute_risk_level(ai_score)

        if ai_score < 0.5:
            return {
                "pred_label": "human",
                "confidence": float(stage1_result["confidence"]),
                "ai_score": ai_score,
                "top2": stage1_result["top2"],
                "risk_level": risk_level,
                "reason": (
                    f"stage1 kcbert predicted human with ai probability {ai_score:.2f}"
                ),
                "stage1_pred_label": "human",
                "stage1_model_version": self.stage1_model_version,
                "stage2_pred_label": None,
                "stage2_model_version": self.stage2_model_version,
            }

        if ai_score < self.ai_threshold:
            return {
                "pred_label": "uncertain",
                "confidence": float(stage1_result["confidence"]),
                "ai_score": ai_score,
                "top2": stage1_result["top2"],
                "risk_level": risk_level,
                "reason": (
                    f"stage1 ai probability {ai_score:.2f} is above human cutoff 0.50 "
                    f"but below ai threshold {self.ai_threshold:.2f}"
                ),
                "stage1_pred_label": "uncertain",
                "stage1_model_version": self.stage1_model_version,
                "stage2_pred_label": None,
                "stage2_model_version": self.stage2_model_version,
            }

        if ai_score < self.llm_threshold:
            return {
                "pred_label": "ai",
                "confidence": float(stage1_result["confidence"]),
                "ai_score": ai_score,
                "top2": stage1_result["top2"],
                "risk_level": risk_level,
                "reason": (
                    f"stage1 kcbert predicted ai with probability {ai_score:.2f}, "
                    f"below llm threshold {self.llm_threshold:.2f}"
                ),
                "stage1_pred_label": "ai",
                "stage1_model_version": self.stage1_model_version,
                "stage2_pred_label": None,
                "stage2_model_version": self.stage2_model_version,
            }

        if self.stage2 is None:
            return {
                "pred_label": "ai",
                "confidence": float(stage1_result["confidence"]),
                "ai_score": ai_score,
                "top2": stage1_result["top2"],
                "risk_level": risk_level,
                "reason": (
                    f"stage1 kcbert predicted ai with probability {ai_score:.2f}, "
                    "but stage2 llm model is not loaded"
                ),
                "stage1_pred_label": "ai",
                "stage1_model_version": self.stage1_model_version,
                "stage2_pred_label": None,
                "stage2_model_version": None,
            }

        stage2_result = self.stage2.predict(text)
        if float(stage2_result["confidence"]) < self.llm_confidence_threshold:
            return {
                "pred_label": "ai",
                "confidence": float(stage1_result["confidence"]),
                "ai_score": ai_score,
                "top2": stage2_result["top2"],
                "risk_level": risk_level,
                "reason": (
                    f"stage1 ai probability {ai_score:.2f} passed threshold "
                    f"{self.llm_threshold:.2f}, but stage2 confidence "
                    f"{stage2_result['confidence']:.2f} was below "
                    f"{self.llm_confidence_threshold:.2f}"
                ),
                "stage1_pred_label": "ai",
                "stage1_model_version": self.stage1_model_version,
                "stage2_pred_label": None,
                "stage2_model_version": self.stage2_model_version,
            }

        return {
            "pred_label": stage2_result["pred_label"],
            "confidence": float(stage2_result["confidence"]),
            "ai_score": ai_score,
            "top2": stage2_result["top2"],
            "risk_level": risk_level,
            "reason": (
                f"stage1 ai probability {ai_score:.2f} passed threshold "
                f"{self.llm_threshold:.2f}; stage2 predicted {stage2_result['pred_label']}"
            ),
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
