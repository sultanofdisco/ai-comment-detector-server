from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    comment_id: str | None = Field(default=None, description="Client-side comment id.")
    author_id: str | None = Field(default=None, description="Comment author id.")
    text: str = Field(..., min_length=1, description="Comment text to analyze.")
    url: str | None = Field(default=None, description="Comment page URL.")
    timestamp: str | None = Field(default=None, description="Comment timestamp from frontend.")


class PredictResponse(BaseModel):
    comment_id: str | None = None
    pred_label: str
    confidence: float
    ai_score: float
    top2: list[tuple[str, float]]
    risk_level: str
    reason: str


class HealthResponse(BaseModel):
    status: str
    stage1_model_dir: str
    stage2_model_dir: str
    device: str
    ai_threshold: float
    llm_threshold: float
    llm_confidence_threshold: float
    stage1_model_version: str | None = None
    stage2_model_version: str | None = None


class UsageItem(BaseModel):
    label: str
    count: int
    ratio: float


class ModelUsageResponse(BaseModel):
    total_predictions: int
    llm_classified_predictions: int
    by_label: list[UsageItem]
    stage2_model_version: str | None = None


class ErrorResponse(BaseModel):
    detail: str
    extra: dict[str, Any] | None = None
