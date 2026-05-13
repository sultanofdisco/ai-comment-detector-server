from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from server.config import settings
from server.inference import TwoStagePredictor
from server.schemas import HealthResponse, ModelUsageResponse, PredictRequest, PredictResponse
from server.stats import aggregate_prediction_usage, append_prediction_log


app = FastAPI(
    title="AI Comment Detector Server",
    version="0.1.0",
    description="Week-1 hybrid inference server for AI comment detection.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_predictor() -> TwoStagePredictor:
    return TwoStagePredictor(
        settings.stage1_model_dir,
        settings.stage2_model_dir,
        settings.device,
        settings.ai_threshold,
        settings.llm_threshold,
        settings.llm_confidence_threshold,
    )


@app.on_event("startup")
def warm_up_model() -> None:
    try:
        get_predictor()
    except FileNotFoundError:
        pass


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return HealthResponse(
        status="ok",
        stage1_model_dir=str(settings.stage1_model_dir),
        stage2_model_dir=str(settings.stage2_model_dir),
        device=settings.device,
        ai_threshold=settings.ai_threshold,
        llm_threshold=settings.llm_threshold,
        llm_confidence_threshold=settings.llm_confidence_threshold,
        stage1_model_version=predictor.stage1_model_version,
        stage2_model_version=predictor.stage2_model_version,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        predictor = get_predictor()
        result = predictor.predict(
            request.text,
            post_text=request.post_text,
            stage1_text=request.stage1_text,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    log_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "comment_id": request.comment_id,
        "author_id": request.author_id,
        "url": request.url,
        "client_timestamp": request.timestamp,
        "stage1_text_provided": bool(request.stage1_text),
        "stage1_text_differs_from_text": bool(
            request.stage1_text and request.stage1_text.strip() != request.text.strip()
        ),
        "final_pred_label": result["pred_label"],
        "stage1_pred_label": result["stage1_pred_label"],
        "stage2_pred_label": result["stage2_pred_label"],
        "confidence": result["confidence"],
        "ai_score": result["ai_score"],
        "risk_level": result["risk_level"],
        "ai_threshold": settings.ai_threshold,
        "llm_threshold": settings.llm_threshold,
        "llm_confidence_threshold": settings.llm_confidence_threshold,
        "stage1_model_version": result["stage1_model_version"],
        "stage2_model_version": result["stage2_model_version"],
    }
    append_prediction_log(settings.prediction_log_path, log_record)
    return PredictResponse(
        comment_id=request.comment_id,
        pred_label=result["pred_label"],
        confidence=result["confidence"],
        ai_score=result["ai_score"],
        top2=result["top2"],
        risk_level=result["risk_level"],
        reason=result["reason"],
    )


@app.get("/admin/model-usage", response_model=ModelUsageResponse)
def model_usage() -> ModelUsageResponse:
    usage = aggregate_prediction_usage(settings.prediction_log_path)
    try:
        predictor = get_predictor()
        usage["stage2_model_version"] = predictor.stage2_model_version
    except FileNotFoundError:
        usage["stage2_model_version"] = None
    return ModelUsageResponse(**usage)
