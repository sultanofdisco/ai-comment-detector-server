from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    stage1_model_dir: Path = Path(
        os.getenv(
            "STAGE1_MODEL_DIR",
            str(BASE_DIR / "saved_models" / "stage1_kcbert_binary"),
        )
    )
    stage2_model_dir: Path = Path(
        os.getenv(
            "STAGE2_MODEL_DIR",
            str(BASE_DIR / "saved_models" / "stage2_llm_hybrid"),
        )
    )
    prediction_log_path: Path = Path(
        os.getenv("PREDICTION_LOG_PATH", str(BASE_DIR / "logs" / "predictions.jsonl"))
    )
    ai_threshold: float = float(os.getenv("AI_THRESHOLD", "0.85"))
    llm_threshold: float = float(os.getenv("LLM_THRESHOLD", "0.92"))
    llm_confidence_threshold: float = float(
        os.getenv("LLM_CONFIDENCE_THRESHOLD", "0.97")
    )
    device: str = os.getenv(
        "MODEL_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    cors_origins: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        origins = os.getenv("CORS_ORIGINS", "*")
        object.__setattr__(
            self,
            "cors_origins",
            [origin.strip() for origin in origins.split(",") if origin.strip()],
        )


settings = Settings()
