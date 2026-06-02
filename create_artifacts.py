import json
from pathlib import Path

output_dir = Path("saved_models/stage1_kcbert_binary")

artifacts = {
    "model_version": "stage1_kcbert_binary",
    "task_type": "binary_human_ai",
    "text_model_subdir": "bert_classifier",
    "max_length": 128,
    "text_col": "transformed_text",
    "label_col": "model_source",
    "source_col": "model_source",
    "label_classes": ["ai", "human"],
    "positive_label": "ai",
    "negative_label": "human",
    "recommended_ai_threshold": 0.5,
    "model_text_transform": "transformed_text_or_raw_to_special_tokens",
}

with (output_dir / "artifacts.json").open("w", encoding="utf-8") as f:
    json.dump(artifacts, f, ensure_ascii=False, indent=2)

print(f"artifacts.json 생성 완료: {output_dir / 'artifacts.json'}")
