from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def append_prediction_log(log_path: str | Path, record: dict[str, Any]) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_prediction_records(log_path: str | Path) -> list[dict[str, Any]]:
    path = Path(log_path)
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def aggregate_prediction_usage(log_path: str | Path) -> dict[str, Any]:
    records = load_prediction_records(log_path)
    counts = Counter(
        record["stage2_pred_label"]
        for record in records
        if record.get("stage2_pred_label")
    )
    total_predictions = len(records)
    llm_classified_predictions = sum(counts.values())

    by_label = []
    for label, count in counts.most_common():
        ratio = (count / llm_classified_predictions) if llm_classified_predictions else 0.0
        by_label.append(
            {
                "label": label,
                "count": int(count),
                "ratio": round(ratio, 4),
            }
        )

    return {
        "total_predictions": int(total_predictions),
        "llm_classified_predictions": int(llm_classified_predictions),
        "by_label": by_label,
    }
