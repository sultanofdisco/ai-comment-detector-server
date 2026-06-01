from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FALSE_POSITIVE_FIELDNAMES = [
    "dedupe_key",
    "exported_at",
    "export_source",
    "cache_key",
    "comment_id",
    "author_id",
    "timestamp",
    "url",
    "text",
    "text_length",
    "root_post_id",
    "root_post_author_id",
    "root_post_timestamp",
    "root_post_url",
    "root_post_text",
    "root_post_text_length",
    "pred_label",
    "confidence",
    "ai_score",
    "risk_level",
]


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


def _normalize_export_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _build_false_positive_dedupe_key(record: dict[str, Any]) -> str:
    parts = [
        _normalize_export_text(record.get("comment_id")),
        _normalize_export_text(record.get("author_id")),
        _normalize_export_text(record.get("timestamp")),
        _normalize_export_text(record.get("url")),
        _normalize_export_text(record.get("text")),
    ]
    payload = "||".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _false_positive_exists(csv_path: Path, dedupe_key: str) -> bool:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return False

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("dedupe_key") == dedupe_key:
                return True
    return False


def append_false_positive_feedback(csv_path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    exported_at = datetime.now(timezone.utc).isoformat()
    record = {
        "export_source": _normalize_export_text(payload.get("export_source")) or "extension",
        "cache_key": _normalize_export_text(payload.get("cache_key")),
        "comment_id": _normalize_export_text(payload.get("comment_id")),
        "author_id": _normalize_export_text(payload.get("author_id")),
        "timestamp": _normalize_export_text(payload.get("timestamp")),
        "url": _normalize_export_text(payload.get("url")),
        "text": _normalize_export_text(payload.get("text")),
        "root_post_id": _normalize_export_text(payload.get("root_post_id")),
        "root_post_author_id": _normalize_export_text(payload.get("root_post_author_id")),
        "root_post_timestamp": _normalize_export_text(payload.get("root_post_timestamp")),
        "root_post_url": _normalize_export_text(payload.get("root_post_url")),
        "root_post_text": _normalize_export_text(payload.get("root_post_text")),
        "pred_label": _normalize_export_text(payload.get("pred_label")),
        "confidence": payload.get("confidence"),
        "ai_score": payload.get("ai_score"),
        "risk_level": _normalize_export_text(payload.get("risk_level")),
    }
    record["text_length"] = len(record["text"])
    record["root_post_text_length"] = len(record["root_post_text"])
    record["exported_at"] = exported_at
    record["dedupe_key"] = _build_false_positive_dedupe_key(record)

    if _false_positive_exists(path, record["dedupe_key"]):
        return {
            "status": "duplicate",
            "exported_at": exported_at,
            "file_path": str(path),
            "dedupe_key": record["dedupe_key"],
        }

    write_header = not path.exists() or path.stat().st_size == 0
    encoding = "utf-8-sig" if write_header else "utf-8"

    with path.open("a", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FALSE_POSITIVE_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in FALSE_POSITIVE_FIELDNAMES})

    return {
        "status": "saved",
        "exported_at": exported_at,
        "file_path": str(path),
        "dedupe_key": record["dedupe_key"],
    }
