from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.preprocess import extract_stat_features_from_text, prepare_model_text


OUTPUT_COLUMNS = [
    "post_text",
    "reply_text",
    "label",
    "transformed_text",
    "model_source",
    "split",
    "transformed_text_original",
    "transformed_text_prev",
    "rep_ratio",
    "has_laughter_rep",
    "punc_total_count",
    "punc_unique_count",
    "endswith_period",
    "endswith_yo",
    "last_char_is_hangul",
    "endswith_da",
    "has_enter",
    "enter_cnt",
    "space_cnt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare manually reviewed human hard negatives into training CSV format."
    )
    parser.add_argument("--input-path", required=True, help="Reviewed CSV path.")
    parser.add_argument("--output-path", required=True, help="Prepared CSV output path.")
    parser.add_argument(
        "--base-dataset-path",
        default="",
        help="Optional original dataset path for merge/dedup.",
    )
    parser.add_argument(
        "--merged-output-path",
        default="",
        help="Optional merged dataset output path.",
    )
    parser.add_argument(
        "--reply-col",
        default="reply_text",
        help="Input CSV column containing comment text.",
    )
    parser.add_argument(
        "--post-col",
        default="post_text",
        help="Optional input CSV column containing parent post text.",
    )
    parser.add_argument(
        "--review-col",
        default="review_label",
        help="Optional input CSV column containing reviewed label.",
    )
    parser.add_argument(
        "--human-value",
        default="human",
        help="Value in review-col that means the row is a verified human comment.",
    )
    return parser.parse_args()


def normalize_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace("\r\n", "\n").str.replace("\r", "\n").str.strip()


def build_output_frame(frame: pd.DataFrame, reply_col: str, post_col: str) -> pd.DataFrame:
    reply_text = normalize_series(frame[reply_col])
    post_text = normalize_series(frame[post_col]) if post_col in frame.columns else pd.Series([""] * len(frame))

    rows: list[dict[str, object]] = []
    for post, reply in zip(post_text.tolist(), reply_text.tolist()):
        if not reply:
            continue
        transformed = prepare_model_text(reply)
        stats = extract_stat_features_from_text(reply)
        rows.append(
            {
                "post_text": post,
                "reply_text": reply,
                "label": 0,
                "transformed_text": transformed,
                "model_source": "human",
                "split": "train",
                "transformed_text_original": transformed,
                "transformed_text_prev": transformed,
                "rep_ratio": stats["rep_ratio"],
                "has_laughter_rep": int(stats["has_laughter_rep"]),
                "punc_total_count": stats["punc_total_count"],
                "punc_unique_count": stats["punc_unique_count"],
                "endswith_period": int(stats["endswith_period"]),
                "endswith_yo": int(stats["endswith_yo"]),
                "last_char_is_hangul": int(stats["last_char_is_hangul"]),
                "endswith_da": int(stats["endswith_da"]),
                "has_enter": int(stats["has_enter"]),
                "enter_cnt": int(stats["enter_cnt"]),
                "space_cnt": int(stats["space_cnt"]),
            }
        )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def main() -> None:
    args = parse_args()

    reviewed = pd.read_csv(args.input_path, encoding="utf-8-sig")
    if args.reply_col not in reviewed.columns:
        raise ValueError(f"Missing required input column: {args.reply_col}")

    filtered = reviewed.copy()
    if args.review_col in filtered.columns:
        filtered[args.review_col] = normalize_series(filtered[args.review_col]).str.lower()
        filtered = filtered.loc[filtered[args.review_col].eq(args.human_value.lower())].copy()

    prepared = build_output_frame(filtered, reply_col=args.reply_col, post_col=args.post_col)
    prepared = prepared.drop_duplicates(subset="reply_text", keep="first").reset_index(drop=True)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Prepared hard negatives: {len(prepared)} rows -> {output_path}")

    if args.base_dataset_path and args.merged_output_path:
        base = pd.read_csv(args.base_dataset_path, encoding="utf-8-sig")
        merged = pd.concat([base, prepared], ignore_index=True)
        if "reply_text" in merged.columns:
            merged["__dedup_key"] = normalize_series(merged["reply_text"])
            merged = merged.drop_duplicates(subset="__dedup_key", keep="first").drop(columns="__dedup_key")

        merged_output_path = Path(args.merged_output_path)
        merged_output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(merged_output_path, index=False, encoding="utf-8-sig")
        print(f"Merged dataset: {len(merged)} rows -> {merged_output_path}")


if __name__ == "__main__":
    main()
