from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.preprocess import (  # noqa: E402
    BODY_MENTION_MODEL_TEXT_TRANSFORM,
    LEGACY_MODEL_TEXT_TRANSFORM,
    STAT_FEATURES,
    X_MODEL_TEXT_TRANSFORM,
    add_model_special_tokens,
    build_stats_frame,
    normalize_text,
    prepare_model_text,
    strip_leading_reply_mentions,
)
from training.train_hybrid import build_xgb_model  # noqa: E402
from training.train_kcbert_binary import (  # noqa: E402
    TextDataset,
    WeightedTrainer,
    balance_train_binary_frame,
    build_threshold_grid,
    choose_decision_threshold,
    choose_dedup_column,
    compute_class_weights,
    compute_metrics,
    deduplicate_dataset,
    evaluate_binary_threshold,
    predict_probabilities,
    prepare_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train stage-1 KC-BERT + stats hybrid human/ai classifier."
    )
    parser.add_argument("--data-path", required=True, help="CSV path.")
    parser.add_argument("--output-dir", required=True, help="Directory to save trained artifacts.")
    parser.add_argument(
        "--text-col",
        default="transformed_text",
        help="Text column name used for KC-BERT input.",
    )
    parser.add_argument(
        "--raw-text-col",
        default="reply_text",
        help="Raw reply text column used for stats features.",
    )
    parser.add_argument(
        "--post-col",
        default="post_text",
        help="Parent post text column used for consistency features.",
    )
    parser.add_argument("--label-col", default="model_source", help="Label/source column name.")
    parser.add_argument(
        "--source-col",
        default="",
        help="Optional original AI source column for train balancing. Defaults to label-col.",
    )
    parser.add_argument("--split-col", default="split", help="Split column name.")
    parser.add_argument(
        "--dedup-col",
        default="",
        help="Optional text column for global dedup before splitting. Defaults to reply_text when available.",
    )
    parser.add_argument(
        "--disable-dedup",
        action="store_true",
        help="Disable global text dedup before splitting.",
    )
    parser.add_argument(
        "--disable-train-balancing",
        action="store_true",
        help="Disable 1:1 human/ai train balancing.",
    )
    parser.add_argument("--human-label", default="human", help="Label value treated as human.")
    parser.add_argument("--model-name", default="beomi/kcbert-base", help="HF model name.")
    parser.add_argument(
        "--model-text-transform",
        default=BODY_MENTION_MODEL_TEXT_TRANSFORM,
        choices=[
            BODY_MENTION_MODEL_TEXT_TRANSFORM,
            LEGACY_MODEL_TEXT_TRANSFORM,
            X_MODEL_TEXT_TRANSFORM,
        ],
        help="Pre-tokenization text transform to apply before encoding.",
    )
    parser.add_argument("--max-length", type=int, default=128, help="Max token length.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Eval batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio.")
    parser.add_argument(
        "--weight-grid",
        default="0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated candidate text weights for hybrid.",
    )
    parser.add_argument(
        "--stats-cols",
        default="",
        help="Optional comma-separated stats feature columns to use.",
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.5,
        help="Minimum AI decision threshold to search on validation.",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.99,
        help="Maximum AI decision threshold to search on validation.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Threshold search step size.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--keep-leading-mentions",
        action="store_true",
        help="Keep leading reply mentions in stage-1 training inputs instead of stripping them.",
    )
    parser.add_argument(
        "--force-text-weight",
        type=float,
        default=None,
        help=(
            "Optional fixed text-model weight to export. "
            "When set, the script reuses the validation-searched threshold for that weight "
            "instead of the auto-selected best weight."
        ),
    )
    return parser.parse_args()


def choose_best_weight_and_threshold(
    val_text_prob: np.ndarray,
    val_stats_prob: np.ndarray,
    y_true: np.ndarray,
    ai_index: int,
    human_index: int,
    weight_grid: list[float],
    threshold_grid: list[float],
) -> tuple[float, float, list[dict[str, float]], dict[str, float]]:
    search_results: list[dict[str, float]] = []
    best_weight = weight_grid[0]
    best_threshold = threshold_grid[0]
    best_metrics: dict[str, float] | None = None
    best_key: tuple[float, float, float, float, float] | None = None

    for text_weight in weight_grid:
        stats_weight = 1.0 - text_weight
        hybrid_prob = (text_weight * val_text_prob) + (stats_weight * val_stats_prob)
        threshold, _, metrics = choose_decision_threshold(
            hybrid_prob,
            y_true,
            ai_index=ai_index,
            human_index=human_index,
            threshold_grid=threshold_grid,
        )
        row = {
            "text_weight": round(text_weight, 4),
            "stats_weight": round(stats_weight, 4),
            **metrics,
        }
        search_results.append(row)

        selection_key = (
            metrics["macro_f1"],
            metrics["human_recall"],
            metrics["ai_precision"],
            metrics["accuracy"],
            text_weight,
        )
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_weight = text_weight
            best_threshold = threshold
            best_metrics = row

    assert best_metrics is not None
    return best_weight, best_threshold, search_results, best_metrics


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_path, encoding="utf-8-sig")

    source_col = args.source_col or args.label_col
    required_columns = {args.text_col, args.label_col, source_col}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = df.copy()
    df[args.label_col] = df[args.label_col].astype(str).str.lower().str.strip()
    if source_col != args.label_col:
        df[source_col] = df[source_col].astype(str).str.lower().str.strip()

    dedup_col = choose_dedup_column(df, args.dedup_col, args.text_col)
    strip_leading_mentions = not args.keep_leading_mentions
    raw_source_col = args.raw_text_col if args.raw_text_col in df.columns else args.text_col

    if strip_leading_mentions:
        df["__stage1_source_text"] = df[raw_source_col].map(strip_leading_reply_mentions)
        df["__dedup_source_text"] = df[raw_source_col].map(strip_leading_reply_mentions)
    else:
        df["__stage1_source_text"] = df[raw_source_col].map(normalize_text)
        df["__dedup_source_text"] = df[dedup_col].map(normalize_text)

    dedup_removed = 0
    if not args.disable_dedup:
        if strip_leading_mentions:
            df, dedup_removed = deduplicate_dataset(df, "__dedup_source_text", args.split_col)
        else:
            df, dedup_removed = deduplicate_dataset(df, dedup_col, args.split_col)

    df["__raw_text"] = df["__stage1_source_text"].map(normalize_text)
    if args.post_col in df.columns:
        df["__raw_post_text"] = df[args.post_col].map(normalize_text)
    else:
        df["__raw_post_text"] = ""
    df["__model_text"] = df["__stage1_source_text"].map(
        lambda text: prepare_model_text(text, args.model_text_transform)
    )
    df = df[df["__model_text"].ne("")].copy()
    df["binary_label"] = np.where(
        df[args.label_col].eq(args.human_label.lower()),
        "human",
        "ai",
    )

    feature_order = (
        [feature.strip() for feature in args.stats_cols.split(",") if feature.strip()]
        if args.stats_cols
        else STAT_FEATURES.copy()
    )
    extracted_stats = build_stats_frame(
        df["__raw_text"].tolist(),
        feature_order,
        post_texts=df["__raw_post_text"].tolist(),
    )
    for feature in feature_order:
        df[feature] = extracted_stats[feature].astype(float)

    train_df, val_df, test_df = prepare_splits(
        df,
        split_col=args.split_col,
        label_col="binary_label",
        random_state=args.random_state,
    )

    original_train_summary = (
        train_df[source_col].astype(str).value_counts().sort_index().to_dict()
    )
    balanced_train_summary: dict[str, object] | None = None
    if not args.disable_train_balancing:
        train_df, balanced_train_summary = balance_train_binary_frame(
            train_df,
            source_col=source_col,
            human_label="human",
            random_state=args.random_state,
        )

    label_classes = ["human", "ai"]
    label_to_id = {label: index for index, label in enumerate(label_classes)}
    train_df["target"] = train_df["binary_label"].map(label_to_id)
    val_df["target"] = val_df["binary_label"].map(label_to_id)
    test_df["target"] = test_df["binary_label"].map(label_to_id)

    xgb_model = build_xgb_model(num_labels=2, random_state=args.random_state)
    xgb_model.fit(train_df[feature_order], train_df["target"])
    stats_prob_val = xgb_model.predict_proba(val_df[feature_order])
    stats_prob_test = xgb_model.predict_proba(test_df[feature_order])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )
    num_added_special_tokens = add_model_special_tokens(
        tokenizer,
        bert_model,
        args.model_text_transform,
    )
    train_encodings = tokenizer(
        train_df["__model_text"].tolist(),
        truncation=True,
        max_length=args.max_length,
    )
    val_encodings = tokenizer(
        val_df["__model_text"].tolist(),
        truncation=True,
        max_length=args.max_length,
    )
    test_encodings = tokenizer(
        test_df["__model_text"].tolist(),
        truncation=True,
        max_length=args.max_length,
    )

    train_dataset = TextDataset(train_encodings, train_df["target"].tolist())
    val_dataset = TextDataset(val_encodings, val_df["target"].tolist())
    class_weights = compute_class_weights(train_df["target"].tolist(), num_labels=2)
    bert_output_dir = output_dir / "bert_classifier"
    training_args_kwargs = {
        "output_dir": str(output_dir / "trainer_runs"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "save_total_limit": 1,
        "logging_steps": 25,
        "report_to": [],
        "metric_for_best_model": "eval_accuracy",
        "greater_is_better": True,
    }
    try:
        training_args = TrainingArguments(
            evaluation_strategy="epoch",
            **training_args_kwargs,
        )
    except TypeError:
        training_args = TrainingArguments(
            eval_strategy="epoch",
            **training_args_kwargs,
        )

    trainer = WeightedTrainer(
        model=bert_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    print("Training stage-1 KC-BERT + stats hybrid classifier...")
    print(f"Dedup column: {dedup_col} | removed rows: {dedup_removed}")
    print(f"Original train source counts: {original_train_summary}")
    if balanced_train_summary is not None:
        print(f"Balanced train summary: {balanced_train_summary}")
    print(f"Stats features: {feature_order}")
    print(f"Model text transform: {args.model_text_transform}")
    print(f"Added special tokens: {num_added_special_tokens}")
    print(f"Class weights: {class_weights}")

    trainer.train()
    trainer.save_model(str(bert_output_dir))
    tokenizer.save_pretrained(str(bert_output_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_text_prob = predict_probabilities(
        trainer.model,
        tokenizer,
        val_df["__model_text"].tolist(),
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        device=device,
    )
    test_text_prob = predict_probabilities(
        trainer.model,
        tokenizer,
        test_df["__model_text"].tolist(),
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        device=device,
    )

    human_index = label_to_id["human"]
    ai_index = label_to_id["ai"]
    weight_grid = [float(value.strip()) for value in args.weight_grid.split(",") if value.strip()]
    threshold_grid = build_threshold_grid(
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
    )
    (
        best_text_weight,
        selected_threshold,
        validation_search,
        selected_val_metrics,
    ) = choose_best_weight_and_threshold(
        val_text_prob,
        stats_prob_val,
        val_df["target"].to_numpy(),
        ai_index=ai_index,
        human_index=human_index,
        weight_grid=weight_grid,
        threshold_grid=threshold_grid,
    )

    if args.force_text_weight is not None:
        forced_row = next(
            (
                row
                for row in validation_search
                if abs(float(row["text_weight"]) - float(args.force_text_weight)) < 1e-9
            ),
            None,
        )
        if forced_row is None:
            raise ValueError(
                f"Requested --force-text-weight={args.force_text_weight} is not in weight-grid {weight_grid}."
            )
        best_text_weight = float(forced_row["text_weight"])
        selected_threshold = float(forced_row["threshold"])
        selected_val_metrics = dict(forced_row)

    best_stats_weight = 1.0 - best_text_weight

    hybrid_test_prob = (best_text_weight * test_text_prob) + (best_stats_weight * stats_prob_test)
    default_metrics = evaluate_binary_threshold(
        hybrid_test_prob[:, ai_index],
        test_df["target"].to_numpy(),
        ai_index=ai_index,
        human_index=human_index,
        threshold=0.5,
    )
    tuned_metrics = evaluate_binary_threshold(
        hybrid_test_prob[:, ai_index],
        test_df["target"].to_numpy(),
        ai_index=ai_index,
        human_index=human_index,
        threshold=selected_threshold,
    )
    test_predictions = np.where(
        hybrid_test_prob[:, ai_index] >= selected_threshold,
        ai_index,
        human_index,
    )
    roc_auc = float(roc_auc_score(test_df["target"], hybrid_test_prob[:, ai_index]))
    report = classification_report(
        test_df["target"],
        test_predictions,
        target_names=label_classes,
        digits=4,
        output_dict=True,
    )

    print(f"Best text weight: {best_text_weight:.2f}")
    print(f"Best stats weight: {best_stats_weight:.2f}")
    print(f"Selected AI threshold from validation: {selected_threshold:.2f}")
    print(f"Validation selection metrics: {selected_val_metrics}")
    if args.force_text_weight is not None:
        print(
            "Forced deployment weight enabled: "
            f"text={best_text_weight:.2f}, stats={best_stats_weight:.2f}"
        )
    print(f"Test accuracy @ tuned threshold: {tuned_metrics['accuracy']:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")

    joblib.dump(xgb_model, output_dir / "xgb_model.joblib")

    metrics = {
        "task_type": "binary_human_ai_hybrid",
        "dedup": {
            "enabled": not args.disable_dedup,
            "dedup_col": dedup_col,
            "removed_rows": int(dedup_removed),
        },
        "train_balancing": {
            "enabled": not args.disable_train_balancing,
            "source_col": source_col,
            "original_train_source_counts": {
                str(key): int(value) for key, value in original_train_summary.items()
            },
            "balanced_train_summary": balanced_train_summary,
        },
        "class_weights": [round(weight, 4) for weight in class_weights],
        "validation_weight_threshold_search": validation_search,
        "selected_text_weight": round(best_text_weight, 4),
        "selected_stats_weight": round(best_stats_weight, 4),
        "selected_ai_threshold": round(selected_threshold, 4),
        "selected_val_metrics": selected_val_metrics,
        "forced_text_weight": (
            None if args.force_text_weight is None else round(float(args.force_text_weight), 4)
        ),
        "test_metrics_at_default_0_5": {
            key: round(value, 4) for key, value in default_metrics.items()
        },
        "test_metrics_at_selected_threshold": {
            key: round(value, 4) for key, value in tuned_metrics.items()
        },
        "test_accuracy": round(tuned_metrics["accuracy"], 4),
        "test_roc_auc": round(roc_auc, 4),
        "classification_report": report,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    artifacts = {
        "model_version": output_dir.name,
        "task_type": "binary_human_ai_hybrid",
        "text_model_subdir": "bert_classifier",
        "xgb_model_filename": "xgb_model.joblib",
        "text_weight": round(best_text_weight, 4),
        "stats_weight": round(best_stats_weight, 4),
        "stats_features": feature_order,
        "max_length": args.max_length,
        "text_col": args.text_col,
        "raw_text_col": args.raw_text_col,
        "post_col": args.post_col,
        "label_col": args.label_col,
        "source_col": source_col,
        "label_classes": label_classes,
        "positive_label": "ai",
        "negative_label": "human",
        "recommended_ai_threshold": round(selected_threshold, 4),
        "model_text_transform": args.model_text_transform,
        "strip_leading_mentions": strip_leading_mentions,
        "effective_text_source_col": raw_source_col if strip_leading_mentions else args.text_col,
    }
    with (output_dir / "artifacts.json").open("w", encoding="utf-8") as handle:
        json.dump(artifacts, handle, ensure_ascii=False, indent=2)

    label_map = {str(index): label for index, label in enumerate(label_classes)}
    with (output_dir / "label_map.json").open("w", encoding="utf-8") as handle:
        json.dump(label_map, handle, ensure_ascii=False, indent=2)

    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
