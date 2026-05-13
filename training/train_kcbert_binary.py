from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
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

from server.preprocess import (
    BODY_MENTION_MODEL_TEXT_TRANSFORM,
    LEGACY_MODEL_TEXT_TRANSFORM,
    X_MODEL_TEXT_TRANSFORM,
    add_model_special_tokens,
    normalize_text,
    prepare_model_text,
    strip_leading_reply_mentions,
)


SPLIT_DEDUP_PRIORITY = {
    "test": 0,
    "val": 1,
    "valid": 1,
    "validation": 1,
    "train": 2,
}


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict[str, list[int]], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(value[index], dtype=torch.long)
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.labels)


class WeightedTrainer(Trainer):
    def __init__(
        self,
        *args,
        class_weights: list[float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = (
            None
            if class_weights is None
            else torch.tensor(class_weights, dtype=torch.float)
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        weights = (
            None
            if self.class_weights is None
            else self.class_weights.to(logits.device)
        )
        loss = F.cross_entropy(logits, labels, weight=weights)
        if return_outputs:
            return loss, outputs
        return loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train stage-1 KC-BERT binary human/ai classifier."
    )
    parser.add_argument("--data-path", required=True, help="CSV path.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save trained artifacts.",
    )
    parser.add_argument(
        "--text-col",
        default="transformed_text",
        help="Text column name used for model input.",
    )
    parser.add_argument(
        "--label-col",
        default="model_source",
        help="Label/source column name.",
    )
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
    parser.add_argument(
        "--human-label",
        default="human",
        help="Label value treated as human.",
    )
    parser.add_argument(
        "--model-name",
        default="beomi/kcbert-base",
        help="HF model name.",
    )
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
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Max token length.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=16,
        help="Train batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Eval batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio.",
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
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--keep-leading-mentions",
        action="store_true",
        help="Keep leading reply mentions in stage-1 training inputs instead of stripping them.",
    )
    return parser.parse_args()


def safe_split(
    frame: pd.DataFrame,
    test_size: float,
    label_col: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_frame, test_frame = train_test_split(
            frame,
            test_size=test_size,
            stratify=frame[label_col],
            random_state=random_state,
        )
    except ValueError:
        train_frame, test_frame = train_test_split(
            frame,
            test_size=test_size,
            random_state=random_state,
        )
    return train_frame.copy(), test_frame.copy()


def prepare_splits(
    df: pd.DataFrame,
    split_col: str,
    label_col: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split_col in df.columns:
        normalized_split = df[split_col].astype(str).str.lower().str.strip()
        train_mask = normalized_split.eq("train")
        test_mask = normalized_split.eq("test")
        val_mask = normalized_split.isin(["val", "valid", "validation"])

        if train_mask.any() and test_mask.any() and val_mask.any():
            return (
                df.loc[train_mask].copy(),
                df.loc[val_mask].copy(),
                df.loc[test_mask].copy(),
            )

        if train_mask.any() and test_mask.any():
            base_train = df.loc[train_mask].copy()
            test_df = df.loc[test_mask].copy()
            train_df, val_df = safe_split(
                base_train,
                test_size=0.1,
                label_col=label_col,
                random_state=random_state,
            )
            return train_df, val_df, test_df

    train_val_df, test_df = safe_split(
        df,
        test_size=0.2,
        label_col=label_col,
        random_state=random_state,
    )
    train_df, val_df = safe_split(
        train_val_df,
        test_size=0.125,
        label_col=label_col,
        random_state=random_state,
    )
    return train_df, val_df, test_df


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


def choose_dedup_column(
    frame: pd.DataFrame,
    requested_col: str,
    text_col: str,
) -> str:
    if requested_col:
        if requested_col not in frame.columns:
            raise ValueError(f"Dedup column '{requested_col}' does not exist.")
        return requested_col
    if "reply_text" in frame.columns:
        return "reply_text"
    return text_col


def deduplicate_dataset(
    frame: pd.DataFrame,
    dedup_col: str,
    split_col: str,
) -> tuple[pd.DataFrame, int]:
    work = frame.copy()
    work["__dedup_key"] = work[dedup_col].map(normalize_text)
    work["__row_order"] = np.arange(len(work))

    if split_col in work.columns:
        normalized_split = work[split_col].astype(str).str.lower().str.strip()
        work["__split_priority"] = normalized_split.map(SPLIT_DEDUP_PRIORITY).fillna(3)
        sort_columns = ["__dedup_key", "__split_priority", "__row_order"]
    else:
        sort_columns = ["__dedup_key", "__row_order"]

    before = len(work)
    work = work.sort_values(sort_columns)
    work = work.drop_duplicates(subset="__dedup_key", keep="first")
    work = work.sort_values("__row_order")
    removed = before - len(work)

    drop_columns = ["__dedup_key", "__row_order"]
    if "__split_priority" in work.columns:
        drop_columns.append("__split_priority")
    return work.drop(columns=drop_columns), removed


def balance_train_binary_frame(
    frame: pd.DataFrame,
    source_col: str,
    human_label: str,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if source_col not in frame.columns:
        raise ValueError(
            f"Train balancing requires source column '{source_col}', but it was not found."
        )

    human_df = frame.loc[frame["binary_label"].eq(human_label)].copy()
    ai_df = frame.loc[frame["binary_label"].ne(human_label)].copy()

    if human_df.empty or ai_df.empty:
        raise ValueError("Train balancing requires both human and ai examples.")

    ai_source_counts = ai_df[source_col].value_counts().sort_index()
    ai_sources = ai_source_counts.index.tolist()
    if not ai_sources:
        raise ValueError("No AI source labels were found for balanced train sampling.")

    per_source_quota = min(len(human_df) // len(ai_sources), int(ai_source_counts.min()))
    if per_source_quota < 1:
        raise ValueError("Balanced train sampling could not allocate at least one row per AI source.")

    target_human_count = per_source_quota * len(ai_sources)
    balanced_human = human_df.sample(
        n=target_human_count,
        random_state=random_state,
    )

    balanced_ai_parts = []
    for offset, source in enumerate(ai_sources):
        sampled = ai_df.loc[ai_df[source_col].eq(source)].sample(
            n=per_source_quota,
            random_state=random_state + offset,
        )
        balanced_ai_parts.append(sampled)

    balanced_ai = pd.concat(balanced_ai_parts, ignore_index=True)
    balanced = pd.concat([balanced_human, balanced_ai], ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    summary: dict[str, object] = {
        "human_count": int(len(balanced_human)),
        "ai_total_count": int(len(balanced_ai)),
        "per_ai_source_count": int(per_source_quota),
        "ai_source_counts": {
            str(source): int(per_source_quota) for source in ai_sources
        },
    }
    return balanced, summary


def compute_class_weights(targets: list[int], num_labels: int) -> list[float]:
    counts = np.bincount(np.asarray(targets, dtype=int), minlength=num_labels)
    total = float(counts.sum())
    weights = total / (num_labels * np.maximum(counts, 1))
    return [float(weight) for weight in weights]


def predict_probabilities(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    all_probs = []
    model.eval()
    model.to(device)

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

    return np.vstack(all_probs)


def build_threshold_grid(
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> list[float]:
    if threshold_step <= 0:
        raise ValueError("threshold-step must be greater than 0.")
    if threshold_min <= 0 or threshold_max >= 1 or threshold_min >= threshold_max:
        raise ValueError("Threshold range must satisfy 0 < min < max < 1.")

    values = []
    current = threshold_min
    while current <= threshold_max + 1e-9:
        values.append(round(current, 4))
        current += threshold_step
    return values


def evaluate_binary_threshold(
    ai_probabilities: np.ndarray,
    y_true: np.ndarray,
    ai_index: int,
    human_index: int,
    threshold: float,
) -> dict[str, float]:
    predictions = np.where(ai_probabilities >= threshold, ai_index, human_index)
    human_precision, human_recall, human_f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=[human_index],
        zero_division=0,
    )
    ai_precision, ai_recall, ai_f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=[ai_index],
        zero_division=0,
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "human_precision": float(human_precision[0]),
        "human_recall": float(human_recall[0]),
        "human_f1": float(human_f1[0]),
        "ai_precision": float(ai_precision[0]),
        "ai_recall": float(ai_recall[0]),
        "ai_f1": float(ai_f1[0]),
    }


def choose_decision_threshold(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    ai_index: int,
    human_index: int,
    threshold_grid: list[float],
) -> tuple[float, list[dict[str, float]], dict[str, float]]:
    search_results = []
    best_metrics: dict[str, float] | None = None
    best_threshold = threshold_grid[0]
    best_key: tuple[float, float, float, float] | None = None

    for threshold in threshold_grid:
        metrics = evaluate_binary_threshold(
            probabilities[:, ai_index],
            y_true,
            ai_index,
            human_index,
            threshold,
        )
        rounded = {key: round(value, 4) for key, value in metrics.items()}
        search_results.append(rounded)

        selection_key = (
            metrics["macro_f1"],
            metrics["human_recall"],
            metrics["ai_precision"],
            metrics["threshold"],
        )
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_threshold = threshold
            best_metrics = rounded

    assert best_metrics is not None
    return best_threshold, search_results, best_metrics


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
    raw_source_col = "reply_text" if "reply_text" in df.columns else args.text_col
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

    df[args.text_col] = df["__stage1_source_text"].map(
        lambda text: prepare_model_text(text, args.model_text_transform)
    )
    df = df[df[args.text_col].ne("")].copy()
    df["binary_label"] = np.where(
        df[args.label_col].eq(args.human_label.lower()),
        "human",
        "ai",
    )

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
        train_df[args.text_col].tolist(),
        truncation=True,
        max_length=args.max_length,
    )
    val_encodings = tokenizer(
        val_df[args.text_col].tolist(),
        truncation=True,
        max_length=args.max_length,
    )
    test_encodings = tokenizer(
        test_df[args.text_col].tolist(),
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

    print("Training stage-1 KC-BERT binary classifier...")
    print(f"Dedup column: {dedup_col} | removed rows: {dedup_removed}")
    print(f"Original train source counts: {original_train_summary}")
    if balanced_train_summary is not None:
        print(f"Balanced train summary: {balanced_train_summary}")
    print(f"Model text transform: {args.model_text_transform}")
    print(f"Added special tokens: {num_added_special_tokens}")
    print(f"Class weights: {class_weights}")

    trainer.train()
    trainer.save_model(str(bert_output_dir))
    tokenizer.save_pretrained(str(bert_output_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_probabilities = predict_probabilities(
        trainer.model,
        tokenizer,
        val_df[args.text_col].tolist(),
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        device=device,
    )
    test_probabilities = predict_probabilities(
        trainer.model,
        tokenizer,
        test_df[args.text_col].tolist(),
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        device=device,
    )

    human_index = label_to_id["human"]
    ai_index = label_to_id["ai"]
    threshold_grid = build_threshold_grid(
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
    )
    selected_threshold, threshold_search, selected_val_metrics = choose_decision_threshold(
        val_probabilities,
        val_df["target"].to_numpy(),
        ai_index=ai_index,
        human_index=human_index,
        threshold_grid=threshold_grid,
    )

    default_metrics = evaluate_binary_threshold(
        test_probabilities[:, ai_index],
        test_df["target"].to_numpy(),
        ai_index=ai_index,
        human_index=human_index,
        threshold=0.5,
    )
    tuned_metrics = evaluate_binary_threshold(
        test_probabilities[:, ai_index],
        test_df["target"].to_numpy(),
        ai_index=ai_index,
        human_index=human_index,
        threshold=selected_threshold,
    )

    test_predictions = np.where(
        test_probabilities[:, ai_index] >= selected_threshold,
        ai_index,
        human_index,
    )
    roc_auc = float(roc_auc_score(test_df["target"], test_probabilities[:, ai_index]))
    report = classification_report(
        test_df["target"],
        test_predictions,
        target_names=label_classes,
        digits=4,
        output_dict=True,
    )

    print(f"Selected AI threshold from validation: {selected_threshold:.2f}")
    print(f"Validation threshold metrics: {selected_val_metrics}")
    print(f"Test accuracy @ tuned threshold: {tuned_metrics['accuracy']:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")

    metrics = {
        "task_type": "binary_human_ai",
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
        "validation_threshold_search": threshold_search,
        "selected_ai_threshold": round(selected_threshold, 4),
        "selected_val_metrics": selected_val_metrics,
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
        "task_type": "binary_human_ai",
        "text_model_subdir": "bert_classifier",
        "max_length": args.max_length,
        "text_col": args.text_col,
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
