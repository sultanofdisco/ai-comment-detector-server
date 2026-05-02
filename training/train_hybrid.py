from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.preprocess import (
    RAW_MODEL_TEXT_TRANSFORM,
    STAT_FEATURES,
    X_MODEL_TEXT_TRANSFORM,
    add_model_special_tokens,
    build_stats_frame,
    normalize_text,
    prepare_model_text,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stage-2 AI-only hybrid LLM classifier.")
    parser.add_argument("--data-path", required=True, help="CSV path.")
    parser.add_argument("--output-dir", required=True, help="Directory to save trained artifacts.")
    parser.add_argument("--text-col", default="reply_text", help="Text column name.")
    parser.add_argument("--label-col", default="model_source", help="Label column name.")
    parser.add_argument("--split-col", default="split", help="Split column name.")
    parser.add_argument(
        "--stats-cols",
        default="",
        help="Optional comma-separated stats feature columns to use.",
    )
    parser.add_argument(
        "--allowed-labels",
        default="gpt,claude,gemini,deepseek",
        help="Optional comma-separated labels to keep, e.g. deepseek,gemini,gpt,claude",
    )
    parser.add_argument("--model-name", default="beomi/kcbert-base", help="HF model name.")
    parser.add_argument(
        "--model-text-transform",
        default=RAW_MODEL_TEXT_TRANSFORM,
        choices=[RAW_MODEL_TEXT_TRANSFORM, X_MODEL_TEXT_TRANSFORM],
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
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
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
            train_df = df.loc[train_mask].copy()
            val_df = df.loc[val_mask].copy()
            test_df = df.loc[test_mask].copy()
            return train_df, val_df, test_df

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


def predict_text_probabilities(
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


def build_xgb_model(num_labels: int, random_state: int) -> XGBClassifier:
    common_args: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "eval_metric": "logloss" if num_labels == 2 else "mlogloss",
        "random_state": random_state,
        "n_jobs": -1,
    }

    if num_labels == 2:
        return XGBClassifier(objective="binary:logistic", **common_args)

    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_labels,
        **common_args,
    )


def safe_roc_auc(y_true: np.ndarray, probabilities: np.ndarray, num_labels: int) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None

    try:
        if num_labels == 2:
            return float(roc_auc_score(y_true, probabilities[:, 1]))
        return float(roc_auc_score(y_true, probabilities, multi_class="ovr"))
    except ValueError:
        return None


def choose_best_weight(
    text_prob: np.ndarray,
    stats_prob: np.ndarray,
    y_true: np.ndarray,
    weight_grid: list[float],
) -> tuple[float, list[dict[str, float]]]:
    results: list[dict[str, float]] = []
    best_weight = weight_grid[0]
    best_accuracy = -1.0

    for text_weight in weight_grid:
        stats_weight = 1.0 - text_weight
        hybrid_prob = (text_weight * text_prob) + (stats_weight * stats_prob)
        hybrid_pred = np.argmax(hybrid_prob, axis=1)
        accuracy = float(accuracy_score(y_true, hybrid_pred))
        results.append(
            {
                "text_weight": round(text_weight, 4),
                "stats_weight": round(stats_weight, 4),
                "accuracy": round(accuracy, 4),
            }
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weight = text_weight

    return best_weight, results


def main() -> None:
    args = parse_args()
    raw_text_col = "__raw_text"
    model_text_col = "__model_text"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_path, encoding="utf-8-sig")

    required_columns = {args.text_col, args.label_col}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = df.copy()
    df[raw_text_col] = df[args.text_col].map(normalize_text)
    df[model_text_col] = df[raw_text_col].map(
        lambda text: prepare_model_text(text, args.model_text_transform)
    )
    df[args.label_col] = df[args.label_col].astype(str).str.lower().str.strip()
    df = df[df[model_text_col].ne("")]

    if args.allowed_labels:
        allowed_labels = {
            label.strip().lower()
            for label in args.allowed_labels.split(",")
            if label.strip()
        }
        if "human" in allowed_labels:
            raise ValueError("Stage-2 hybrid training must not include the 'human' label.")
        df = df[df[args.label_col].isin(allowed_labels)].copy()

    if df.empty:
        raise ValueError("Dataset is empty after filtering.")

    if args.stats_cols:
        feature_order = [feature.strip() for feature in args.stats_cols.split(",") if feature.strip()]
    else:
        feature_order = STAT_FEATURES.copy()

    extracted_stats = build_stats_frame(df[raw_text_col].tolist(), feature_order)
    for feature in feature_order:
        df[feature] = extracted_stats[feature].astype(float)

    train_df, val_df, test_df = prepare_splits(
        df,
        split_col=args.split_col,
        label_col=args.label_col,
        random_state=args.random_state,
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(df[args.label_col])

    if len(label_encoder.classes_) < 2:
        raise ValueError("At least two labels are required to train a classifier.")

    train_df["target"] = label_encoder.transform(train_df[args.label_col])
    val_df["target"] = label_encoder.transform(val_df[args.label_col])
    test_df["target"] = label_encoder.transform(test_df[args.label_col])

    missing_train_labels = set(label_encoder.classes_) - set(train_df[args.label_col].unique())
    if missing_train_labels:
        raise ValueError(
            "Training split is missing labels: "
            + ", ".join(sorted(missing_train_labels))
        )

    num_labels = len(label_encoder.classes_)
    task_type = "llm_multiclass"

    print(f"Task type: {task_type}")
    print(f"Label map: {dict(enumerate(label_encoder.classes_.tolist()))}")
    print(f"Stats features: {feature_order}")
    print(
        f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )

    xgb_model = build_xgb_model(num_labels=num_labels, random_state=args.random_state)
    xgb_model.fit(train_df[feature_order], train_df["target"])
    stats_prob_val = xgb_model.predict_proba(val_df[feature_order])
    stats_prob_test = xgb_model.predict_proba(test_df[feature_order])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
    )
    num_added_special_tokens = add_model_special_tokens(
        tokenizer,
        bert_model,
        args.model_text_transform,
    )
    train_encodings = tokenizer(
        train_df[model_text_col].tolist(),
        truncation=True,
        max_length=args.max_length,
    )
    val_encodings = tokenizer(
        val_df[model_text_col].tolist(),
        truncation=True,
        max_length=args.max_length,
    )
    test_encodings = tokenizer(
        test_df[model_text_col].tolist(),
        truncation=True,
        max_length=args.max_length,
    )

    train_dataset = TextDataset(train_encodings, train_df["target"].tolist())
    val_dataset = TextDataset(val_encodings, val_df["target"].tolist())
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

    trainer = Trainer(
        model=bert_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Training KcBERT...")
    print(f"Model text transform: {args.model_text_transform}")
    print(f"Added special tokens: {num_added_special_tokens}")
    trainer.train()
    trainer.save_model(str(bert_output_dir))
    tokenizer.save_pretrained(str(bert_output_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = trainer.model

    bert_prob_val = predict_text_probabilities(
        trained_model,
        tokenizer,
        val_df[model_text_col].tolist(),
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        device=device,
    )
    bert_prob_test = predict_text_probabilities(
        trained_model,
        tokenizer,
        test_df[model_text_col].tolist(),
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        device=device,
    )

    weight_grid = [float(value.strip()) for value in args.weight_grid.split(",") if value.strip()]
    best_text_weight, validation_results = choose_best_weight(
        bert_prob_val,
        stats_prob_val,
        val_df["target"].to_numpy(),
        weight_grid,
    )
    best_stats_weight = 1.0 - best_text_weight

    final_prob_test = (best_text_weight * bert_prob_test) + (best_stats_weight * stats_prob_test)
    final_pred_test = np.argmax(final_prob_test, axis=1)

    test_accuracy = float(accuracy_score(test_df["target"], final_pred_test))
    test_roc_auc = safe_roc_auc(test_df["target"].to_numpy(), final_prob_test, num_labels)
    report = classification_report(
        test_df["target"],
        final_pred_test,
        target_names=label_encoder.classes_,
        digits=4,
        output_dict=True,
    )

    print(f"Best text weight: {best_text_weight:.2f}")
    print(f"Best stats weight: {best_stats_weight:.2f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    if test_roc_auc is not None:
        print(f"Test ROC-AUC: {test_roc_auc:.4f}")

    joblib.dump(xgb_model, output_dir / "xgb_model.joblib")

    label_map = {str(index): label for index, label in enumerate(label_encoder.classes_.tolist())}
    with (output_dir / "label_map.json").open("w", encoding="utf-8") as handle:
        json.dump(label_map, handle, ensure_ascii=False, indent=2)

    metrics = {
        "task_type": task_type,
        "validation_weight_search": validation_results,
        "selected_text_weight": round(best_text_weight, 4),
        "selected_stats_weight": round(best_stats_weight, 4),
        "test_accuracy": round(test_accuracy, 4),
        "test_roc_auc": None if test_roc_auc is None else round(test_roc_auc, 4),
        "classification_report": report,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    artifacts = {
        "model_version": output_dir.name,
        "task_type": task_type,
        "text_model_name": args.model_name,
        "text_model_subdir": "bert_classifier",
        "xgb_model_filename": "xgb_model.joblib",
        "text_weight": round(best_text_weight, 4),
        "stats_weight": round(best_stats_weight, 4),
        "stats_features": feature_order,
        "max_length": args.max_length,
        "text_col": args.text_col,
        "model_text_transform": args.model_text_transform,
        "label_col": args.label_col,
        "label_classes": label_encoder.classes_.tolist(),
    }
    with (output_dir / "artifacts.json").open("w", encoding="utf-8") as handle:
        json.dump(artifacts, handle, ensure_ascii=False, indent=2)

    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
