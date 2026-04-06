from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
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

from server.preprocess import prepare_model_text


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
    parser = argparse.ArgumentParser(description="Train stage-1 KC-BERT binary human/ai classifier.")
    parser.add_argument("--data-path", required=True, help="CSV path.")
    parser.add_argument("--output-dir", required=True, help="Directory to save trained artifacts.")
    parser.add_argument("--text-col", default="transformed_text", help="Text column name.")
    parser.add_argument("--label-col", default="model_source", help="Label column name.")
    parser.add_argument("--split-col", default="split", help="Split column name.")
    parser.add_argument("--model-name", default="beomi/kcbert-base", help="HF model name.")
    parser.add_argument("--max-length", type=int, default=128, help="Max token length.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Eval batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio.")
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_path, encoding="utf-8-sig")
    required_columns = {args.text_col, args.label_col}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = df.copy()
    df[args.text_col] = df[args.text_col].map(prepare_model_text)
    df[args.label_col] = df[args.label_col].astype(str).str.lower().str.strip()
    df = df[df[args.text_col].ne("")]
    df["binary_label"] = np.where(df[args.label_col].eq("human"), "human", "ai")

    train_df, val_df, test_df = prepare_splits(
        df,
        split_col=args.split_col,
        label_col="binary_label",
        random_state=args.random_state,
    )

    label_classes = ["human", "ai"]
    label_to_id = {label: index for index, label in enumerate(label_classes)}

    train_df["target"] = train_df["binary_label"].map(label_to_id)
    val_df["target"] = val_df["binary_label"].map(label_to_id)
    test_df["target"] = test_df["binary_label"].map(label_to_id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
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
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

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

    print("Training stage-1 KC-BERT binary classifier...")
    trainer.train()
    trainer.save_model(str(bert_output_dir))
    tokenizer.save_pretrained(str(bert_output_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probabilities = predict_probabilities(
        trainer.model,
        tokenizer,
        test_df[args.text_col].tolist(),
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        device=device,
    )
    predictions = np.argmax(probabilities, axis=1)

    ai_index = label_to_id["ai"]
    accuracy = float(accuracy_score(test_df["target"], predictions))
    roc_auc = float(roc_auc_score(test_df["target"], probabilities[:, ai_index]))
    report = classification_report(
        test_df["target"],
        predictions,
        target_names=label_classes,
        digits=4,
        output_dict=True,
    )

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")

    metrics = {
        "task_type": "binary_human_ai",
        "test_accuracy": round(accuracy, 4),
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
        "label_classes": label_classes,
        "positive_label": "ai",
        "negative_label": "human",
        "model_text_transform": "transformed_text_or_raw_to_special_tokens",
    }
    with (output_dir / "artifacts.json").open("w", encoding="utf-8") as handle:
        json.dump(artifacts, handle, ensure_ascii=False, indent=2)

    label_map = {str(index): label for index, label in enumerate(label_classes)}
    with (output_dir / "label_map.json").open("w", encoding="utf-8") as handle:
        json.dump(label_map, handle, ensure_ascii=False, indent=2)

    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
