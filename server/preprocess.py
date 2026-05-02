from __future__ import annotations

import re
import string
from typing import Iterable

import pandas as pd


STAT_FEATURES = [
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

LEGACY_MODEL_TEXT_TRANSFORM = "transformed_text_or_raw_to_special_tokens"
RAW_MODEL_TEXT_TRANSFORM = "raw_to_special_tokens"
X_MODEL_TEXT_TRANSFORM = "x_text_special_tokens_v1"
LEGACY_MODEL_SPECIAL_TOKENS = [
    "<SPACE>",
    "<ENTER>",
    "<REP>",
    "</REP>",
]
X_MODEL_SPECIAL_TOKENS = [
    "<URL>",
    "<MENTION>",
    "<HASHTAG>",
    "<ENTER>",
    "<REP>",
    "</REP>",
    "<EMOJI>",
]
ALL_PRETRANSFORM_MARKERS = tuple(
    dict.fromkeys(LEGACY_MODEL_SPECIAL_TOKENS + X_MODEL_SPECIAL_TOKENS)
)

LEGACY_STAT_FEATURES = [
    "rep_span_count",
    "rep_total_count",
]

PUNCTUATION_SET = set(string.punctuation) | {
    "\u2026",
    "\u201c",
    "\u201d",
    "\u2018",
    "\u2019",
    "\u00b7",
    "\u3002",
    "\uff01",
    "\uff1f",
}
TRAILING_PUNCT_PATTERN = re.compile(r"[.!?~\s]+$")
LAUGHTER_PATTERN = re.compile(r"(\u314b{2,}|\u314e{2,}|\u3160{2,}|\u315c{2,})")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1+")
REP_TOKEN_CHARS = {"?", "!", "\u314b", "\u314e", "\u3160", "\u315c", "~"}
EMOJI_RANGES = (
    (0x1F300, 0x1FAFF),
    (0x2600, 0x26FF),
    (0x2700, 0x27BF),
)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#([가-힣A-Za-z0-9_]+)")
THREE_PLUS_REPEAT_PATTERN = re.compile(r"(.)\1{2,}")


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()


def get_model_special_tokens(transform_name: str | None = None) -> list[str]:
    normalized = (transform_name or LEGACY_MODEL_TEXT_TRANSFORM).strip().lower()
    if normalized == X_MODEL_TEXT_TRANSFORM:
        return X_MODEL_SPECIAL_TOKENS.copy()
    return LEGACY_MODEL_SPECIAL_TOKENS.copy()


def add_model_special_tokens(tokenizer, model=None, transform_name: str | None = None) -> int:
    tokens = get_model_special_tokens(transform_name)
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    if model is not None and num_added:
        model.resize_token_embeddings(len(tokenizer))
    return int(num_added)


def prepare_model_text(
    text: str | None,
    transform_name: str | None = None,
) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    if any(marker in normalized for marker in ALL_PRETRANSFORM_MARKERS):
        return normalized
    transform_name = (transform_name or LEGACY_MODEL_TEXT_TRANSFORM).strip().lower()
    if transform_name == X_MODEL_TEXT_TRANSFORM:
        return preprocess_x_text(normalized)
    return transform_text_for_model(normalized)


def _strip_trailing_punctuation(text: str) -> str:
    return TRAILING_PUNCT_PATTERN.sub("", text)


def _count_repeated_char_runs(text: str) -> tuple[int, int]:
    compact = re.sub(r"\s+", "", text)
    spans = list(REPEATED_CHAR_PATTERN.finditer(compact))
    rep_span_count = len(spans)
    rep_total_count = sum(len(match.group(0)) - 1 for match in spans)
    return rep_span_count, rep_total_count


def _last_visible_char(text: str) -> str:
    stripped = text.rstrip()
    return stripped[-1] if stripped else ""


def _is_hangul_char(char: str) -> bool:
    if not char:
        return False
    codepoint = ord(char)
    return (
        0xAC00 <= codepoint <= 0xD7A3
        or 0x1100 <= codepoint <= 0x11FF
        or 0x3130 <= codepoint <= 0x318F
    )


def _is_emoji_char(char: str) -> bool:
    if not char:
        return False
    codepoint = ord(char)
    return any(start <= codepoint <= end for start, end in EMOJI_RANGES)


def _replace_emoji_with_token(text: str) -> str:
    output: list[str] = []
    for char in text:
        if _is_emoji_char(char):
            output.append(" <EMOJI> ")
        else:
            output.append(char)
    return "".join(output)


def preprocess_x_text(text: str | None) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""

    transformed = URL_PATTERN.sub(" <URL> ", normalized)
    transformed = MENTION_PATTERN.sub(" <MENTION> ", transformed)
    transformed = HASHTAG_PATTERN.sub(r" <HASHTAG> \1 ", transformed)
    transformed = _replace_emoji_with_token(transformed)
    transformed = transformed.replace("\n", " <ENTER> ")

    def replace_rep(match: re.Match[str]) -> str:
        char = match.group(1)
        count = len(match.group(0))
        return f" <REP> {char} {count} </REP> "

    transformed = THREE_PLUS_REPEAT_PATTERN.sub(replace_rep, transformed)
    transformed = re.sub(r"\s+", " ", transformed).strip()
    return transformed


def transform_text_for_model(text: str | None) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""

    output: list[str] = []
    index = 0

    while index < len(normalized):
        char = normalized[index]

        if char == "\n":
            output.append("<ENTER>")
            index += 1
            continue

        if char.isspace():
            output.append("<SPACE>")
            index += 1
            continue

        if char in REP_TOKEN_CHARS:
            run_end = index + 1
            while run_end < len(normalized) and normalized[run_end] == char:
                run_end += 1

            run_length = run_end - index
            if run_length >= 2:
                output.append(f"<REP> {char} {run_length} </REP>")
                index = run_end
                continue

        output.append(char)
        index += 1

    return "".join(output)


def extract_stat_features_from_text(text: str | None) -> dict[str, float]:
    normalized = normalize_text(text)
    compact_text = re.sub(r"\s+", "", normalized)
    rep_span_count, rep_total_count = _count_repeated_char_runs(normalized)
    stripped_for_ending = _strip_trailing_punctuation(normalized)
    last_visible_char = _last_visible_char(stripped_for_ending)
    unique_punctuation = {char for char in normalized if char in PUNCTUATION_SET}
    punctuation_total_count = sum(1 for char in normalized if char in PUNCTUATION_SET)

    features = {
        "rep_ratio": float(rep_total_count / len(compact_text)) if compact_text else 0.0,
        "has_laughter_rep": float(bool(LAUGHTER_PATTERN.search(normalized))),
        "punc_total_count": float(punctuation_total_count),
        "punc_unique_count": float(len(unique_punctuation)),
        "endswith_period": float(normalized.endswith(".")),
        "endswith_yo": float(stripped_for_ending.endswith("\uC694")),
        "last_char_is_hangul": float(_is_hangul_char(last_visible_char)),
        "endswith_da": float(stripped_for_ending.endswith("\uB2E4")),
        "has_enter": float("\n" in normalized),
        "enter_cnt": float(normalized.count("\n")),
        "space_cnt": float(normalized.count(" ")),
        "rep_span_count": float(rep_span_count),
        "rep_total_count": float(rep_total_count),
    }
    return features


def build_stats_frame(
    texts: Iterable[str | None],
    feature_order: list[str] | None = None,
) -> pd.DataFrame:
    feature_order = feature_order or STAT_FEATURES
    rows = []
    for text in texts:
        extracted = extract_stat_features_from_text(text)
        rows.append({feature: extracted[feature] for feature in feature_order})
    return pd.DataFrame(rows, columns=feature_order)
