"""
Dataset loading utilities for emotion recognition datasets.
"""

import pandas as pd
from datasets import load_dataset


def _resolve_splits(ds):
    """Resolve available splits, handling validation/dev naming."""
    if "validation" in ds and "dev" not in ds:
        ds["dev"] = ds["validation"]
    available = set(ds.keys())
    if "train" in available and "test" in available:
        return "train", "test"
    elif "train" in available and "dev" in available:
        return "train", "dev"
    elif "dev" in available and "test" in available:
        return "dev", "test"
    elif "train" in available:
        return "train", "train"
    else:
        first = list(available)[0]
        return first, first


def load_emotion_dataset(data_config: dict, max_train: int = None,
                         max_eval: int = None) -> tuple:
    """
    Load emotion dataset in multi-label format.

    Returns:
        (train_df, eval_df) — each has text + emotion columns + 'target' column
    """
    lk = {"path": data_config["hf_dataset_id"]}
    if data_config.get("hf_subset"):
        lk["name"] = data_config["hf_subset"]
    ds = load_dataset(**lk)

    emo_cols = data_config["emotion_classes"]
    text_col = data_config["text_column"]
    non_neutral = [e for e in emo_cols if e != "neutral"]

    def to_df(split_name, max_samples):
        split = ds[split_name]
        if max_samples:
            split = split.select(range(min(max_samples, len(split))))
        df = split.to_pandas()
        for e in emo_cols:
            if e not in df.columns:
                df[e] = 0
            df[e] = df[e].astype(int)
        if "neutral" in emo_cols:
            df["neutral"] = (df[non_neutral].sum(axis=1) == 0).astype(int)
        df["target"] = df.apply(
            lambda r: ", ".join(sorted([e for e in emo_cols if r[e] == 1])) or "none",
            axis=1,
        )
        df = df[[text_col] + emo_cols + ["target"]]
        return df

    train_sp, eval_sp = _resolve_splits(ds)
    train_df = to_df(train_sp, max_train)
    eval_df = to_df(eval_sp, max_eval)
    return train_df, eval_df


def load_binary_pairs(data_config: dict, max_samples: int = None,
                      split_name: str = None) -> tuple:
    """
    Load dataset exploded into binary (text, emotion_query, label) pairs.
    Used by SP-2 binary mode and analysis.

    Args:
        data_config: dataset config dict
        max_samples: max number of *texts* (not pairs) to load
        split_name: explicit split to use ('train' or 'test'/'dev').
                    If None, uses eval split for backward compat.
    """
    lk = {"path": data_config["hf_dataset_id"]}
    if data_config.get("hf_subset"):
        lk["name"] = data_config["hf_subset"]
    ds = load_dataset(**lk)

    emo_cols = data_config["emotion_classes"]
    text_col = data_config["text_column"]
    non_neutral = [e for e in emo_cols if e != "neutral"]

    train_sp, eval_sp = _resolve_splits(ds)
    if split_name == "train":
        use_split = train_sp
    elif split_name is not None:
        use_split = split_name
    else:
        use_split = eval_sp

    split = ds[use_split]
    if max_samples:
        split = split.select(range(min(max_samples, len(split))))

    df = split.to_pandas()
    for e in emo_cols:
        if e not in df.columns:
            df[e] = 0
        df[e] = df[e].astype(int)
    if "neutral" in emo_cols:
        df["neutral"] = (df[non_neutral].sum(axis=1) == 0).astype(int)

    rows = []
    for _, row in df.iterrows():
        for emo in emo_cols:
            rows.append({
                "text": row[text_col],
                "emotion_query": emo,
                "label": int(row[emo]),
            })
    return pd.DataFrame(rows), emo_cols


def load_emotional_texts(data_config: dict, max_texts: int = 500) -> list:
    """Load texts that have at least one emotion label."""
    lk = {"path": data_config["hf_dataset_id"]}
    if data_config.get("hf_subset"):
        lk["name"] = data_config["hf_subset"]
    ds = load_dataset(**lk)

    non_neutral = [e for e in data_config["emotion_classes"] if e != "neutral"]
    train_sp, _ = _resolve_splits(ds)
    texts = []
    for example in ds[train_sp]:
        if len(texts) >= max_texts:
            break
        t = example[data_config["text_column"]]
        has_emotion = any(example.get(e, 0) == 1 for e in non_neutral)
        if has_emotion and t and len(t.strip()) > 20:
            texts.append(t.strip())
    return texts


def load_neutral_texts(neutral_config: dict, max_texts: int = 500) -> list:
    """Load neutral/general texts from CulturaX or similar."""
    lk = {"path": neutral_config["path"]}
    if neutral_config.get("name"):
        lk["name"] = neutral_config["name"]
    if neutral_config.get("split"):
        lk["split"] = neutral_config["split"]
    if neutral_config.get("streaming"):
        lk["streaming"] = True

    ds = load_dataset(**lk)
    text_col = neutral_config.get("text_col", "text")
    texts = []
    for example in ds:
        if len(texts) >= max_texts:
            break
        t = example[text_col]
        if t and len(t.strip()) > 20:
            texts.append(t.strip())
    return texts
