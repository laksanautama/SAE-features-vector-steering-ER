"""
Evaluation metrics for emotion recognition.
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def compute_metrics(y_true, y_pred, emotion_classes: list = None) -> dict:
    """
    Compute comprehensive metrics for binary or multi-label predictions.

    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        emotion_classes: list of emotion names (for per-emotion breakdown)

    Returns:
        dict with macro_f1, micro_f1, accuracy, per_emotion, etc.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    result = {
        "accuracy": accuracy_score(y_true.flatten(), y_pred.flatten()),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    # Multi-label metrics
    if y_true.ndim == 2:
        result["micro_f1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
        result["sample_f1"] = f1_score(y_true, y_pred, average="samples", zero_division=0)
        if emotion_classes:
            result["per_emotion"] = {}
            for i, emo in enumerate(emotion_classes):
                result["per_emotion"][emo] = f1_score(
                    y_true[:, i], y_pred[:, i], zero_division=0
                )

    return result


def threshold_sweep(scores, labels, thresholds=None):
    """
    Find optimal threshold for binary classification scores.

    Returns:
        (best_threshold, best_f1, sweep_df)
    """
    import pandas as pd

    if thresholds is None:
        thresholds = np.arange(-5.0, 5.0, 0.1)

    results = []
    for t in thresholds:
        preds = (scores > t).astype(float)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        n_yes = int(preds.sum())
        results.append({"threshold": round(t, 1), "f1": f1, "n_yes": n_yes})

    df = pd.DataFrame(results)
    best = df.loc[df["f1"].idxmax()]
    return float(best["threshold"]), float(best["f1"]), df
