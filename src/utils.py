"""
Plotting utilities and path helpers.
All plots save to disk (no interactive display on HPC).
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt
from pathlib import Path


def save_json(data: dict, path: Path):
    """Save dict as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ═══════════════════════════════════════════════════════════════
# Plotting functions — all save to disk
# ═══════════════════════════════════════════════════════════════

def plot_training_curves(history: dict, save_path: Path, title_prefix: str = ""):
    """Plot loss, nnz, and L1 curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="Train")
    if "eval_loss" in history:
        axes[0].plot(history["eval_loss"], label="Eval")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title_prefix}Loss")
    axes[0].legend()

    if "alpha_nnz" in history:
        axes[1].plot(history["alpha_nnz"], color="purple")
        axes[1].set_title(f"{title_prefix}Active Features")
        axes[1].set_xlabel("Epoch")

    if "alpha_l1" in history:
        axes[2].plot(history["alpha_l1"], color="green")
        axes[2].set_title(f"{title_prefix}L1 Norm")
        axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {save_path}")


def plot_threshold_sweep(sweep_dfs: dict, labels, save_path: Path):
    """
    Plot F1 vs threshold for multiple configurations.
    sweep_dfs: {name: DataFrame with 'threshold' and 'f1' columns}
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, df in sweep_dfs.items():
        lw = 2 if name in ["Unsteered", "All layers"] else 1
        ax.plot(df["threshold"], df["f1"], label=name, linewidth=lw)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Macro F1")
    ax.set_title("F1 vs Threshold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {save_path}")


def plot_score_distributions(scores_dict: dict, labels, save_path: Path):
    """
    Plot score distributions for multiple configurations.
    scores_dict: {name: numpy array of scores}
    """
    n = len(scores_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for i, (name, scores) in enumerate(scores_dict.items()):
        axes[i].hist(scores[labels == 0], bins=40, alpha=0.6, label="no", color="#E57373")
        axes[i].hist(scores[labels == 1], bins=40, alpha=0.6, label="yes", color="#64B5F6")
        axes[i].set_title(name)
        axes[i].legend(fontsize=8)

    plt.suptitle("Score Distributions")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {save_path}")


def plot_probing_results(probe_df, emotion_classes: list, save_path: Path):
    """Plot probing F1 bar chart and per-emotion lines."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Bar chart
    best_idx = probe_df["macro_f1"].idxmax()
    colors = ["#2ECC71" if i == best_idx else "#7F77DD" for i in range(len(probe_df))]
    axes[0].bar(probe_df["layer_name"], probe_df["macro_f1"], color=colors)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("Probe Macro F1 by Layer")
    axes[0].tick_params(axis="x", rotation=90, labelsize=6)

    # Per-emotion lines
    for emo in emotion_classes:
        col = f"f1_{emo}"
        if col in probe_df.columns:
            axes[1].plot(probe_df["layer_name"], probe_df[col],
                         label=emo, marker="o", markersize=2)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("F1")
    axes[1].set_title("Per-Emotion Probe F1")
    axes[1].tick_params(axis="x", rotation=90, labelsize=6)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {save_path}")


def plot_probing_heatmap(probe_df, emotion_classes: list, save_path: Path):
    """Plot layer x emotion heatmap."""
    emo_cols = [f"f1_{e}" for e in emotion_classes]
    data = probe_df[emo_cols].values.T
    layer_names = probe_df["layer_name"].values

    fig, ax = plt.subplots(figsize=(18, 4))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(emotion_classes)))
    ax.set_yticklabels(emotion_classes)
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=90, fontsize=6)
    ax.set_xlabel("Layer")
    ax.set_title("Probe F1: Layer x Emotion")
    plt.colorbar(im, label="F1")

    # Mark best per emotion
    for i in range(len(emotion_classes)):
        best_col = data[i].argmax()
        ax.plot(best_col, i, "k*", markersize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {save_path}")


def plot_layer_ablation(config_results: dict, baseline_f1: float,
                        emotion_classes: list, per_emotion_df, save_path: Path):
    """Plot ablation bar chart + per-emotion heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart
    names = list(config_results.keys())
    macros = [config_results[n]["macro_f1"] for n in names]
    colors = ["#999" if n == "Unsteered"
              else ("#2ECC71" if config_results[n]["macro_f1"] > baseline_f1 else "#E74C3C")
              for n in names]
    axes[0].bar(names, macros, color=colors)
    axes[0].axhline(baseline_f1, color="black", linestyle="--", alpha=0.5, label="baseline")
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("Macro F1 by Configuration")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend()

    # Per-emotion heatmap
    steered_names = [n for n in names if n != "Unsteered"]
    if steered_names and per_emotion_df is not None:
        base_row = per_emotion_df[per_emotion_df["Config"] == "Unsteered"].iloc[0]
        delta_matrix = np.zeros((len(steered_names), len(emotion_classes)))
        for i, name in enumerate(steered_names):
            row = per_emotion_df[per_emotion_df["Config"] == name]
            if len(row) > 0:
                for j, emo in enumerate(emotion_classes):
                    if emo in row.columns:
                        delta_matrix[i, j] = row.iloc[0][emo] - base_row[emo]

        im = axes[1].imshow(delta_matrix, aspect="auto", cmap="RdYlGn", vmin=-0.2, vmax=0.2)
        axes[1].set_xticks(range(len(emotion_classes)))
        axes[1].set_xticklabels([e[:5] for e in emotion_classes], rotation=45)
        axes[1].set_yticks(range(len(steered_names)))
        axes[1].set_yticklabels(steered_names)
        axes[1].set_title("Per-Emotion F1 Delta")
        plt.colorbar(im, ax=axes[1], label="Delta F1")

        for i in range(len(steered_names)):
            for j in range(len(emotion_classes)):
                axes[1].text(j, i, f"{delta_matrix[i, j]:+.3f}",
                            ha="center", va="center", fontsize=7,
                            color="white" if abs(delta_matrix[i, j]) > 0.1 else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {save_path}")


def plot_classifier_selection(all_weights, per_emo_weights, selected_set,
                              emotion_classes, layer, save_path: Path):
    """Plot classifier weight distribution for one layer."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of weights
    w = all_weights[all_weights > 0]
    if len(w) > 0:
        axes[0].hist(w, bins=60, color="#7F77DD", edgecolor="white", alpha=0.8)
    axes[0].set_title(f"L{layer}: classifier weight distribution")
    axes[0].set_xlabel("Max |weight| across emotions")

    # Per-emotion heatmap of top features
    sel = sorted(selected_set)[:50]
    if len(sel) > 0:
        hm = np.array([[per_emo_weights[emo][idx] for idx in sel]
                        for emo in emotion_classes])
        im = axes[1].imshow(hm, aspect="auto", cmap="YlOrRd")
        axes[1].set_yticks(range(len(emotion_classes)))
        axes[1].set_yticklabels(emotion_classes)
        axes[1].set_xlabel("Feature rank")
        axes[1].set_title(f"L{layer}: Per-emotion weights (top {len(sel)})")
        plt.colorbar(im, ax=axes[1], label="|weight|")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {save_path}")
