#!/usr/bin/env python3
"""
Layer Probing Experiment
========================
For each model, train binary probes at every layer to find where
emotion information is most linearly accessible.

Usage:
    python scripts/run_layer_probing.py --language indonesia
"""

import argparse
import gc
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (load_experiment_config, load_dataset_registry,
                         get_model_config, model_slug, resolve_output_dir,
                         hf_login)
from src.data import load_binary_pairs
from src.prompt import build_binary_prompt
from src.model import load_model_and_tokenizer, free_model, DEVICE
from src.utils import (save_json, ensure_dir, plot_probing_results,
                        plot_probing_heatmap)


def extract_hidden_states(df, tokenizer, model, batch_size, max_seq_len):
    """
    Extract last-token hidden states for binary (text, emotion_query, label) pairs.
    Returns: dict mapping layer_idx -> numpy array (n_samples, d_model), labels, emotions
    """
    prompts = [build_binary_prompt(r["text"], r["emotion_query"]) for _, r in df.iterrows()]
    labels = df["label"].values
    emotions = df["emotion_query"].values

    n_layers_total = model.config.num_hidden_layers + 1
    layer_states = {l: [] for l in range(n_layers_total)}

    n_batches = (len(prompts) + batch_size - 1) // batch_size
    print(f"  Extracting from {len(prompts)} prompts ({n_batches} batches) ...")

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_seq_len).to(DEVICE)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                output_hidden_states=True,
            )

        seq_lengths = enc.attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(enc.input_ids.size(0), device=DEVICE)

        for l, hs in enumerate(outputs.hidden_states):
            last_tok = hs[batch_idx, seq_lengths].float().cpu().numpy()
            layer_states[l].append(last_tok)

        # Free GPU memory each batch
        del outputs, enc
        torch.cuda.empty_cache()

        if (i // batch_size + 1) % 20 == 0:
            print(f"    Batch {i // batch_size + 1}/{n_batches}")

    for l in layer_states:
        layer_states[l] = np.concatenate(layer_states[l], axis=0)

    print(f"  Done. Shape per layer: {layer_states[0].shape}")
    return layer_states, labels, emotions


def run_probing(model_id, language, data_config, exp_config):
    ds_reg = load_dataset_registry()
    probe_cfg = ds_reg.get("probing", {})

    out_dir = resolve_output_dir(
        exp_config["output_dir"], model_id, "probing", language
    )
    ensure_dir(out_dir)
    plots_dir = ensure_dir(out_dir / "plots")

    emotion_classes = data_config["emotion_classes"]

    print(f"\n{'='*60}")
    print(f"  PROBING: {model_id} x {language}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_id)
    n_layers = model.config.num_hidden_layers

    # Load data as binary pairs
    train_df, _ = load_binary_pairs(
        data_config,
        max_samples=probe_cfg.get("max_train_samples", 300),
        split_name="train",
    )
    eval_df, _ = load_binary_pairs(
        data_config,
        max_samples=probe_cfg.get("max_eval_samples", 100),
    )
    print(f"  Train: {len(train_df)} pairs, Eval: {len(eval_df)} pairs")

    # Extract hidden states
    bs = probe_cfg.get("batch_size", 8)
    max_seq = probe_cfg.get("max_seq_len", 128)

    print("Extracting TRAIN hidden states:")
    train_states, train_labels, train_emotions = extract_hidden_states(
        train_df, tokenizer, model, bs, max_seq)

    print("\nExtracting EVAL hidden states:")
    eval_states, eval_labels, eval_emotions = extract_hidden_states(
        eval_df, tokenizer, model, bs, max_seq)

    # ── Free model + tokenizer immediately after extraction ──
    free_model(model, tokenizer)
    del model, tokenizer

    # === Overall binary probe at each layer ===
    print("\n== Training binary probes at each layer ==\n")
    probe_results = []

    for layer_idx in range(n_layers + 1):
        X_train = train_states[layer_idx]
        X_eval = eval_states[layer_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_eval_s = scaler.transform(X_eval)

        probe = LogisticRegression(
            max_iter=500, C=1.0, solver="lbfgs", random_state=42
        )
        probe.fit(X_train_s, train_labels)
        y_pred = probe.predict(X_eval_s)

        acc = np.mean(eval_labels == y_pred)
        macro_f1 = f1_score(eval_labels, y_pred, average="macro", zero_division=0)
        f1_yes = f1_score(eval_labels, y_pred, pos_label=1, zero_division=0)
        f1_no = f1_score(eval_labels, y_pred, pos_label=0, zero_division=0)

        layer_name = "emb" if layer_idx == 0 else f"L{layer_idx - 1}"

        probe_results.append({
            "layer_idx": layer_idx,
            "layer_name": layer_name,
            "actual_layer": layer_idx - 1,
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "f1_yes": float(f1_yes),
            "f1_no": float(f1_no),
        })

        print(f"  {layer_name:>4}  acc={acc:.4f}  macro_f1={macro_f1:.4f}  "
              f"f1(yes)={f1_yes:.4f}  f1(no)={f1_no:.4f}")

    probe_df = pd.DataFrame(probe_results)

    # Best overall
    best = probe_df.loc[probe_df["macro_f1"].idxmax()]
    print(f"\nBest layer: {best['layer_name']} (F1={best['macro_f1']:.4f})")

    # === Per-emotion binary probing (full layer × emotion matrix) ===
    print("\n== Per-Emotion Probing ==\n")
    per_emo_results = []
    layer_emo_map = defaultdict(list)

    # Collect F1 for every (layer, emotion) pair — this feeds the heatmap
    emo_layer_f1 = {emo: np.zeros(n_layers + 1) for emo in emotion_classes}

    for emo in emotion_classes:
        train_mask = (train_emotions == emo)
        eval_mask = (eval_emotions == emo)

        if train_mask.sum() == 0 or eval_mask.sum() == 0:
            print(f"  {emo:<10} skipped (no data)")
            continue

        y_train_e = train_labels[train_mask]
        y_eval_e = eval_labels[eval_mask]

        if len(np.unique(y_train_e)) < 2 or len(np.unique(y_eval_e)) < 2:
            print(f"  {emo:<10} skipped (single class)")
            continue

        best_f1 = 0
        best_layer = 0

        for layer_idx in range(n_layers + 1):
            X_tr = train_states[layer_idx][train_mask]
            X_ev = eval_states[layer_idx][eval_mask]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_ev_s = scaler.transform(X_ev)

            probe = LogisticRegression(
                max_iter=500, C=1.0, solver="lbfgs", random_state=42)
            probe.fit(X_tr_s, y_train_e)
            y_pred = probe.predict(X_ev_s)
            f1 = f1_score(y_eval_e, y_pred, average="macro", zero_division=0)

            emo_layer_f1[emo][layer_idx] = f1

            if f1 > best_f1:
                best_f1 = f1
                best_layer = layer_idx

        layer_name = "emb" if best_layer == 0 else f"L{best_layer - 1}"
        actual_layer = best_layer - 1 if best_layer > 0 else 0

        per_emo_results.append({
            "Emotion": emo,
            "Best layer": layer_name,
            "Actual layer": actual_layer,
            "Best F1": best_f1,
        })
        layer_emo_map[actual_layer].append(emo)
        print(f"  {emo:<10} best={layer_name:<5} F1={best_f1:.4f}")

    # Populate probe_df with the full per-emotion F1 columns for plots
    for emo in emotion_classes:
        probe_df[f"f1_{emo}"] = emo_layer_f1[emo]

    # ── Free hidden states (can be several GB) ──
    del train_states, eval_states, train_labels, eval_labels
    del train_emotions, eval_emotions, train_df, eval_df
    gc.collect()

    # Save results
    probe_df.to_csv(out_dir / "probe_results.csv", index=False)
    per_emo_df = pd.DataFrame(per_emo_results)
    per_emo_df.to_csv(out_dir / "per_emotion_probe.csv", index=False)

    # Save full layer × emotion F1 matrix as its own CSV for easy inspection
    emo_cols = ["layer_name"] + [f"f1_{e}" for e in emotion_classes]
    probe_df[emo_cols].to_csv(out_dir / "layer_emotion_f1_matrix.csv", index=False)

    summary = {
        "model_id": model_id,
        "language": language,
        "n_layers": n_layers,
        "overall_best_layer": int(best["actual_layer"]),
        "overall_best_f1": float(best["macro_f1"]),
        "per_emotion_best": per_emo_results,
        "layer_emotion_map": {int(k): v for k, v in layer_emo_map.items()},
        "emotion_classes": emotion_classes,
    }
    save_json(summary, out_dir / "probing_summary.json")

    # Plots
    plot_probing_results(probe_df, emotion_classes, plots_dir / "probing_f1.png")
    plot_probing_heatmap(probe_df, emotion_classes, plots_dir / "probing_heatmap.png")

    print(f"\n  Results saved to {out_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run layer probing experiments")
    parser.add_argument("--language", required=True, help="Language key (e.g., indonesia)")
    args = parser.parse_args()

    exp_config = load_experiment_config()
    ds_reg = load_dataset_registry()
    data_config = ds_reg["languages"][args.language]

    hf_login()

    for model_id in exp_config["models"]:
        try:
            run_probing(model_id, args.language, data_config, exp_config)
        except Exception as e:
            print(f"\n  ERROR for {model_id}: {e}")
            import traceback
            traceback.print_exc()
            # Ensure cleanup even on error
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    print("\n\nAll probing experiments complete.")


if __name__ == "__main__":
    main()