#!/usr/bin/env python3
"""
SP-1 Classifier-Weight Feature Selection
==========================================
Selects SAE features by training a multi-label L1-regularised classifier
on pooled SAE activations. Features with non-zero classifier weights
are selected as candidates for SP-2.

Usage:
    python scripts/run_sp1_classifier.py --language indonesia
"""

import argparse
import gc
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (load_experiment_config, load_dataset_registry, get_model_config,
                         get_sae_id, get_width_config, model_slug, resolve_output_dir,
                         resolve_layers, hf_login)
from src.model import load_model_and_tokenizer, free_model, DEVICE, clear_hf_cache
from src.sae import load_sae_encoder
from src.utils import save_json, ensure_dir


def load_texts_with_labels(data_config, max_texts=500):
    """
    Load texts with multi-hot emotion labels from BRIGHTER dataset.

    Returns:
        (texts, labels) — texts: list[str], labels: np.array (n_texts, n_emotions)
    """
    from datasets import load_dataset

    lk = {"path": data_config["hf_dataset_id"]}
    if data_config.get("hf_subset"):
        lk["name"] = data_config["hf_subset"]
    ds = load_dataset(**lk)

    emotion_classes = data_config["emotion_classes"]
    emotion_cols = [e for e in emotion_classes if e != "neutral"]
    text_col = data_config["text_column"]

    # Resolve split
    if "validation" in ds and "dev" not in ds:
        ds["dev"] = ds["validation"]
    available = set(ds.keys())
    if "train" in available:
        use_split = "train"
    elif "dev" in available:
        use_split = "dev"
    elif "test" in available:
        use_split = "test"
    else:
        use_split = list(available)[0]
    print(f"  Using split: {use_split} (available: {available})")

    all_texts = []
    all_labels = []

    for example in ds[use_split]:
        if len(all_texts) >= max_texts:
            break
        t = example[text_col]
        if not t or len(t.strip()) <= 20:
            continue

        label_row = []
        for emo in emotion_classes:
            if emo == "neutral":
                has_any = any(example.get(e, 0) == 1 for e in emotion_cols)
                label_row.append(0 if has_any else 1)
            else:
                label_row.append(int(example.get(emo, 0)))
        all_texts.append(t.strip())
        all_labels.append(label_row)

    all_labels = np.array(all_labels)
    print(f"  Loaded {len(all_texts)} texts with multi-hot labels")
    print(f"  Label shape: {all_labels.shape}")
    print(f"  Per-emotion counts:")
    for i, emo in enumerate(emotion_classes):
        print(f"    {emo:<10} {all_labels[:, i].sum():>4} / {len(all_texts)}")

    return all_texts, all_labels


def extract_pooled_activations(texts, model, tokenizer, sae_encoder, sae_b_dec,
                                layer_j, config):
    """Extract max-pooled SAE activations for each text."""
    pool_method = config.get("pool_method", "max")
    batch_size = config.get("batch_size", 4)
    max_seq = config.get("max_seq_len", 128)

    all_pooled = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_seq).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids=enc.input_ids,
                          attention_mask=enc.attention_mask,
                          output_hidden_states=True)

        hs = outputs.hidden_states[layer_j + 1]
        with torch.no_grad():
            h_centered = hs.float() - sae_b_dec.float()
            z = F.relu(sae_encoder(h_centered))

        mask = enc.attention_mask.unsqueeze(-1).float()
        z_masked = z * mask

        if pool_method == "max":
            z_for_max = z_masked.clone()
            z_for_max[mask.squeeze(-1) == 0] = -float("inf")
            pooled, _ = z_for_max.max(dim=1)
            pooled = pooled.clamp(min=0)
        else:
            token_counts = mask.sum(dim=1).clamp(min=1)
            pooled = z_masked.sum(dim=1) / token_counts

        all_pooled.append(pooled.cpu())
        del outputs, hs, z, z_masked, pooled
        torch.cuda.empty_cache()

    all_pooled = torch.cat(all_pooled, dim=0)
    return all_pooled.numpy()


def run_classifier_sp1(model_id, width, language, data_config, exp_config, mode):
    ds_reg = load_dataset_registry()
    cls_cfg = ds_reg.get("classifier_selection", {})

    base_dir = resolve_output_dir(exp_config["output_dir"], model_id, width, language)
    out_dir = ensure_dir(base_dir / "classifier-based" / "sp1_classifier")

    unique_layers, lem, _ = resolve_layers(model_id, mode)
    if mode == "single":
        model_cfg = get_model_config(model_id)
        unique_layers = [model_cfg.get("default_layer", 0)]

    d_model = get_model_config(model_id)["d_model"]
    emotion_classes = data_config["emotion_classes"]

    print(f"\n  SP-1 CLASSIFIER: {model_id} / {width} / {mode}")
    print(f"  Layers: {unique_layers}")

    # Load data with multi-hot labels
    all_texts, all_labels = load_texts_with_labels(
        data_config, cls_cfg.get("max_texts", 500))

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_id)

    all_results = {}
    for layer in unique_layers:
        print(f"\n  -- Layer {layer} --")
        layer_dir = ensure_dir(out_dir / f"layer_{layer}")

        # Load SAE encoder
        try:
            sae_encoder, sae_b_dec, d_sae = load_sae_encoder(model_id, width, layer, d_model)
        except Exception as e:
            print(f"    SKIPPED: {e}")
            continue

        print(f"  Extracting pooled activations ({len(all_texts)} texts) ...")
        pooled = extract_pooled_activations(
            all_texts, model, tokenizer, sae_encoder, sae_b_dec, layer, cls_cfg)
        print(f"    pooled: {pooled.shape}")

        # Train multi-label L1 classifier
        classifier_C = cls_cfg.get("classifier_C", 0.1)
        print(f"  Training OneVsRest L1 classifier (C={classifier_C}) ...")
        scaler = StandardScaler()
        X = scaler.fit_transform(pooled)
        y = all_labels

        clf = OneVsRestClassifier(
            LogisticRegression(
                max_iter=500, C=classifier_C,
                penalty="l1", solver="saga", random_state=42
            )
        )
        clf.fit(X, y)

        # Extract feature importance from classifier weights
        per_emo_weights = {}
        all_weights = np.zeros(d_sae)

        print(f"  Per-emotion non-zero features:")
        min_w = cls_cfg.get("min_weight", 1e-4)
        for i, emo in enumerate(emotion_classes):
            w = np.abs(clf.estimators_[i].coef_[0])
            per_emo_weights[emo] = w
            n_nonzero = (w > min_w).sum()
            all_weights = np.maximum(all_weights, w)
            print(f"    {emo:<10} non-zero: {n_nonzero:>4}  max_w: {w.max():.4f}")

        # Select features with non-zero weight in ANY emotion
        min_features = cls_cfg.get("min_features", 10)
        ranked = np.argsort(-all_weights)
        selected = [int(idx) for idx in ranked if all_weights[idx] > min_w]
        n_above = len(selected)

        if len(selected) < min_features:
            print(f"    Only {len(selected)} above min_weight={min_w}, "
                  f"taking top {min_features} by rank")
            selected = [int(idx) for idx in ranked[:min_features]]

        print(f"    Selected: {len(selected)} features ({n_above} with non-zero weight)")

        # Build f_cand.json
        candidates = []
        for rank, idx in enumerate(selected):
            emo_weights_dict = {emo: float(per_emo_weights[emo][idx])
                                for emo in emotion_classes}
            candidates.append({
                "index": idx,
                "rank": rank,
                "max_weight": float(all_weights[idx]),
                "emo_weights": emo_weights_dict,
                "n_emo_nonzero": sum(1 for v in emo_weights_dict.values() if v > min_w),
                "description": f"classifier-selected (max_w={all_weights[idx]:.3f}, "
                               f"n_emo={sum(1 for v in emo_weights_dict.values() if v > min_w)})",
            })

        # Save
        with open(layer_dir / "f_cand.json", "w") as f:
            json.dump(candidates, f, indent=2)

        # Stats CSV
        selected_set = set(selected)
        stats_df = pd.DataFrame({
            "index": list(range(d_sae)),
            "max_weight": all_weights,
            **{f"w_{emo}": per_emo_weights[emo] for emo in emotion_classes},
            "selected": [i in selected_set for i in range(d_sae)],
        })
        stats_df.to_csv(layer_dir / "classifier_stats.csv", index=False)

        save_json({
            "layer": layer, "d_sae": d_sae,
            "n_candidates": len(selected), "n_nonzero": n_above,
            "pool_method": cls_cfg.get("pool_method", "max"),
            "classifier_C": classifier_C,
            "min_weight": min_w,
            "n_texts": len(all_texts),
            "method": "classifier-weight",
        }, layer_dir / "config.json")

        all_results[layer] = {"n_cand": len(selected), "n_above": n_above}

        del sae_encoder, sae_b_dec, pooled, X, clf
        torch.cuda.empty_cache(); gc.collect()

    # Metadata
    save_json({
        "model_id": model_id, "width": width, "language": language,
        "mode": mode, "method": "classifier-weight",
        "layers": unique_layers,
        "data_source": data_config["hf_dataset_id"],
        "pool_method": cls_cfg.get("pool_method", "max"),
        "classifier_C": cls_cfg.get("classifier_C", 0.1),
        "min_weight": cls_cfg.get("min_weight", 1e-4),
        "results": {str(k): v for k, v in all_results.items()},
    }, out_dir / "metadata.json")

    # ── Free model, tokenizer, and all large arrays ──
    free_model(model, tokenizer)
    del model, tokenizer, all_texts, all_labels
    gc.collect()

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run classifier-based SP-1")
    parser.add_argument("--language", required=True)
    args = parser.parse_args()

    exp_config = load_experiment_config()
    ds_reg = load_dataset_registry()
    data_config = ds_reg["languages"][args.language]

    hf_login()

    for model_id in exp_config["models"]:
        widths = exp_config.get("model_widths", {}).get(model_id, ["16k"])
        for width in widths:
            for mode in exp_config.get("layer_modes", ["single", "multi"]):
                try:
                    run_classifier_sp1(model_id, width, args.language,
                                       data_config, exp_config, mode)
                except Exception as e:
                    print(f"\n  ERROR: {model_id}/{width}/{mode}: {e}")
                    import traceback; traceback.print_exc()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

        # ── Clear HF cache after all widths/modes for this model ──
        clear_hf_cache()
        print(f"  Cache cleared after model: {model_id}")

    print("\n\nAll classifier SP-1 experiments complete.")


if __name__ == "__main__":
    main()
