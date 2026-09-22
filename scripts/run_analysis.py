#!/usr/bin/env python3
"""
SP-2 Analysis & Layer Ablation
================================
Loads saved SP-2 results and runs ablation + per-emotion analysis.

Usage:
    python scripts/run_analysis.py --language indonesia --model google/gemma-2-2b \
        --width 16k --mode multi --cand_method semantic
"""

import argparse
import gc
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import f1_score

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (load_experiment_config, load_dataset_registry, get_model_config,
                         get_width_config, get_sae_id, model_slug, resolve_output_dir,
                         resolve_layers, hf_login)
from src.data import load_binary_pairs
from src.model import load_model_and_tokenizer, free_model, get_token_ids, DEVICE, clear_hf_cache
from src.sae import load_pretrained_sae_decoder, load_candidates
from src.steering import SelectiveSteeringContext
from src.prompt import build_binary_prompt
from src.evaluation import threshold_sweep
from src.utils import (save_json, load_json, ensure_dir, plot_training_curves,
                        plot_threshold_sweep, plot_score_distributions,
                        plot_layer_ablation)


def forward_unsteered(ids, mask, model, yes_id, no_id):
    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = model(input_ids=ids, attention_mask=mask)
    sl = mask.sum(dim=1) - 1
    bi = torch.arange(ids.size(0), device=ids.device)
    return (out.logits[bi, sl, yes_id] - out.logits[bi, sl, no_id]).float()


def forward_with_active_layers(ids, mask, model, alpha_dict, V_dict,
                                unique_layers, active_layers, n_layers,
                                yes_id, no_id, emo_ids_batch=None,
                                layer_emotion_map=None, emotion_to_idx=None):
    """
    Forward pass with steering applied only at `active_layers`.
    Uses SelectiveSteeringContext which handles both 1D and 2D alpha,
    multi-GPU device placement, and emotion-conditioned routing.
    """
    if not active_layers:
        return forward_unsteered(ids, mask, model, yes_id, no_id)

    with SelectiveSteeringContext(
        model, alpha_dict, V_dict, unique_layers, active_layers, n_layers,
        emo_ids=emo_ids_batch,
        layer_emotion_map=layer_emotion_map,
        emotion_to_idx=emotion_to_idx,
    ):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=ids, attention_mask=mask)

    sl = mask.sum(dim=1) - 1
    bi = torch.arange(ids.size(0), device=ids.device)
    return (out.logits[bi, sl, yes_id] - out.logits[bi, sl, no_id]).float()


def main():
    parser = argparse.ArgumentParser(description="Run SP-2 analysis")
    parser.add_argument("--language", required=True)
    parser.add_argument("--model", required=True, help="Model ID (e.g., google/gemma-2-2b)")
    parser.add_argument("--width", required=True, help="SAE width (e.g., 16k)")
    parser.add_argument("--mode", required=True, choices=["single", "multi"])
    parser.add_argument("--cand_method", required=True, choices=["semantic-based", "classifier-based"])
    args = parser.parse_args()

    exp_config = load_experiment_config()
    ds_reg = load_dataset_registry()
    data_config = ds_reg["languages"][args.language]
    sp2_cfg = ds_reg.get("sp2", {})

    hf_login()

    model_id = args.model
    width = args.width
    mode = args.mode
    cand_method = args.cand_method

    emotion_classes = data_config["emotion_classes"]
    emotion_to_idx = {e: i for i, e in enumerate(emotion_classes)}

    base_dir = resolve_output_dir(exp_config["output_dir"], model_id, width, args.language)
    unique_layers_raw, lem, emo_to_layer = resolve_layers(model_id, mode, language=args.language)
    if mode == "single":
        model_cfg = get_model_config(model_id, language=args.language)
        dl = model_cfg.get("default_layer", 0)
        unique_layers_raw = [dl]

    mode_tag = "multilayer" if mode == "multi" else f"layer{unique_layers_raw[0]}"
    sp2_dir = base_dir / cand_method / f"sp2_{mode_tag}"
    analysis_dir = ensure_dir(sp2_dir / "analysis")
    plots_dir = ensure_dir(analysis_dir / "plots")

    print(f"\n{'='*60}")
    print(f"  ANALYSIS: {model_id} / {width} / {mode} / {cand_method}")
    print(f"  SP2 dir: {sp2_dir}")
    print(f"{'='*60}")

    # ── Load saved outputs ──
    alpha_path = sp2_dir / "alpha_star.json"
    if not alpha_path.exists():
        print(f"  ERROR: {alpha_path} not found")
        return

    save_dict = load_json(alpha_path)
    train_history = load_json(sp2_dir / "train_history.json")

    UNIQUE_LAYERS = save_dict["unique_layers"]
    alpha_star = {}
    for ul_idx, layer in enumerate(UNIQUE_LAYERS):
        alpha_star[ul_idx] = torch.tensor(save_dict["alpha_per_layer"][str(layer)])

    # Rebuild INDICES
    if save_dict.get("indices_per_layer"):
        INDICES = {int(k): v for k, v in save_dict["indices_per_layer"].items()}
    else:
        INDICES = {}
        for ul_idx, layer in enumerate(UNIQUE_LAYERS):
            n = alpha_star[ul_idx].shape[-1]
            INDICES[layer] = list(range(n))

    print(f"  Layers: {UNIQUE_LAYERS}")
    print(f"  Steered F1: {save_dict.get('steered_f1', '?')}")
    print(f"  Unsteered F1: {save_dict.get('unsteered_f1', '?')}")

    # ── Offline: training curves ──
    plot_training_curves(train_history, plots_dir / "training_curves.png",
                         title_prefix=f"{model_slug(model_id)}/{width} ")

    # ── Load model for online analysis ──
    model, tokenizer = load_model_and_tokenizer(model_id)
    n_layers_model = model.config.num_hidden_layers
    yes_id, no_id = get_token_ids(tokenizer)

    # Load SAE decoders — keyed by ul_idx to match SelectiveSteeringContext
    V_CAND = {}
    for ul_idx, layer in enumerate(UNIQUE_LAYERS):
        V_CAND[ul_idx] = load_pretrained_sae_decoder(
            model_id, width, layer, language=args.language, indices=INDICES.get(layer))

    # Rebuild layer_emotion_map from saved data
    saved_lem = save_dict.get("layer_emotion_map", {})
    layer_emotion_map = {int(k): v for k, v in saved_lem.items()} if saved_lem else None

    # Load eval data
    eval_df, _ = load_binary_pairs(data_config, sp2_cfg.get("max_eval_samples", 100))
    prompts = [build_binary_prompt(r["text"], r["emotion_query"]) for _, r in eval_df.iterrows()]
    labels_t = torch.tensor(eval_df["label"].values, dtype=torch.float32)
    emo_ids_t = torch.tensor([emotion_to_idx[e] for e in eval_df["emotion_query"]], dtype=torch.long)

    bs = sp2_cfg.get("batch_size", 4)
    eval_batches = []
    for i in range(0, len(prompts), bs):
        enc = tokenizer(prompts[i:i + bs], return_tensors="pt", padding=True,
                        truncation=True, max_length=sp2_cfg.get("max_seq_len", 128)).to(DEVICE)
        eval_batches.append((enc.input_ids, enc.attention_mask,
                             labels_t[i:i + bs].to(DEVICE), emo_ids_t[i:i + bs].to(DEVICE)))

    # ── Layer ablation ──
    configs = {"Unsteered": []}
    for layer in UNIQUE_LAYERS:
        configs[f"L{layer} only"] = [layer]
    configs["All layers"] = list(UNIQUE_LAYERS)

    print(f"\n  Running ablation ({len(configs)} configs) ...")
    all_scores = {}
    all_labels = None

    for name, active in configs.items():
        print(f"    {name} ...", end="", flush=True)
        scores_list = []
        labels_list = []
        with torch.no_grad():
            for ids, mask, labels, emo_ids in eval_batches:
                s = forward_with_active_layers(
                    ids, mask, model, alpha_star, V_CAND,
                    UNIQUE_LAYERS, active, n_layers_model, yes_id, no_id,
                    emo_ids_batch=emo_ids,
                    layer_emotion_map=layer_emotion_map,
                    emotion_to_idx=emotion_to_idx).cpu()
                scores_list.append(s)
                labels_list.append(labels.cpu())
        all_scores[name] = torch.cat(scores_list).numpy()
        if all_labels is None:
            all_labels = torch.cat(labels_list).numpy()
        print(" done")

    emotions_arr = eval_df["emotion_query"].values[:len(all_labels)]

    # Threshold sweep per config
    config_results = {}
    sweep_dfs = {}
    for name in configs:
        bt, f1, sdf = threshold_sweep(all_scores[name], all_labels)
        config_results[name] = {"macro_f1": f1, "threshold": bt}
        sweep_dfs[name] = sdf

    base_f1 = config_results["Unsteered"]["macro_f1"]
    best_name = max(config_results, key=lambda n: config_results[n]["macro_f1"])

    print(f"\n  Baseline:  {base_f1:.4f}")
    print(f"  Best:      {best_name} ({config_results[best_name]['macro_f1']:.4f})")

    # Per-emotion breakdown
    pe_rows = []
    for name in configs:
        sc = all_scores[name]
        bt = config_results[name]["threshold"]
        preds = (sc > bt).astype(float)
        row = {"Config": name}
        for emo in emotion_classes:
            m = (emotions_arr == emo)
            row[emo] = f1_score(all_labels[m], preds[m], zero_division=0)
        row["Macro"] = config_results[name]["macro_f1"]
        pe_rows.append(row)

    pe_df = pd.DataFrame(pe_rows)
    pe_df.to_csv(analysis_dir / "per_emotion_ablation.csv", index=False)
    print(f"\n  Per-emotion results saved")

    # Plots
    plot_threshold_sweep(sweep_dfs, all_labels, plots_dir / "ablation_threshold.png")
    plot_score_distributions(all_scores, all_labels, plots_dir / "ablation_scores.png")
    plot_layer_ablation(config_results, base_f1, emotion_classes, pe_df,
                        plots_dir / "ablation_summary.png")

    # Recommendation
    print(f"\n== Recommendation ==")
    for name in configs:
        if name == "Unsteered":
            continue
        d = config_results[name]["macro_f1"] - base_f1
        verdict = "HELPS" if d > 0.005 else "HURTS" if d < -0.005 else "neutral"
        print(f"  {name:<16} F1={config_results[name]['macro_f1']:.4f} ({d:+.4f}) {verdict}")

    # Save analysis
    save_json({
        "configs": {n: config_results[n] for n in configs},
        "best_config": best_name,
        "baseline_f1": base_f1,
        "model_id": model_id, "width": width, "mode": mode,
        "cand_method": cand_method, "language": args.language,
    }, analysis_dir / "ablation_results.json")

    # ── Free all GPU tensors, model, tokenizer ──
    free_model(model)
    del model
    del V_CAND, alpha_star, eval_batches
    del all_scores, all_labels
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    clear_hf_cache()

    print(f"\n  Analysis saved to {analysis_dir}")


if __name__ == "__main__":
    main()