#!/usr/bin/env python3
"""
SP-2 Coefficient Optimisation
===============================
Learns steering coefficients α via gradient-based optimization.
Runs all model × width × mode × candidate_method combinations for a language.

Usage:
    python scripts/run_sp2_optimization.py --language indonesia
"""

import argparse
import gc
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from pathlib import Path
from sklearn.metrics import f1_score

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (load_experiment_config, load_dataset_registry, get_model_config,
                         get_width_config, get_sae_id, model_slug, resolve_output_dir,
                         resolve_layers, get_layer_emotion_map, hf_login)
from src.data import load_emotion_dataset, load_binary_pairs
from src.model import load_model_and_tokenizer, free_model, get_token_ids, DEVICE
from src.sae import load_pretrained_sae_decoder, load_candidates
from src.steering import SharedSteeringContext
from src.evaluation import threshold_sweep
from src.prompt import build_binary_prompt
from src.utils import (save_json, ensure_dir, plot_training_curves,
                        plot_threshold_sweep, plot_score_distributions)


def prepare_batches(df, tokenizer, emotion_to_idx, unique_layers, emo_to_layer,
                    batch_size, max_seq_len):
    prompts = [build_binary_prompt(r["text"], r["emotion_query"]) for _, r in df.iterrows()]
    labels = torch.tensor(df["label"].values, dtype=torch.float32)
    emo_ids = torch.tensor([emotion_to_idx[e] for e in df["emotion_query"]], dtype=torch.long)

    emo_to_layer_idx = {e: unique_layers.index(emo_to_layer[e])
                        for e in emotion_to_idx if e in emo_to_layer}
    layer_ids = torch.tensor([emo_to_layer_idx.get(e, 0) for e in df["emotion_query"]],
                             dtype=torch.long)

    batches = []
    for i in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                        truncation=True, max_length=max_seq_len).to(DEVICE)
        batches.append((
            enc.input_ids, enc.attention_mask,
            labels[i:i + batch_size].to(DEVICE),
            emo_ids[i:i + batch_size].to(DEVICE),
            layer_ids[i:i + batch_size].to(DEVICE),
        ))
    return batches


def forward_unsteered(ids, mask, model, yes_id, no_id):
    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = model(input_ids=ids, attention_mask=mask)
    sl = mask.sum(dim=1) - 1
    bi = torch.arange(ids.size(0), device=DEVICE)
    return (out.logits[bi, sl, yes_id] - out.logits[bi, sl, no_id]).float()


def forward_steered(ids, mask, model, alpha_dict, V_dict, unique_layers, n_layers,
                    yes_id, no_id):
    with SharedSteeringContext(model, alpha_dict, V_dict, unique_layers, n_layers):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=ids, attention_mask=mask)
    sl = mask.sum(dim=1) - 1
    bi = torch.arange(ids.size(0), device=DEVICE)
    return (out.logits[bi, sl, yes_id] - out.logits[bi, sl, no_id]).float()


def train_sp2(model, V_CAND, INDICES, train_batches, eval_batches, unique_layers,
              n_layers, yes_id, no_id, config):
    alpha_dict = {}
    V_dict = {}
    for ul_idx, layer in enumerate(unique_layers):
        n_cand = V_CAND[layer].shape[0]
        alpha_dict[ul_idx] = nn.Parameter(torch.full((n_cand,), 0.01, device=DEVICE))
        V_dict[ul_idx] = V_CAND[layer]
        print(f"  Layer {layer:>2}: alpha ({n_cand},)")

    all_params = list(alpha_dict.values())
    optimizer = Adam([{"params": all_params, "lr": config["lr"]}])

    n_pos = sum(int((l > 0.5).sum()) for _, _, l, _, _ in train_batches)
    n_neg = sum(int((l <= 0.5).sum()) for _, _, l, _, _ in train_batches)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    lam = config["lambda_l1"]
    epochs = config["n_epochs"]
    pat = config.get("patience", 10)
    tau = config.get("tau", 0.01)

    history = {"train_loss": [], "eval_loss": [], "alpha_nnz": [], "alpha_l1": []}
    best_loss = float("inf")
    best_alpha = {k: v.detach().cpu().clone() for k, v in alpha_dict.items()}
    best_ep = 0
    no_imp = 0

    total_params = sum(a.numel() for a in all_params)
    print(f"\n  Training: {epochs} epochs, lr={config['lr']}, lam={lam}")
    print(f"  Total params: {total_params}, pos_weight={pos_weight.item():.2f}")

    for ep in range(epochs):
        ep_loss = 0.0
        nb = 0
        for ids, mask, labels, emo_ids, layer_ids in train_batches:
            optimizer.zero_grad()
            scores = forward_steered(ids, mask, model, alpha_dict, V_dict,
                                     unique_layers, n_layers, yes_id, no_id)
            l1 = sum(F.softplus(a).sum() for a in all_params)
            loss = loss_fn(scores, labels) + lam * l1
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            nb += 1
        tl = ep_loss / max(nb, 1)

        el = 0.0
        ne = 0
        with torch.no_grad():
            for ids, mask, labels, emo_ids, layer_ids in eval_batches:
                scores = forward_steered(ids, mask, model, alpha_dict, V_dict,
                                         unique_layers, n_layers, yes_id, no_id)
                el += loss_fn(scores, labels).item()
                ne += 1
        el = el / max(ne, 1)

        with torch.no_grad():
            all_sp = torch.cat([F.softplus(a).flatten() for a in all_params])
            nnz = int((all_sp > tau).sum())
            l1v = float(all_sp.sum())

        history["train_loss"].append(tl)
        history["eval_loss"].append(el)
        history["alpha_nnz"].append(nnz)
        history["alpha_l1"].append(l1v)

        if el < best_loss:
            best_loss = el
            best_alpha = {k: v.detach().cpu().clone() for k, v in alpha_dict.items()}
            best_ep = ep + 1
            no_imp = 0
            mk = " * best"
        else:
            no_imp += 1
            mk = f" ({no_imp}/{pat})"

        print(f"  Epoch {ep + 1:>3}/{epochs}  train={tl:.4f}  eval={el:.4f}  "
              f"nnz={nnz:>5}  L1={l1v:.4f}{mk}")

        if no_imp >= pat:
            print(f"\n  Early stopping at epoch {ep + 1}")
            break

    print(f"  Restoring best from epoch {best_ep}")
    return best_alpha, V_dict, history


def run_sp2(model_id, width, language, data_config, exp_config, mode, cand_method):
    ds_reg = load_dataset_registry()
    sp2_cfg = ds_reg.get("sp2", {})
    config = {**sp2_cfg}

    base_dir = resolve_output_dir(exp_config["output_dir"], model_id, width, language)
    emotion_classes = data_config["emotion_classes"]
    emotion_to_idx = {e: i for i, e in enumerate(emotion_classes)}

    # Resolve layers
    unique_layers, lem, emo_to_layer = resolve_layers(model_id, mode)
    if mode == "single":
        model_cfg = get_model_config(model_id)
        dl = model_cfg.get("default_layer", 0)
        unique_layers = [dl]
        emo_to_layer = {e: dl for e in emotion_classes}
        lem = {dl: emotion_classes}

    # Output dir
    mode_tag = "multilayer" if mode == "multi" else f"layer{unique_layers[0]}"
    sp2_dir = ensure_dir(base_dir / cand_method / f"sp2_{mode_tag}")
    plots_dir = ensure_dir(sp2_dir / "plots")

    # Candidate paths — new directory structure
    if cand_method == "classifier-based":
        sp1_subdir = "classifier-based/sp1_classifier"
    else:
        sp1_subdir = "semantic-based/sp1_semantic"
    sp1_paths = {l: base_dir / sp1_subdir / f"layer_{l}" / "f_cand.json" for l in unique_layers}

    print(f"\n{'='*60}")
    print(f"  SP-2: {model_id} / {width} / {mode} / {cand_method}")
    print(f"  Layers: {unique_layers}")
    print(f"  Output: {sp2_dir}")
    print(f"{'='*60}")

    # Check candidates exist
    for l, p in sp1_paths.items():
        if not p.exists():
            print(f"  SKIPPED: candidates missing at {p}")
            return None

    # Load candidates
    CANDS = {}
    INDICES = {}
    for layer in unique_layers:
        cands = load_candidates(sp1_paths[layer])
        CANDS[layer] = cands
        INDICES[layer] = [f["index"] for f in cands]
        print(f"  Layer {layer:>2}: {len(cands)} candidates")

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_id)
    n_layers = model.config.num_hidden_layers
    yes_id, no_id = get_token_ids(tokenizer)

    # Load SAE decoders
    V_CAND = {}
    for layer in unique_layers:
        V_CAND[layer] = load_pretrained_sae_decoder(
            model_id, width, layer, INDICES[layer])

    # Load data — train from train split, eval from eval split
    train_df, _ = load_binary_pairs(data_config, config.get("max_train_samples", 500),
                                     split_name="train")
    eval_df, _ = load_binary_pairs(data_config, config.get("max_eval_samples", 100))

    train_batches = prepare_batches(
        train_df, tokenizer, emotion_to_idx, unique_layers, emo_to_layer,
        config.get("batch_size", 4), config.get("max_seq_len", 128))
    eval_batches = prepare_batches(
        eval_df, tokenizer, emotion_to_idx, unique_layers, emo_to_layer,
        config.get("batch_size", 4), config.get("max_seq_len", 128))

    print(f"  Train: {len(train_batches)} batches, Eval: {len(eval_batches)} batches")

    # Train
    alpha_star, V_dict, train_history = train_sp2(
        model, V_CAND, INDICES, train_batches, eval_batches,
        unique_layers, n_layers, yes_id, no_id, config)

    # Evaluate
    print("\n  Collecting scores ...")
    all_su, all_ss, all_lb = [], [], []
    with torch.no_grad():
        alpha_dev = {k: v.to(DEVICE) for k, v in alpha_star.items()}
        for ids, mask, labels, emo_ids, layer_ids in eval_batches:
            all_su.append(forward_unsteered(ids, mask, model, yes_id, no_id).cpu())
            all_ss.append(forward_steered(ids, mask, model, alpha_dev, V_dict,
                                          unique_layers, n_layers, yes_id, no_id).cpu())
            all_lb.append(labels.cpu())

    all_su = torch.cat(all_su).numpy()
    all_ss = torch.cat(all_ss).numpy()
    all_lb = torch.cat(all_lb).numpy()

    bts, f1s, sweep_s = threshold_sweep(all_ss, all_lb)
    btu, f1u, sweep_u = threshold_sweep(all_su, all_lb)

    print(f"\n  Steered:   t={bts:+.1f}  F1={f1s:.4f}")
    print(f"  Unsteered: t={btu:+.1f}  F1={f1u:.4f}")
    print(f"  Delta:     {f1s - f1u:+.4f}")

    # Save
    alpha_ser = {str(l): alpha_star[i].tolist() for i, l in enumerate(unique_layers)}
    save_dict = {
        "mode": mode, "cand_method": cand_method,
        "alpha_per_layer": alpha_ser,
        "indices_per_layer": {str(l): INDICES[l] for l in unique_layers},
        "layer_emotion_map": {str(l): lem[l] for l in unique_layers} if lem else {},
        "unique_layers": unique_layers, "emotion_classes": emotion_classes,
        "tau": config.get("tau", 0.01), "lambda_l1": config["lambda_l1"],
        "lr": config["lr"], "n_epochs": config["n_epochs"],
        "model_id": model_id, "width": width, "language": language,
        "steered_f1": float(f1s), "unsteered_f1": float(f1u),
        "best_thresh_steered": float(bts), "best_thresh_unsteered": float(btu),
    }
    save_json(save_dict, sp2_dir / "alpha_star.json")
    save_json(train_history, sp2_dir / "train_history.json")

    # Plots
    plot_training_curves(train_history, plots_dir / "training_curves.png",
                         title_prefix=f"{model_slug(model_id)}/{width}/{mode} ")
    plot_threshold_sweep({"Steered": sweep_s, "Unsteered": sweep_u}, all_lb,
                          plots_dir / "threshold_sweep.png")
    plot_score_distributions({"Steered": all_ss, "Unsteered": all_su}, all_lb,
                              plots_dir / "score_distributions.png")

    # ── Free all GPU tensors, model, tokenizer ──
    free_model(model, tokenizer)
    del model, tokenizer
    del V_CAND, alpha_star, alpha_dev, V_dict
    del train_batches, eval_batches, train_df, eval_df
    del all_su, all_ss, all_lb
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n  Saved to {sp2_dir}")
    return save_dict


def main():
    parser = argparse.ArgumentParser(description="Run SP-2 coefficient optimisation")
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
                for cand_method in exp_config.get("candidate_methods", ["semantic-based"]):
                    try:
                        run_sp2(model_id, width, args.language,
                                data_config, exp_config, mode, cand_method)
                    except Exception as e:
                        print(f"\n  ERROR: {model_id}/{width}/{mode}/{cand_method}: {e}")
                        import traceback; traceback.print_exc()
                        # Ensure cleanup even on error
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

    print("\n\nAll SP-2 experiments complete.")


if __name__ == "__main__":
    main()
