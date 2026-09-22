#!/usr/bin/env python3
"""
SP-1 Semantic Feature Selection
================================
Filters SAE features by cosine similarity between their descriptions
and a broad Indonesian-targeting task prompt.

Usage:
    python scripts/run_sp1_semantic.py --language indonesia
"""

import argparse
import gc
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (load_experiment_config, load_dataset_registry, get_model_config,
                         get_sae_id, get_width_config, model_slug, resolve_output_dir,
                         resolve_layers, get_layer_emotion_map, resolve_sae_explns_dir,
                         hf_login)
from src.data import load_emotion_dataset
from src.utils import save_json, ensure_dir


def build_task_prompt(language, emotion_classes):
    emotions_str = ", ".join(emotion_classes)
    return (
        f"Detecting and recognising human emotions expressed in {language} text. "
        f"The target emotions are: {emotions_str}. "
        f"Cultural norms, social context, idiomatic expressions, politeness strategies, "
        f"indirect speech, and community-specific emotional displays in {language} "
        f"are important for correctly interpreting emotional cues."
    )


def load_sae_repo(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SAE explanations not found: {path}")
    with open(path) as f:
        ch = f.read(1); f.seek(0)
        repo = json.load(f) if ch == "[" else [json.loads(l) for l in f if l.strip()]
    for feat in repo:
        if "index" not in feat:
            feat["index"] = int(feat["id"].split(":")[-1])
        else:
            feat["index"] = int(feat["index"])
        feat["description"] = feat.get("description") or feat.get("explanationText", "")
    total = len(repo)
    repo = [f for f in repo if f["description"].strip()]
    print(f"    Total: {total}, with descriptions: {len(repo)}")
    return repo


def candidate_retrieval(repo, e_task, encoder, epsilon):
    descs = [f["description"] for f in repo]
    e_feats = encoder.encode(descs, normalize_embeddings=True, show_progress_bar=True)
    scores = (e_feats @ e_task.T).squeeze()
    df = pd.DataFrame(repo).copy()
    df["similarity"] = scores
    df = df.sort_values("similarity", ascending=False)
    cand = df[df["similarity"] >= epsilon].reset_index(drop=True)
    print(f"    |F|={len(repo)}, |F_cand|={len(cand)}, reduction={1 - len(cand) / len(repo):.1%}")
    return cand, scores, df


def run_sp1_for_config(model_id, width, language, data_config, exp_config, mode):
    ds_reg = load_dataset_registry()
    sp1_cfg = ds_reg.get("sp1", {})
    epsilon = sp1_cfg.get("epsilon", 0.30)
    encoder_model = sp1_cfg.get("encoder_model", "all-MiniLM-L6-v2")

    emotion_classes = data_config["emotion_classes"]
    base_dir = resolve_output_dir(exp_config["output_dir"], model_id, width, language)
    sae_expl_dir = resolve_sae_explns_dir(exp_config["output_dir"], model_id, width)

    # Resolve layers
    unique_layers, lem, _ = resolve_layers(model_id, mode, language=language)
    if mode == "single":
        model_cfg = get_model_config(model_id, language=language)
        unique_layers = [model_cfg.get("default_layer", 0)]

    out_dir = ensure_dir(base_dir / "semantic-based" / "sp1_semantic")
    plots_dir = ensure_dir(out_dir / "plots")

    print(f"\n  SP-1 SEMANTIC: {model_id} / {width} / {mode}")
    print(f"  Layers: {unique_layers}")
    print(f"  Epsilon: {epsilon}")

    # Build task prompt
    task_prompt = build_task_prompt(language, emotion_classes)
    print(f"  Task prompt: {task_prompt[:80]}...")

    # Load encoder
    encoder = SentenceTransformer(encoder_model)
    e_task = encoder.encode([task_prompt], normalize_embeddings=True)

    # Process each layer
    all_results = {}
    for layer in unique_layers:
        expl_path = sae_expl_dir / f"layer_{layer}_explanations.jsonl"
        layer_dir = ensure_dir(out_dir / f"layer_{layer}")

        print(f"\n  -- Layer {layer} --")
        if not expl_path.exists():
            print(f"    SKIPPED: {expl_path} not found")
            continue

        repo = load_sae_repo(expl_path)
        cand_df, scores, full_df = candidate_retrieval(repo, e_task, encoder, epsilon)

        # Save
        records = cand_df.to_dict(orient="records")
        with open(layer_dir / "f_cand.json", "w") as f:
            json.dump(records, f, indent=2, default=str)
        full_df.to_csv(layer_dir / "all_scores.csv", index=False)

        layer_cfg = {
            "layer": layer, "n_candidates": len(records),
            "epsilon": epsilon, "encoder_model": encoder_model,
            "method": "semantic",
        }
        save_json(layer_cfg, layer_dir / "config.json")

        all_results[layer] = {"n_cand": len(records), "repo_size": len(repo)}
        print(f"    Saved {len(records)} candidates")

    # Metadata
    meta = {
        "model_id": model_id, "width": width, "language": language,
        "mode": mode, "method": "semantic",
        "layers": unique_layers, "epsilon": epsilon,
        "task_prompt": task_prompt,
    }
    save_json(meta, out_dir / "sp1_metadata.json")

    # ── Free encoder and intermediate data ──
    del encoder, e_task
    gc.collect()

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run semantic SP-1 feature selection",
        epilog="Examples:\n"
               "  python scripts/run_sp1_semantic.py --language indonesia\n"
               "  python scripts/run_sp1_semantic.py --language indonesia --model google/gemma-2-2b --width 65k --mode single\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--language", required=True)
    parser.add_argument("--model", default=None, help="Run only this model (e.g., google/gemma-2-2b)")
    parser.add_argument("--width", default=None, help="Run only this width (e.g., 16k)")
    parser.add_argument("--mode", default=None, choices=["single", "multi"],
                        help="Run only this layer mode")
    args = parser.parse_args()

    exp_config = load_experiment_config()
    ds_reg = load_dataset_registry()
    data_config = ds_reg["languages"][args.language]

    hf_login()

    models = [args.model] if args.model else exp_config["models"]

    for model_id in models:
        all_widths = exp_config.get("model_widths", {}).get(model_id, ["16k"])
        widths = [args.width] if args.width else all_widths
        all_modes = exp_config.get("layer_modes", ["single", "multi"])
        modes = [args.mode] if args.mode else all_modes

        for width in widths:
            for mode in modes:
                try:
                    run_sp1_for_config(
                        model_id, width, args.language, data_config, exp_config, mode)
                except Exception as e:
                    print(f"\n  ERROR: {model_id}/{width}/{mode}: {e}")
                    import traceback; traceback.print_exc()
                    gc.collect()
                    continue

    print("\n\nAll SP-1 semantic experiments complete.")


if __name__ == "__main__":
    main()