

################################################################################
# FILE: layer_probing_experiment.ipynb
################################################################################

# ── CELL 3 ──
import json, numpy as np, pandas as pd
from pathlib import Path

import torch, torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

import warnings; warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# ── CELL 5 ──
EXPERIMENT = {
    "model_id"  : "google/gemma-2-9b",
    "language"  : "indonesia",
}

DATASET_CONFIG = {
    "hf_dataset_id"   : "brighter-dataset/BRIGHTER-emotion-categories",
    "hf_subset"       : "ind",
    "text_column"     : "text",
    "emotion_classes" : ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"],
}

PROBE_CONFIG = {
    "max_train_samples" : 300,      # texts (not pairs) — probing doesn't need huge data
    "max_eval_samples"  : 100,
    "max_seq_len"       : 128,
    "batch_size"        : 8,        # for hidden state extraction
}

print(f"Model: {EXPERIMENT['model_id']}")
print(f"Dataset: {DATASET_CONFIG['hf_dataset_id']} ({DATASET_CONFIG['hf_subset']})")

# ── CELL 7 ──
tokenizer = AutoTokenizer.from_pretrained(EXPERIMENT["model_id"])
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    EXPERIMENT["model_id"], torch_dtype=torch.float16,
    device_map="auto", attn_implementation="eager",
)
model.eval(); model.to(DEVICE)
for p in model.parameters(): p.requires_grad = False

d_model  = model.config.hidden_size
n_layers = model.config.num_hidden_layers
print(f"d_model: {d_model}, n_layers: {n_layers}")

# ── CELL 9 ──
def load_data(cfg, probe_cfg):
    load_kwargs = {"path": cfg["hf_dataset_id"]}
    if cfg["hf_subset"]: load_kwargs["name"] = cfg["hf_subset"]
    ds = load_dataset(**load_kwargs)

    label_names = cfg["emotion_classes"]
    emotion_cols = [e for e in label_names if e != "neutral"]

    def to_df(split_name, max_samples):
        split = ds[split_name]
        if max_samples: split = split.select(range(min(max_samples, len(split))))
        df = split.to_pandas()
        df["neutral"] = (df[emotion_cols].sum(axis=1) == 0).astype(int)
        rows = []
        for _, row in df.iterrows():
            for emo in label_names:
                rows.append({"text": row["text"], "emotion_query": emo, "label": int(row[emo])})
        return pd.DataFrame(rows)

    if "validation" in ds and "dev" not in ds: ds["dev"] = ds["validation"]
    available = set(ds.keys())
    if "train" in available and "test" in available:
        train_sp, eval_sp = "train", "test"
    elif "train" in available and "dev" in available:
        train_sp, eval_sp = "train", "dev"
    else:
        train_sp, eval_sp = "dev", "test"

    train_df = to_df(train_sp, probe_cfg["max_train_samples"])
    eval_df  = to_df(eval_sp,  probe_cfg["max_eval_samples"])
    print(f"  Train: {len(train_df)} pairs, Eval: {len(eval_df)} pairs")
    return train_df, eval_df, label_names

PROMPT_TEMPLATE = 'Is the emotion "{emotion}" present in the following text? Answer only yes or no.\nText: "{text}"\nAnswer:'
def build_prompt(text, emotion): return PROMPT_TEMPLATE.format(text=text, emotion=emotion)

train_df, eval_df, LABEL_NAMES = load_data(DATASET_CONFIG, PROBE_CONFIG)
print(f"\nLabel balance (train):")
print(f"  yes: {train_df['label'].sum()}, no: {len(train_df) - train_df['label'].sum()}")

# ── CELL 11 ──
def extract_hidden_states(df, tokenizer, model, batch_size, max_seq_len):
    """
    Returns: dict mapping layer_idx -> numpy array of shape (n_samples, d_model)
    """
    prompts = [build_prompt(r["text"], r["emotion_query"]) for _, r in df.iterrows()]
    labels  = df["label"].values
    emotions = df["emotion_query"].values

    # Initialize storage: one list per layer
    n_layers = model.config.num_hidden_layers
    layer_states = {l: [] for l in range(n_layers + 1)}  # +1 for embedding layer

    n_batches = (len(prompts) + batch_size - 1) // batch_size
    print(f"  Extracting hidden states from {len(prompts)} prompts ({n_batches} batches) ...")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_seq_len).to(DEVICE)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                output_hidden_states=True,
            )

        # Extract last-token hidden state at each layer
        seq_lengths = enc.attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(enc.input_ids.size(0), device=DEVICE)

        # outputs.hidden_states: tuple of (n_layers+1) tensors, each (batch, seq, d_model)
        # Index 0 = embedding output, index 1..n_layers = layer outputs
        for l, hs in enumerate(outputs.hidden_states):
            last_token = hs[batch_idx, seq_lengths].float().cpu().numpy()
            layer_states[l].append(last_token)

        if (i // batch_size + 1) % 20 == 0:
            print(f"    Batch {i // batch_size + 1}/{n_batches}")

    # Concatenate
    for l in layer_states:
        layer_states[l] = np.concatenate(layer_states[l], axis=0)

    print(f"  Done. Shape per layer: {layer_states[0].shape}")
    return layer_states, labels, emotions


print("Extracting TRAIN hidden states:")
train_states, train_labels, train_emotions = extract_hidden_states(
    train_df, tokenizer, model, PROBE_CONFIG["batch_size"], PROBE_CONFIG["max_seq_len"]
)

print("\nExtracting EVAL hidden states:")
eval_states, eval_labels, eval_emotions = extract_hidden_states(
    eval_df, tokenizer, model, PROBE_CONFIG["batch_size"], PROBE_CONFIG["max_seq_len"]
)

# ── CELL 13 ──
print("== Training probes at each layer ==\n")

probe_results = []

for layer_idx in range(n_layers + 1):
    X_train = train_states[layer_idx]
    X_eval  = eval_states[layer_idx]
    y_train = train_labels
    y_eval  = eval_labels

    # Standardize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_eval_s  = scaler.transform(X_eval)

    # Train logistic regression probe
    probe = LogisticRegression(
        max_iter=500, C=1.0, solver="lbfgs", random_state=42
    )
    probe.fit(X_train_s, y_train)

    # Evaluate
    y_pred = probe.predict(X_eval_s)
    acc    = accuracy_score(y_eval, y_pred)
    f1_mac = f1_score(y_eval, y_pred, average="macro", zero_division=0)
    f1_yes = f1_score(y_eval, y_pred, pos_label=1, zero_division=0)
    f1_no  = f1_score(y_eval, y_pred, pos_label=0, zero_division=0)

    layer_name = "emb" if layer_idx == 0 else f"L{layer_idx - 1}"

    probe_results.append({
        "layer_idx"  : layer_idx,
        "layer_name" : layer_name,
        "actual_layer": layer_idx - 1,    # -1 = embedding
        "accuracy"   : acc,
        "macro_f1"   : f1_mac,
        "f1_yes"     : f1_yes,
        "f1_no"      : f1_no,
    })

    print(f"  {layer_name:>4}  |  acc={acc:.4f}  macro_f1={f1_mac:.4f}  f1(yes)={f1_yes:.4f}  f1(no)={f1_no:.4f}")

probe_df = pd.DataFrame(probe_results)

# ── CELL 15 ──
# Find best layer
best_row = probe_df.loc[probe_df["macro_f1"].idxmax()]
print(f"Best layer by macro F1: {best_row['layer_name']} (F1 = {best_row['macro_f1']:.4f})\n")

# Top 5
print("Top 5 layers:")
top5 = probe_df.nlargest(7, "macro_f1")
display(top5[["layer_name", "actual_layer", "accuracy", "macro_f1", "f1_yes", "f1_no"]]
    .style.format({"accuracy": "{:.4f}", "macro_f1": "{:.4f}", "f1_yes": "{:.4f}", "f1_no": "{:.4f}"}))

# Plot
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Macro F1 by layer
    colors = ["#2ECC71" if r["layer_name"] == best_row["layer_name"] else "#7F77DD"
              for _, r in probe_df.iterrows()]
    axes[0].bar(probe_df["layer_name"], probe_df["macro_f1"], color=colors)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("Probe Macro F1 by Layer")
    axes[0].tick_params(axis="x", rotation=90, labelsize=7)
    axes[0].axhline(best_row["macro_f1"], color="green", linestyle="--", alpha=0.3)

    # F1(yes) vs F1(no)
    axes[1].plot(probe_df["layer_name"], probe_df["f1_yes"], label="F1(yes)", color="#3498DB", marker="o", markersize=3)
    axes[1].plot(probe_df["layer_name"], probe_df["f1_no"], label="F1(no)", color="#E74C3C", marker="o", markersize=3)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("F1(yes) vs F1(no) by Layer")
    axes[1].tick_params(axis="x", rotation=90, labelsize=7)
    axes[1].legend()

    plt.tight_layout()
    plt.show()
except ImportError:
    pass

# ── CELL 17 ──
print("== Per-Emotion Probing ==\n")

per_emo_results = []

for emo in DATASET_CONFIG["emotion_classes"]:
    train_mask = (train_emotions == emo)
    eval_mask  = (eval_emotions == emo)

    if train_mask.sum() == 0 or eval_mask.sum() == 0:
        continue

    y_train_e = train_labels[train_mask]
    y_eval_e  = eval_labels[eval_mask]

    # Skip if only one class present
    if len(np.unique(y_train_e)) < 2 or len(np.unique(y_eval_e)) < 2:
        print(f"  {emo:<10} skipped (single class in train or eval)")
        continue

    best_f1 = 0
    best_layer = 0
    layer_f1s = []

    for layer_idx in range(n_layers + 1):
        X_tr = train_states[layer_idx][train_mask]
        X_ev = eval_states[layer_idx][eval_mask]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_ev_s = scaler.transform(X_ev)

        probe = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=42)
        probe.fit(X_tr_s, y_train_e)
        y_pred = probe.predict(X_ev_s)
        f1 = f1_score(y_eval_e, y_pred, average="macro", zero_division=0)
        layer_f1s.append(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_layer = layer_idx

    layer_name = "emb" if best_layer == 0 else f"L{best_layer - 1}"
    per_emo_results.append({
        "Emotion"    : emo,
        "Best layer" : layer_name,
        "Best F1"    : best_f1,
        "N_pos"      : int(y_eval_e.sum()),
        "N_total"    : int(eval_mask.sum()),
        "layer_f1s"  : layer_f1s,
    })
    print(f"  {emo:<10} best={layer_name:<5} F1={best_f1:.4f}  (pos={int(y_eval_e.sum())}/{int(eval_mask.sum())})")

print()
per_emo_df = pd.DataFrame([{k:v for k,v in r.items() if k != "layer_f1s"} for r in per_emo_results])
display(per_emo_df.style.format({"Best F1": "{:.4f}"}))

# Heatmap
try:
    import matplotlib.pyplot as plt

    emo_names = [r["Emotion"] for r in per_emo_results]
    heatmap_data = np.array([r["layer_f1s"] for r in per_emo_results])
    layer_names = ["emb"] + [f"L{i}" for i in range(n_layers)]

    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(emo_names)))
    ax.set_yticklabels(emo_names)
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=90, fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_title("Probe F1 by Layer x Emotion (brighter = better)")
    plt.colorbar(im, label="Macro F1")
    plt.tight_layout()
    plt.show()
except ImportError:
    pass

# ── CELL 19 ──
print("== Layer Selection Recommendation ==\n")

# Overall best
overall_best = probe_df.loc[probe_df["macro_f1"].idxmax()]
print(f"1. OVERALL BEST LAYER: {overall_best['layer_name']} (actual layer {int(overall_best['actual_layer'])})")
print(f"   Macro F1 = {overall_best['macro_f1']:.4f}\n")

# Per-emotion consensus
print("2. PER-EMOTION BEST LAYERS:")
for _, r in per_emo_df.iterrows():
    print(f"   {r['Emotion']:<10} -> {r['Best layer']}")

# Most common best layer across emotions
from collections import Counter
layer_votes = Counter(per_emo_df["Best layer"].values)
consensus_layer = layer_votes.most_common(1)[0]
print(f"\n3. CONSENSUS (most common best): {consensus_layer[0]} ({consensus_layer[1]}/{len(per_emo_df)} emotions)")

# Check if there is a clear winner or spread
top3_overall = probe_df.nlargest(3, "macro_f1")
spread = top3_overall["macro_f1"].max() - top3_overall["macro_f1"].min()
print(f"\n4. TOP-3 SPREAD: {spread:.4f}")
if spread < 0.01:
    print("   Very close -- any of the top 3 would work")
else:
    print(f"   Clear winner: {overall_best['layer_name']}")

# Final recommendation
rec_layer = int(overall_best["actual_layer"])
if rec_layer < 0: rec_layer = 0
print(f"\n5. RECOMMENDED injection_layer = {rec_layer}")
print(f"\n   Update SAE_REGISTRY:")
print(f'   "sae_id": "layer_{rec_layer}/width_16k/canonical",')
print(f'   "injection_layer": {rec_layer},')

# Save results
out_dir = Path(f"/content/drive/MyDrive/sae_outputs/{EXPERIMENT['model_id'].split('/')[-1].lower()}/{EXPERIMENT['language']}/probing")
out_dir.mkdir(parents=True, exist_ok=True)
probe_df.to_csv(out_dir / "probe_results.csv", index=False)
per_emo_df.to_csv(out_dir / "per_emotion_probe_results.csv", index=False)
print(f"\n   Results saved to {out_dir}")


################################################################################
# FILE: sp1_candidate_retrieval_multi_layer.ipynb
################################################################################

# ── CELL 3 ──
import json, numpy as np, pandas as pd
from pathlib import Path
from google.colab import runtime

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

import warnings; warnings.filterwarnings('ignore')
print('Imports OK.')

# ── CELL 5 ──
# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT IDENTITY
# ══════════════════════════════════════════════════════════════════════════
EXPERIMENT = {
    "model_id"  : "meta-llama/llama-3.1-8b",
    "language"  : "indonesia",
    "width"     : "32k",
}

# ══════════════════════════════════════════════════════════════════════════
# LAYER-EMOTION MAP (from probing experiment)
# Set to None for single-layer mode.
# ══════════════════════════════════════════════════════════════════════════
MODEL_EMOTION_LAYER = {
    "google/gemma-2-2b": {
        1  : ["disgust"],
        6  : ["sadness"],
        7  : ["anger", "neutral"],
        8  : ["surprise"],
        11 : ["joy"],
        19 : ["fear"],
    },
    "google/gemma-2-9b-it": {
        9  : ["anger", "surprise", "neutral"],
        20 : ["fear", "joy", "sadness"],
        31 : ["disgust"],
    },
    "google/gemma-2-9b": {
        3  : ["anger"],
        8  : ["surprise"],
        11 : ["joy"],
        13 : ["fear"],
        14 : ["sadness"],
        18 : ["neutral"],
        35 : ["disgust"],
    },
     "meta-llama/llama-3.1-8b": {
        0  : ["neutral"],
        4  : ["fear"],
        10 : ["anger", "disgust"],
        15 : ["surprise"],
        24 : ["joy"],
        26 : ["sadness"],
    },
}

# ══════════════════════════════════════════════════════════════════════════
# AUTO-RESOLVE
# ══════════════════════════════════════════════════════════════════════════
def _model_slug(mid): return mid.split("/")[-1].lower()

MODEL_SLUG = _model_slug(EXPERIMENT["model_id"])
LANG_SLUG  = EXPERIMENT["language"].lower().replace(" ", "_")
WIDTH_SLUG = EXPERIMENT["width"].lower()

# ══════════════════════════════════════════════════════════════════════════
# SAE REGISTRY — model → width → config
#
# For each model+width combination:
#   sae_release : the HuggingFace release name
#   sae_id_fmt  : format string for sae_id, {layer} and {l0} are substituted
#   default_l0  : default average_l0 value (pick closest to ~88 for consistency)
#   layer_l0    : per-layer L0 overrides (when different layers have different L0s)
#   note        : any caveats
# ══════════════════════════════════════════════════════════════════════════
SAE_EXPLANATIONS_DIR = f"/content/drive/MyDrive/sae_features_jsonl/{MODEL_SLUG}/{WIDTH_SLUG}"

SAE_REGISTRY = {
    "google/gemma-2-2b": {
        "16k": {
            "sae_release" : "gemma-scope-2b-pt-res-canonical",
            "sae_id_fmt"  : "layer_{layer}/width_16k/canonical",
            "default_l0"  : None,    # canonical = no L0 choice
            "layer_l0"    : {},
            "note"        : "Canonical pick (L0 ~100). Available at all 26 layers.",
        },
        "65k": {
            "sae_release" : "gemma-scope-2b-pt-res",
            "sae_id_fmt"  : "layer_{layer}/width_65k/average_l0_{l0}",
            "default_l0"  : 107,      # check repo for exact values per layer
            "layer_l0"    : {
                              1: 121, 6: 107, 7: 107, 8: 111, 11: 70, 19: 115,
            },
            "note"        : "Available at all 26 layers. Check HF repo for L0 values.",
        },

        "default_layer" : 7,
    },

    "google/gemma-2-9b": {
        "16k": {
            "sae_release" : "gemma-scope-9b-pt-res-canonical",
            "sae_id_fmt"  : "layer_{layer}/width_16k/canonical",
            "default_l0"  : None,
            "layer_l0"    : {},
            "note"        : "Canonical pick. Available at all 42 layers.",
        },

        "131k": {
            "sae_release" : "gemma-scope-9b-pt-res",
            "sae_id_fmt"  : "layer_{layer}/width_131k/average_l0_{l0}",
            "default_l0"  : 121,
            "layer_l0"    : {3: 103, 8: 129, 11: 88, 13: 99, 14: 105, 18: 113, 35: 94,},
            "note"        : "Available at subset of layers. Check HF repo.",
        },
        "default_layer" : 16,
    },

    "google/gemma-2-9b-it": {
        "16k": {
            "sae_release" : "gemma-scope-9b-it-res",
            "sae_id_fmt"  : "layer_{layer}/width_16k/average_l0_{l0}",
            "default_l0"  : 91,
            "layer_l0"    : {
                9  : 88,
                20 : 91,
                31 : 76,
            },
            "note"        : "IT-specific SAEs. Only layers 9, 20, 31 available.",
        },

        "131k": {
            # IT model has no 131k SAEs — fall back to base model SAEs
            "sae_release" : "gemma-scope-9b-it-res",
            "sae_id_fmt"  : "layer_{layer}/width_131k/average_l0_{l0}",
            "default_l0"  : 81,
            "layer_l0"    : {9: 121, 20: 81, 31: 109,},
            "note"        : "Using BASE model SAEs (no IT 131k). Transfers well per Google's report.",
        },
        "default_layer" : 20,
    },
     "meta-llama/llama-3.1-8b": {
        "32k": {
            "sae_release" : "llama_scope_lxr_8x",
            "sae_id_fmt"  : "l{layer}r_8x",
            "default_l0"  : None,
            "layer_l0"    : {},
            "note"        : "Canonical pick. Available at all layers.",
        },
        "131k": {
            "sae_release" : "llama_scope_lxr_32x",
            "sae_id_fmt"  : "l{layer}r_32x",
            "default_l0"  : None,
            "layer_l0"    : {},
            "note"        : "Canonical pick. Available at all layers.",
        },
        "default_layer" : 20,
    },
}

# ══════════════════════════════════════════════════════════════════════════
# DATASET REGISTRY
# ══════════════════════════════════════════════════════════════════════════
DATASET_REGISTRY = {
    "indonesia": {
        "hf_dataset_id"   : "brighter-dataset/BRIGHTER-emotion-categories",
        "hf_subset"       : "ind",
        "text_column"     : "text",
        "emotion_classes" : ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"],
    },
}

# ══════════════════════════════════════════════════════════════════════════
# SP-1 PARAMETERS
# ══════════════════════════════════════════════════════════════════════════
SP1_CONFIG = {
    "encoder_model"     : "all-MiniLM-L6-v2",
    "epsilon"           : 0.30,
    "max_train_samples" : 500,
    "max_eval_samples"  : 100,
}

# ══════════════════════════════════════════════════════════════════════════
# RESOLVE SAE CONFIG
# ══════════════════════════════════════════════════════════════════════════
model_id = EXPERIMENT["model_id"]
width    = EXPERIMENT["width"]

model_registry = SAE_REGISTRY[model_id]
width_config   = model_registry[width]
data_config    = DATASET_REGISTRY[EXPERIMENT["language"]]

sae_release = width_config["sae_release"]

def get_sae_id(layer):
    """Resolve the full SAE ID for a given layer, model, and width."""
    fmt = width_config["sae_id_fmt"]

    # Canonical format has no L0 — just substitute layer
    if "{l0}" not in fmt:
        return fmt.format(layer=layer)

    # Resolve L0: per-layer override first, then default
    l0 = width_config["layer_l0"].get(layer, width_config["default_l0"])
    if l0 is None:
        raise ValueError(f"No L0 value for layer {layer}. "
                         f"Check HF repo and add to layer_l0 dict.")
    return fmt.format(layer=layer, l0=l0)


# ══════════════════════════════════════════════════════════════════════════
# RESOLVE LAYERS AND PATHS
# ══════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(f"/content/drive/MyDrive/sae_outputs/{MODEL_SLUG}/{WIDTH_SLUG}/{LANG_SLUG}")

LAYER_EMOTION_MAP = None #MODEL_EMOTION_LAYER.get(model_id)   # Change to None if you want to use single-default layer

if LAYER_EMOTION_MAP is not None:
    MODE = "multi-layer"
    UNIQUE_LAYERS = sorted(LAYER_EMOTION_MAP.keys())
else:
    MODE = "single-layer"
    UNIQUE_LAYERS = [model_registry.get("default_layer", 0)]

# Build paths per layer
LAYER_PATHS = {}
for layer in UNIQUE_LAYERS:
    layer_dir = BASE_DIR / "semantic-based" / "sp1_semantic" / f"layer_{layer}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    # Resolve explanations file path
    overrides = width_config.get("explanations_override", {})
    if layer in overrides:
        expl_path = Path(overrides[layer])
    else:
        expl_path = Path(f"{SAE_EXPLANATIONS_DIR}/layer_{layer}_explanations.jsonl")

    # Resolve SAE ID for this layer
    sae_id = get_sae_id(layer)

    LAYER_PATHS[layer] = {
        "dir"          : layer_dir,
        "f_cand"       : layer_dir / "f_cand.json",
        "all_scores"   : layer_dir / "all_scores.csv",
        "config"       : layer_dir / "config.json",
        "explanations" : expl_path,
        "sae_release"  : sae_release,
        "sae_id"       : sae_id,
    }

(BASE_DIR / "sp1").mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print(f"  Mode     : {MODE}")
print(f"  Model    : {model_id}")
print(f"  Width    : {width}")
print(f"  Language : {EXPERIMENT['language']}")
print(f"  Dataset  : {data_config['hf_dataset_id']} ({data_config.get('hf_subset','')})")
print(f"  Epsilon  : {SP1_CONFIG['epsilon']}")
print(f"  Encoder  : {SP1_CONFIG['encoder_model']}")
print(f"  SAE rel  : {sae_release}")
print(f"  Base dir : {BASE_DIR}")
print(f"  Layers   : {UNIQUE_LAYERS}")
if width_config.get("note"):
    print(f"  Note     : {width_config['note']}")
print("=" * 60)

if LAYER_EMOTION_MAP:
    print(f"\nLayer-Emotion mapping:")
    for layer, emos in LAYER_EMOTION_MAP.items():
        print(f"  Layer {layer:>2} -> {emos}")

print(f"\nPer-layer paths:")
for layer, paths in LAYER_PATHS.items():
    expl_exists = paths["explanations"].exists()
    print(f"  Layer {layer:>2}: sae_id = {paths['sae_id']}")
    print(f"           explanations: {paths['explanations']} {'OK' if expl_exists else 'MISSING'}")



# ── CELL 8 ──
def load_emotion_dataset(data_cfg, sp1_cfg):
    print(f"Loading dataset: {data_cfg['hf_dataset_id']} ...")
    load_kwargs = {"path": data_cfg["hf_dataset_id"]}
    if data_cfg["hf_subset"]: load_kwargs["name"] = data_cfg["hf_subset"]
    ds = load_dataset(**load_kwargs)

    label_names = data_cfg["emotion_classes"]
    emotion_cols = [e for e in label_names if e != "neutral"]

    def to_df(split_name, max_samples):
        split = ds[split_name]
        if max_samples: split = split.select(range(min(max_samples, len(split))))
        df = split.to_pandas()
        df["neutral"] = (df[emotion_cols].sum(axis=1) == 0).astype(int)
        df["label_name"] = df.apply(
            lambda row: ", ".join([e for e in label_names if row.get(e, 0) == 1]) or "none", axis=1)
        df = df[["text"] + label_names + ["label_name"]]
        print(f"    {split_name}: {len(df)} samples")
        return df

    if "validation" in ds and "dev" not in ds: ds["dev"] = ds["validation"]
    available = set(ds.keys())
    if "train" in available and "test" in available: t, e = "train", "test"
    elif "train" in available and "dev" in available: t, e = "train", "dev"
    else: t, e = "dev", "test"

    train_df = to_df(t, sp1_cfg["max_train_samples"])
    eval_df  = to_df(e, sp1_cfg["max_eval_samples"])
    return train_df, eval_df, label_names

train_df, eval_df, LABEL_NAMES = load_emotion_dataset(data_config, SP1_CONFIG)

display(train_df.head(5))
print(f"\nPer-emotion (train):")
display(train_df[LABEL_NAMES].sum().to_frame("count"))

# ── CELL 10 ──
def build_task_prompt(language, emotion_labels):
    emotions_str = ", ".join(emotion_labels)
    return (
        f"Detecting and recognising human emotions expressed in {language} text. "
        f"The target emotions are: {emotions_str}. "
        f"Cultural norms, social context, idiomatic expressions, politeness strategies, "
        f"indirect speech, and community-specific emotional displays in {language} "
        f"are important for correctly interpreting emotional cues."
    )

TASK_PROMPT = build_task_prompt(EXPERIMENT["language"], LABEL_NAMES)
print("Task prompt:")
print("-" * 72)
print(TASK_PROMPT)
print("-" * 72)

# ── CELL 12 ──
print(f"Loading encoder: {SP1_CONFIG['encoder_model']} ...")
encoder = SentenceTransformer(SP1_CONFIG["encoder_model"])
print(f"  Embedding dimension: {encoder.get_sentence_embedding_dimension()}")

# Pre-encode task prompt (shared across all layers)
e_task = encoder.encode([TASK_PROMPT], normalize_embeddings=True)
print("Task prompt encoded.")

# ── CELL 14 ──
def load_sae_repo(explanations_path):
    path = Path(explanations_path)
    if not path.exists():
        raise FileNotFoundError(
            f"SAE explanations not found: {path}\n"
            f"Download from Neuronpedia S3 for this layer.")

    with open(path) as f:
        first_char = f.read(1); f.seek(0)
        if first_char == "[":
            repo = json.load(f)
        else:
            repo = [json.loads(line) for line in f if line.strip()]

    for feat in repo:
        if "index" not in feat:
            feat["index"] = int(feat["id"].split(":")[-1])
        else:
            feat["index"] = int(feat["index"])
        feat["description"] = feat.get("description") or feat.get("explanationText", "")

    total = len(repo)
    repo = [f for f in repo if f["description"].strip()]
    print(f"    Total: {total}, with descriptions: {len(repo)}, empty: {total - len(repo)}")
    return repo

print("SAE loader ready.")

# ── CELL 16 ──
def candidate_retrieval(repo, e_task, encoder, epsilon):
    descriptions = [feat["description"] for feat in repo]

    print("    Encoding feature descriptions ...")
    e_feats = encoder.encode(descriptions, normalize_embeddings=True,
                             show_progress_bar=True)

    # Cosine similarity (both L2-normalised)
    all_scores = (e_feats @ e_task.T).squeeze()

    result_df = pd.DataFrame(repo).copy()
    result_df["similarity"] = all_scores
    result_df = result_df.sort_values("similarity", ascending=False)

    f_cand_df = result_df[result_df["similarity"] >= epsilon].reset_index(drop=True)

    print(f"    |F| = {len(repo)}, |F_cand| = {len(f_cand_df)}, "
          f"reduction = {1 - len(f_cand_df)/len(repo):.1%}")

    return f_cand_df, all_scores, result_df

print("Candidate retrieval function ready.")

# ── CELL 18 ──
ALL_RESULTS = {}    # {layer: {"f_cand_df": ..., "full_df": ..., "repo_size": ...}}

print(f"Processing {len(UNIQUE_LAYERS)} layer(s): {UNIQUE_LAYERS}\n")
print("=" * 60)

for layer in UNIQUE_LAYERS:
    paths = LAYER_PATHS[layer]
    emotions = LAYER_EMOTION_MAP.get(layer, LABEL_NAMES) if LAYER_EMOTION_MAP else LABEL_NAMES

    print(f"\n── Layer {layer} ──")
    print(f"  Emotions: {emotions}")
    print(f"  SAE: {paths['sae_id']}")
    print(f"  Explanations: {paths['explanations']}")

    # Load SAE repo for this layer
    repo = load_sae_repo(paths["explanations"])

    # Run candidate retrieval
    f_cand_df, all_scores, full_df = candidate_retrieval(
        repo, e_task, encoder, SP1_CONFIG["epsilon"]
    )

    ALL_RESULTS[layer] = {
        "f_cand_df"  : f_cand_df,
        "full_df"    : full_df,
        "repo_size"  : len(repo),
        "all_scores" : all_scores,
        "emotions"   : emotions,
    }

print("\n" + "=" * 60)
print("All layers processed.")

# ── CELL 20 ──
print("== Candidate Summary ==\n")

summary_rows = []
for layer in UNIQUE_LAYERS:
    r = ALL_RESULTS[layer]
    summary_rows.append({
        "Layer"     : layer,
        "Emotions"  : ", ".join(r["emotions"]),
        "|F|"       : r["repo_size"],
        "|F_cand|"  : len(r["f_cand_df"]),
        "Reduction" : f"{1 - len(r['f_cand_df'])/r['repo_size']:.1%}",
        "Top sim"   : f"{r['f_cand_df']['similarity'].max():.4f}" if len(r["f_cand_df"]) > 0 else "N/A",
    })

display(pd.DataFrame(summary_rows))

# Show top 10 candidates per layer
for layer in UNIQUE_LAYERS:
    r = ALL_RESULTS[layer]
    print(f"\n── Layer {layer}: Top 10 candidates ──")
    if len(r["f_cand_df"]) > 0:
        display(
            r["f_cand_df"].head(10)[["index", "similarity", "description"]]
            .style.format({"similarity": "{:.4f}"})
        )
    else:
        print("  No candidates above threshold!")

# ── CELL 22 ──
print("== Threshold Sensitivity ==\n")
thresholds = np.arange(0.05, 0.85, 0.05)

for layer in UNIQUE_LAYERS:
    scores = ALL_RESULTS[layer]["all_scores"]
    total  = ALL_RESULTS[layer]["repo_size"]
    print(f"  Layer {layer}:")
    for t in thresholds:
        n = int((scores >= t).sum())
        mark = " <--" if abs(t - SP1_CONFIG["epsilon"]) < 0.001 else ""
        print(f"    eps={t:.2f}  |F_cand|={n:>5}  ({100*n/total:.1f}%){mark}")
    print()

# ── CELL 24 ──
print("== Saving Outputs ==\n")

for layer in UNIQUE_LAYERS:
    paths = LAYER_PATHS[layer]
    r = ALL_RESULTS[layer]

    # 1. f_cand.json
    records = r["f_cand_df"].to_dict(orient="records")
    with open(paths["f_cand"], "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"  Layer {layer}: {len(records)} candidates -> {paths['f_cand']}")

    # 2. all_scores.csv
    r["full_df"].to_csv(paths["all_scores"], index=False)

    # 3. Per-layer config
    layer_cfg = {
        "layer"          : layer,
        "sae_id"         : paths["sae_id"],
        "emotions"       : r["emotions"],
        "repo_size"      : r["repo_size"],
        "n_candidates"   : len(r["f_cand_df"]),
        "epsilon"        : SP1_CONFIG["epsilon"],
        "encoder_model"  : SP1_CONFIG["encoder_model"],
    }
    with open(paths["config"], "w") as f:
        json.dump(layer_cfg, f, indent=2)

# 4. Shared metadata
shared_meta = {
    "mode"             : MODE,
    "model_id"         : EXPERIMENT["model_id"],
    "language"         : EXPERIMENT["language"],
    "dataset"          : data_config["hf_dataset_id"],
    "dataset_subset"   : data_config.get("hf_subset"),
    "emotion_classes"  : LABEL_NAMES,
    "epsilon"          : SP1_CONFIG["epsilon"],
    "encoder_model"    : SP1_CONFIG["encoder_model"],
    "task_prompt"      : TASK_PROMPT,
    "layers_processed" : UNIQUE_LAYERS,
    "layer_emotion_map": LAYER_EMOTION_MAP,
}
with open(BASE_DIR / "semantic-based" / "sp1_semantic" / "sp1_metadata.json", "w") as f:
    json.dump(shared_meta, f, indent=2)
print(f"\n  Metadata -> {BASE_DIR / 'semantic-based' / 'sp1_semantic' / 'sp1_metadata.json'}")

# ── CELL 26 ──
print("\n== SP-1 Summary ==")
print(f"  Mode:     {MODE}")
print(f"  Model:    {EXPERIMENT['model_id']}")
print(f"  Language: {EXPERIMENT['language']}")
print(f"  Epsilon:  {SP1_CONFIG['epsilon']}")
print(f"  Layers:   {UNIQUE_LAYERS}")
print()

for layer in UNIQUE_LAYERS:
    r = ALL_RESULTS[layer]
    emotions = r["emotions"]
    print(f"  Layer {layer:>2}: |F_cand|={len(r['f_cand_df']):>4}  emotions={emotions}")

total_cand = sum(len(ALL_RESULTS[l]["f_cand_df"]) for l in UNIQUE_LAYERS)
print(f"\n  Total candidates across all layers: {total_cand}")

print(f"\nOutput directory:")
sp1_dir = BASE_DIR / "sp1"
for p in sorted(sp1_dir.rglob("*")):
    if p.is_file():
        print(f"    {p.relative_to(sp1_dir)}")

print(f"\nReady for SP-2.")
if MODE == "multi-layer":
    print(f"  SP-2 will load one f_cand.json per layer from:")
    for layer in UNIQUE_LAYERS:
        print(f"    Layer {layer:>2}: {LAYER_PATHS[layer]['f_cand']}")
else:
    print(f"  SP-2 will load: {LAYER_PATHS[UNIQUE_LAYERS[0]]['f_cand']}")


################################################################################
# FILE: sp1_activation_selection.ipynb
################################################################################

# ── CELL 3 ──
import json, time, numpy as np, pandas as pd, gc
from pathlib import Path
from collections import defaultdict
from google.colab import runtime

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sae_lens import SAE
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

import warnings; warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# ── CELL 5 ──
EXPERIMENT = {
    "model_id"   : "meta-llama/llama-3.1-8b",
    "language"   : "indonesia",
    "custom_sae" : False,
    "width"      : "131k",
}

MODEL_EMOTION_LAYER = {
    "google/gemma-2-2b": {
        1  : ["disgust"],
        6  : ["sadness"],
        7  : ["anger", "neutral"],
        8  : ["surprise"],
        11 : ["joy"],
        19 : ["fear"],
    },
    "google/gemma-2-9b-it": {
        9  : ["anger", "surprise", "neutral"],
        20 : ["fear", "joy", "sadness"],
        31 : ["disgust"],
    },
    "google/gemma-2-9b": {
        3  : ["anger"],
        8  : ["surprise"],
        11 : ["joy"],
        13 : ["fear"],
        14 : ["sadness"],
        18 : ["neutral"],
        35 : ["disgust"],
    },
    "meta-llama/llama-3.1-8b": {
        0  : ["neutral"],
        4  : ["fear"],
        10 : ["anger", "disgust"],
        15 : ["surprise"],
        24 : ["joy"],
        26 : ["sadness"],
    },
}

SAE_REGISTRY = {
    "google/gemma-2-2b": {
        "16k": {
            "sae_release" : "gemma-scope-2b-pt-res-canonical",
            "sae_id_fmt"  : "layer_{layer}/width_16k/canonical",
            "default_l0"  : None,
            "layer_l0"    : {},
        },
        "65k": {
            "sae_release" : "gemma-scope-2b-pt-res",
            "sae_id_fmt"  : "layer_{layer}/width_65k/average_l0_{l0}",
            "default_l0"  : 107,
            "layer_l0"    : {1: 121, 6: 107, 7: 107, 8: 111, 11: 70, 19: 115},
        },
        "default_layer" : 7,
    },
    "google/gemma-2-9b": {
        "16k": {
            "sae_release" : "gemma-scope-9b-pt-res-canonical",
            "sae_id_fmt"  : "layer_{layer}/width_16k/canonical",
            "default_l0"  : None,
            "layer_l0"    : {},
        },
        "131k": {
            "sae_release" : "gemma-scope-9b-pt-res",
            "sae_id_fmt"  : "layer_{layer}/width_131k/average_l0_{l0}",
            "default_l0"  : 121,
            "layer_l0"    : {3: 103, 8: 129, 11: 88, 13: 99, 14: 105, 18: 113, 35: 94},
        },
        "default_layer" : 16,
    },
    "meta-llama/llama-3.1-8b": {
        "32k": {
            "sae_release" : "llama_scope_lxr_8x",
            "sae_id_fmt"  : "l{layer}r_8x",
            "default_l0"  : None,
            "layer_l0"    : {},
        },
        "131k": {
            "sae_release" : "llama_scope_lxr_32x",
            "sae_id_fmt"  : "l{layer}r_32x",
            "default_l0"  : None,
            "layer_l0"    : {},
        },
        "default_layer" : 20,
    },
        "google/gemma-2-9b-it": {
        "16k": {
            "sae_release" : "gemma-scope-9b-it-res",
            "sae_id_fmt"  : "layer_{layer}/width_16k/average_l0_{l0}",
            "default_l0"  : 91,
            "layer_l0"    : {9: 88, 20: 91, 31: 76},
        },
        "131k": {
            "sae_release" : "gemma-scope-9b-it-res",
            "sae_id_fmt"  : "layer_{layer}/width_131k/average_l0_{l0}",
            "default_l0"  : 81,
            "layer_l0"    : {9: 121, 20: 81, 31: 109},
        },
        "default_layer" : 20,
    },
}

DATASET_REGISTRY = {
    "indonesia": {
        "hf_dataset_id"   : "brighter-dataset/BRIGHTER-emotion-categories",
        "hf_subset"       : "ind",
        "text_column"     : "text",
        "emotion_classes" : ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"],
    },
}

ACTSEL_CONFIG = {
    "max_texts"           : 500,
    "max_seq_len"         : 128,
    "batch_size"          : 4,
    "pool_method"         : "max",
    "binarize_threshold"  : 0.0,
    "classifier_C"        : 0.1,       # L1 regularization strength (lower = more sparse)
    "min_weight"          : 1e-4,      # minimum abs weight to select a feature
    "min_features"        : 10,
    "use_4bit"            : False,
}

# ── Auto-resolve ─────────────────────────────────────────────────────────
def _slug(mid): return mid.split("/")[-1].lower()

MODEL_SLUG = _slug(EXPERIMENT["model_id"])
LANG_SLUG  = EXPERIMENT["language"].lower().replace(" ", "_")
WIDTH_SLUG = EXPERIMENT["width"].lower()
BASE_DIR   = Path(f"/content/drive/MyDrive/sae_outputs/{MODEL_SLUG}/{WIDTH_SLUG}/{LANG_SLUG}")

model_registry = SAE_REGISTRY[EXPERIMENT["model_id"]]
width_config   = model_registry[EXPERIMENT["width"]]
data_config    = DATASET_REGISTRY[EXPERIMENT["language"]]
EMOTION_CLASSES = data_config["emotion_classes"]

def get_sae_id(layer):
    fmt = width_config["sae_id_fmt"]
    if "{l0}" not in fmt:
        return fmt.format(layer=layer)
    l0 = width_config["layer_l0"].get(layer, width_config["default_l0"])
    if l0 is None:
        raise ValueError(f"No L0 value for layer {layer}.")
    return fmt.format(layer=layer, l0=l0)

LAYER_EMOTION_MAP = None #MODEL_EMOTION_LAYER.get(EXPERIMENT["model_id"]) #<--change this to None to activate single-layer mode
if LAYER_EMOTION_MAP is not None:
    UNIQUE_LAYERS = sorted(LAYER_EMOTION_MAP.keys())
else:
    dl = model_registry.get("default_layer", 0)
    UNIQUE_LAYERS = [dl]

CUST_SAE_PATHS = {layer: BASE_DIR / "cust_sae" / f"layer_{layer}" / "W_dec.pt" for layer in UNIQUE_LAYERS}
OUT_PATHS = {}
for layer in UNIQUE_LAYERS:
    out_dir = BASE_DIR / "classifier-based" / "sp1_classifier" / f"layer_{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    OUT_PATHS[layer] = {
        "dir"     : out_dir,
        "f_cand"  : out_dir / "f_cand.json",
        "stats"   : out_dir / "classifier_stats.csv",
        "config"  : out_dir / "config.json",
    }

print("=" * 60)
print(f"  Model:      {EXPERIMENT['model_id']}")
print(f"  Width:      {EXPERIMENT['width']}")
print(f"  Custom SAE: {EXPERIMENT['custom_sae']}")
print(f"  Layers:     {UNIQUE_LAYERS}")
print(f"  Pool:       {ACTSEL_CONFIG['pool_method']}")
print(f"  Classifier C: {ACTSEL_CONFIG['classifier_C']}")
print(f"  Max texts:  {ACTSEL_CONFIG['max_texts']}")
print(f"  Output:     {BASE_DIR / 'classifier-based' / 'sp1_classifier'}")
print("=" * 60)

# ── CELL 7 ──
# Emotional texts (BRIGHTER - Indonesian)
print("Loading BRIGHTER dataset with emotion labels ...")
cfg = data_config
lk = {"path": cfg["hf_dataset_id"]}
if cfg["hf_subset"]: lk["name"] = cfg["hf_subset"]
ds = load_dataset(**lk)

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

emotion_cols = [e for e in EMOTION_CLASSES if e != "neutral"]

# Build DataFrame with texts and multi-hot labels
all_texts = []
all_labels = []    # (n_texts, n_emotions) multi-hot

for example in ds[use_split]:
    if len(all_texts) >= ACTSEL_CONFIG["max_texts"]: break
    t = example[cfg["text_column"]]
    if not t or len(t.strip()) <= 20: continue

    label_row = []
    for emo in EMOTION_CLASSES:
        if emo == "neutral":
            # neutral = no other emotion present
            has_any = any(example.get(e, 0) == 1 for e in emotion_cols)
            label_row.append(0 if has_any else 1)
        else:
            label_row.append(int(example.get(emo, 0)))
    all_texts.append(t.strip())
    all_labels.append(label_row)

all_labels = np.array(all_labels)    # (n_texts, n_emotions)

print(f"  Loaded {len(all_texts)} texts with multi-hot labels")
print(f"  Label shape: {all_labels.shape}")
print(f"  Per-emotion counts:")
for i, emo in enumerate(EMOTION_CLASSES):
    print(f"    {emo:<10} {all_labels[:, i].sum():>4} / {len(all_texts)}")
print(f"\n  Sample: {all_texts[0][:80]}...")

# ── CELL 9 ──
tokenizer = AutoTokenizer.from_pretrained(EXPERIMENT["model_id"])
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

load_kw = {"pretrained_model_name_or_path": EXPERIMENT["model_id"],
           "device_map": "auto", "attn_implementation": "eager"}
if ACTSEL_CONFIG["use_4bit"]:
    load_kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
else:
    load_kw["torch_dtype"] = torch.float16

model = AutoModelForCausalLM.from_pretrained(**load_kw)
model.eval()
for p in model.parameters(): p.requires_grad = False
n_layers = model.config.num_hidden_layers
d_model  = model.config.hidden_size
print(f"Model loaded: d={d_model}, layers={n_layers}")

# ── CELL 11 ──
def extract_pooled_activations(texts, model, tokenizer, sae_encoder, sae_b_dec,
                               layer_j, config):
    """
    Returns:
        pooled: (n_texts, d_sae) float — pooled activation per text
    """
    pool_method = config["pool_method"]
    batch_size  = config["batch_size"]
    max_seq     = config["max_seq_len"]

    all_pooled = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
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
            z_for_max[mask.squeeze(-1) == 0] = -float('inf')
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

print("Extraction function ready.")

# ── CELL 13 ──
ALL_RESULTS = {}

for layer in UNIQUE_LAYERS:
    print(f"\n{'='*60}")
    print(f"  Layer {layer}")
    print(f"{'='*60}")

    # ── Load SAE encoder for this layer ──
    if EXPERIMENT["custom_sae"]:
        wdec_path = CUST_SAE_PATHS[layer]
        print(f"  Loading custom SAE from {wdec_path} ...")
        W_dec_raw = torch.load(wdec_path, weights_only=True).to(DEVICE)
        if W_dec_raw.shape[0] == d_model:
            W_dec_T = W_dec_raw.T
        else:
            W_dec_T = W_dec_raw
        d_sae = W_dec_T.shape[0]
        sae_encoder = torch.nn.Linear(d_model, d_sae, bias=True).to(DEVICE)
        with torch.no_grad():
            sae_encoder.weight.data = W_dec_T.clone().float().T
            sae_encoder.bias.data.zero_()
        sae_b_dec = torch.zeros(d_model, device=DEVICE)
        print(f"  Custom SAE: d_sae={d_sae}")
        del W_dec_raw, W_dec_T
    else:
        sae_id = get_sae_id(layer)
        print(f"  Loading SAE: {sae_id} ...")
        sae_obj, _, _ = SAE.from_pretrained(
            release=width_config["sae_release"], sae_id=sae_id, device=DEVICE)
        d_sae = sae_obj.cfg.d_sae
        sae_encoder = torch.nn.Linear(d_model, d_sae, bias=True).to(DEVICE)
        with torch.no_grad():
            sae_encoder.weight.data = sae_obj.W_enc.detach().clone().float().T
            if hasattr(sae_obj, 'b_enc'):
                sae_encoder.bias.data = sae_obj.b_enc.detach().clone().float()
            else:
                sae_encoder.bias.data.zero_()
        sae_b_dec = sae_obj.b_dec.detach().clone().float().to(DEVICE)
        del sae_obj
    torch.cuda.empty_cache()
    print(f"  d_sae = {d_sae}")

    # ── Extract pooled activations ──
    print(f"  Extracting pooled activations ({len(all_texts)} texts) ...")
    pooled = extract_pooled_activations(
        all_texts, model, tokenizer, sae_encoder, sae_b_dec, layer, ACTSEL_CONFIG)
    print(f"    pooled: {pooled.shape}")

    # ── Train multi-label classifier ──
    print(f"  Training OneVsRest L1 classifier (C={ACTSEL_CONFIG['classifier_C']}) ...")
    scaler = StandardScaler()
    X = scaler.fit_transform(pooled)
    y = all_labels    # (n_texts, n_emotions) multi-hot

    clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=500, C=ACTSEL_CONFIG["classifier_C"],
            penalty="l1", solver="saga", random_state=42
        )
    )
    clf.fit(X, y)

    # ── Extract feature importance from classifier weights ──
    # clf.estimators_ is a list of n_emotions classifiers
    # Each has .coef_ of shape (1, d_sae)
    per_emo_weights = {}    # {emotion: abs weights array}
    all_weights = np.zeros(d_sae)

    print(f"  Per-emotion non-zero features:")
    for i, emo in enumerate(EMOTION_CLASSES):
        w = np.abs(clf.estimators_[i].coef_[0])    # (d_sae,)
        per_emo_weights[emo] = w
        n_nonzero = (w > ACTSEL_CONFIG["min_weight"]).sum()
        all_weights = np.maximum(all_weights, w)    # max across emotions
        print(f"    {emo:<10} non-zero: {n_nonzero:>4}  max_w: {w.max():.4f}")

    # ── Select features with non-zero weight in ANY emotion ──
    min_w = ACTSEL_CONFIG["min_weight"]
    min_features = ACTSEL_CONFIG["min_features"]

    # Rank by max weight across emotions (descending)
    ranked = np.argsort(-all_weights)
    selected = [int(idx) for idx in ranked if all_weights[idx] > min_w]
    n_above = len(selected)

    # Fallback: if too few, take top by rank
    if len(selected) < min_features:
        print(f"  Only {len(selected)} above min_weight={min_w}, "
              f"taking top {min_features} by rank")
        selected = [int(idx) for idx in ranked[:min_features]]

    print(f"  Selected: {len(selected)} features ({n_above} with non-zero weight)")

    # ── Report which emotions selected which features ──
    print(f"\n  Top 5 features:")
    for rank, idx in enumerate(selected[:5]):
        # Which emotions use this feature?
        emo_contribs = []
        for emo in EMOTION_CLASSES:
            w = per_emo_weights[emo][idx]
            if w > min_w:
                emo_contribs.append(f"{emo[:3]}={w:.3f}")
        emo_str = ", ".join(emo_contribs) if emo_contribs else "below threshold"
        print(f"    #{rank+1} feature {idx}  max_w={all_weights[idx]:.4f}  [{emo_str}]")

    # ── Build f_cand compatible with SP-2 ──
    candidates = []
    for rank, idx in enumerate(selected):
        emo_weights_dict = {emo: float(per_emo_weights[emo][idx]) for emo in EMOTION_CLASSES}
        candidates.append({
            "index"         : idx,
            "rank"          : rank,
            "max_weight"    : float(all_weights[idx]),
            "emo_weights"   : emo_weights_dict,
            "n_emo_nonzero" : sum(1 for v in emo_weights_dict.values() if v > min_w),
            "description"   : f"classifier-selected (max_w={all_weights[idx]:.3f}, "
                             f"n_emo={sum(1 for v in emo_weights_dict.values() if v > min_w)})",
        })

    # ── Build full stats ──
    selected_set = set(selected)
    stats_df = pd.DataFrame({
        "index"      : list(range(d_sae)),
        "max_weight" : all_weights,
        **{f"w_{emo}": per_emo_weights[emo] for emo in EMOTION_CLASSES},
        "selected"   : [i in selected_set for i in range(d_sae)],
    })

    ALL_RESULTS[layer] = {
        "candidates"      : candidates,
        "selected"        : selected,
        "stats_df"        : stats_df,
        "d_sae"           : d_sae,
        "all_weights"     : all_weights,
        "per_emo_weights" : per_emo_weights,
        "n_above"         : n_above,
    }

    del sae_encoder, sae_b_dec, pooled, X, clf
    torch.cuda.empty_cache(); gc.collect()

print("\n\nAll layers processed.")

print("\n== Cross-Layer Summary ==\n")
total = 0
for layer in UNIQUE_LAYERS:
    r = ALL_RESULTS[layer]
    total += len(r["selected"])
    emotions = LAYER_EMOTION_MAP.get(layer, ["all"]) if LAYER_EMOTION_MAP else ["all"]
    print(f"  Layer {layer:>2}: {len(r['selected']):>4} selected "
          f"({r['n_above']} non-zero weight)  emotions={emotions}")
print(f"\n  Total features across all layers: {total}")


# ── CELL 16 ──
print("== Classifier-Weight Feature Selection Summary ==\n")

summary_rows = []
for layer in UNIQUE_LAYERS:
    r = ALL_RESULTS[layer]
    summary_rows.append({
        "Layer": layer,
        "d_sae": r["d_sae"],
        "|F_cand|": len(r["selected"]),
        "Non-zero": r["n_above"],
        "Top weight": f"{r['all_weights'].max():.4f}",
        "Emotions": LAYER_EMOTION_MAP.get(layer, ["all"]) if LAYER_EMOTION_MAP else ["all"],
    })
display(pd.DataFrame(summary_rows))

for layer in UNIQUE_LAYERS:
    r = ALL_RESULTS[layer]
    print(f"\n-- Layer {layer}: Top 10 features --")
    for feat in r["candidates"][:10]:
        ew = feat["emo_weights"]
        top_emos = sorted(ew.items(), key=lambda x: x[1], reverse=True)[:3]
        emo_str = ", ".join([f"{e[:3]}={w:.3f}" for e, w in top_emos if w > ACTSEL_CONFIG["min_weight"]])
        print(f"  [{feat['index']:>5}]  max_w={feat['max_weight']:.4f}  "
              f"n_emo={feat['n_emo_nonzero']}  [{emo_str}]")

# ── CELL 18 ──
try:
    import matplotlib.pyplot as plt
    n = len(UNIQUE_LAYERS)

    # Weight distribution per layer
    fig, axes = plt.subplots(1, n, figsize=(6*n, 4))
    if n == 1: axes = [axes]
    for i, layer in enumerate(UNIQUE_LAYERS):
        r = ALL_RESULTS[layer]
        w = r["all_weights"]
        axes[i].hist(w[w > 0], bins=60, color="#7F77DD", edgecolor="white", alpha=0.8)
        axes[i].axvline(ACTSEL_CONFIG["min_weight"], color="red",
                       linestyle="--", label=f"min_w={ACTSEL_CONFIG['min_weight']}")
        axes[i].set_title(f"L{layer}: classifier weight distribution")
        axes[i].set_xlabel("Max |weight| across emotions")
        axes[i].legend(fontsize=8)
    plt.suptitle("Feature Importance by Classifier Weight")
    plt.tight_layout(); plt.show()

    # Per-emotion heatmap of selected features
    for layer in UNIQUE_LAYERS:
        r = ALL_RESULTS[layer]
        sel = r["selected"][:50]    # top 50 for readability
        if len(sel) == 0: continue
        hm = np.array([[r["per_emo_weights"][emo][idx] for idx in sel] for emo in EMOTION_CLASSES])
        fig, ax = plt.subplots(figsize=(min(20, len(sel)*0.4), 4))
        im = ax.imshow(hm, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(len(EMOTION_CLASSES)))
        ax.set_yticklabels(EMOTION_CLASSES)
        ax.set_xlabel("Feature rank")
        ax.set_title(f"L{layer}: Per-emotion classifier weights (top {len(sel)} features)")
        plt.colorbar(im, label="|weight|")
        plt.tight_layout(); plt.show()
except ImportError:
    pass

# ── CELL 20 ──
print("Saving outputs ...\n")

for layer in UNIQUE_LAYERS:
    r = ALL_RESULTS[layer]
    p = OUT_PATHS[layer]

    with open(p["f_cand"], "w") as f:
        json.dump(r["candidates"], f, indent=2)
    print(f"  Layer {layer:>2}: {len(r['candidates'])} candidates -> {p['f_cand']}")

    r["stats_df"].to_csv(p["stats"], index=False)

    layer_cfg = {
        "layer"          : layer,
        "d_sae"          : r["d_sae"],
        "n_candidates"   : len(r["selected"]),
        "n_nonzero"      : r["n_above"],
        "pool_method"    : ACTSEL_CONFIG["pool_method"],
        "classifier_C"   : ACTSEL_CONFIG["classifier_C"],
        "min_weight"     : ACTSEL_CONFIG["min_weight"],
        "n_texts"        : len(all_texts),
        "method"         : "classifier-weight",
    }
    with open(p["config"], "w") as f:
        json.dump(layer_cfg, f, indent=2)

meta = {
    "model_id"         : EXPERIMENT["model_id"],
    "width"            : EXPERIMENT["width"],
    "custom_sae"       : EXPERIMENT["custom_sae"],
    "language"         : EXPERIMENT["language"],
    "layers"           : UNIQUE_LAYERS,
    "method"           : "classifier-weight",
    "data_source"      : data_config["hf_dataset_id"],
    "pool_method"      : ACTSEL_CONFIG["pool_method"],
    "classifier_C"     : ACTSEL_CONFIG["classifier_C"],
    "min_weight"       : ACTSEL_CONFIG["min_weight"],
}
meta_path = BASE_DIR / "classifier-based" / "sp1_classifier" / "metadata.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"\n  Metadata -> {meta_path}")

# ── CELL 22 ──
print("== SP-2 Integration ==\n")
print("To use classifier-selected candidates in SP-2, change SP1_PATHS:\n")
print("# In SP-2 config section:")
print('# SP1_PATHS = {layer: BASE_DIR / "classifier-based" / "sp1_classifier" / f"layer_{layer}" / "f_cand.json" ...}')
print()
for layer in UNIQUE_LAYERS:
    p = OUT_PATHS[layer]["f_cand"]
    n = len(ALL_RESULTS[layer]["selected"])
    print(f"  SP1_PATHS[{layer}] = Path('{p}')  # {n} candidates")
print()
print("Everything else in SP-2 stays the same.")
print("The f_cand.json format is identical to semantic SP-1 output.")

# ── CELL 24 ──
print("\n== Classifier-Weight Feature Selection Summary ==")
print(f"  Model:      {EXPERIMENT['model_id']}")
print(f"  Width:      {EXPERIMENT['width']}")
print(f"  Custom SAE: {EXPERIMENT['custom_sae']}")
print(f"  Pool:       {ACTSEL_CONFIG['pool_method']}")
print(f"  Classifier: L1 LogisticRegression, C={ACTSEL_CONFIG['classifier_C']}")
print(f"  Texts:      {len(all_texts)} (BRIGHTER)")

for layer in UNIQUE_LAYERS:
    r = ALL_RESULTS[layer]
    print(f"  Layer {layer:>2}: {len(r['selected'])} selected ({r['n_above']} non-zero)")

print(f"\n  Output: {BASE_DIR / 'sp1_activation'}")
print(f"\nReady for SP-2.")

del model
torch.cuda.empty_cache(); gc.collect()
print("Model freed.")


################################################################################
# FILE: sp2_coefficient_optimisation.ipynb
################################################################################

# ── CELL 3 ──
import json, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict

import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
from google.colab import runtime

from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import warnings; warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# ── CELL 5 ──
EXPERIMENT = {
    "model_id"   : "google/gemma-2-9b",
    "language"   : "indonesia",
    "custom_sae" : False,
    "width"      : "131k",
    "cand_selection" : "classifier-based", #semantic-based or classifier-based
}

# ── Layer-Emotion map (from probing). Set None for single-layer. ─────────
MODEL_EMOTION_LAYER = {
    "google/gemma-2-2b": {
        1  : ["disgust"],
        6  : ["sadness"],
        7  : ["anger", "neutral"],
        8  : ["surprise"],
        11 : ["joy"],
        19 : ["fear"],
    },
    "google/gemma-2-9b-it": {
        9  : ["anger", "surprise", "neutral"],
        20 : ["fear", "joy", "sadness"],
        31 : ["disgust"],
    },
    "google/gemma-2-9b": {
        3  : ["anger"],
        8  : ["surprise"],
        11 : ["joy"],
        13 : ["fear"],
        14 : ["sadness"],
        18 : ["neutral"],
        35 : ["disgust"],
    },
    "meta-llama/llama-3.1-8b": {
        0  : ["neutral"],
        4  : ["fear"],
        10 : ["anger", "disgust"],
        15 : ["surprise"],
        24 : ["joy"],
        26 : ["sadness"],
    },
}

SAE_REGISTRY = {
    "google/gemma-2-2b": {
        "16k": {
            "sae_release" : "gemma-scope-2b-pt-res-canonical",
            "sae_id_fmt"  : "layer_{layer}/width_16k/canonical",
            "default_l0"  : None,
            "layer_l0"    : {},
            "note"        : "Canonical pick (L0 ~100). Available at all 26 layers.",
        },
        "65k": {
            "sae_release" : "gemma-scope-2b-pt-res",
            "sae_id_fmt"  : "layer_{layer}/width_65k/average_l0_{l0}",
            "default_l0"  : 107,
            "layer_l0"    : {1: 121, 6: 107, 7: 107, 8: 111, 11: 70, 19: 115},
            "note"        : "Available at all 26 layers.",
        },
        "default_layer" : 7,
    },
    "google/gemma-2-9b": {
        "16k": {
            "sae_release" : "gemma-scope-9b-pt-res-canonical",
            "sae_id_fmt"  : "layer_{layer}/width_16k/canonical",
            "default_l0"  : None,
            "layer_l0"    : {},
            "note"        : "Canonical pick. Available at all 42 layers.",
        },
        "131k": {
            "sae_release" : "gemma-scope-9b-pt-res",
            "sae_id_fmt"  : "layer_{layer}/width_131k/average_l0_{l0}",
            "default_l0"  : 121,
            "layer_l0"    : {3: 103, 8: 129, 11: 88, 13: 99, 14: 105, 18: 113, 35: 94},
            "note"        : "Available at subset of layers.",
        },
        "default_layer" : 16,
    },
    "google/gemma-2-9b-it": {
        "16k": {
            "sae_release" : "gemma-scope-9b-it-res",
            "sae_id_fmt"  : "layer_{layer}/width_16k/average_l0_{l0}",
            "default_l0"  : 91,
            "layer_l0"    : {9: 88, 20: 91, 31: 76},
            "note"        : "IT-specific SAEs. Only layers 9, 20, 31.",
        },
        "131k": {
            "sae_release" : "gemma-scope-9b-it-res",
            "sae_id_fmt"  : "layer_{layer}/width_131k/average_l0_{l0}",
            "default_l0"  : 81,
            "layer_l0"    : {9: 121, 20: 81, 31: 109},
            "note"        : "Using BASE model SAEs. Transfers well per Google's report.",
        },
        "default_layer" : 20,
    },

    "meta-llama/llama-3.1-8b": {
        "32k": {
            "sae_release" : "llama_scope_lxr_8x",
            "sae_id_fmt"  : "l{layer}r_8x",
            "default_l0"  : None,
            "layer_l0"    : {},
            "note"        : "Canonical pick. Available at all layers.",
        },
        "131k": {
            "sae_release" : "llama_scope_lxr_32x",
            "sae_id_fmt"  : "l{layer}r_32x",
            "default_l0"  : None,
            "layer_l0"    : {},
            "note"        : "Canonical pick. Available at all layers.",
        },
        "default_layer" : 20,
    },
}

DATASET_REGISTRY = {
    "indonesia": {
        "hf_dataset_id"   : "brighter-dataset/BRIGHTER-emotion-categories",
        "hf_subset"       : "ind",
        "text_column"     : "text",
        "emotion_classes" : ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"],
    },
}

CONFIG = {
    "lr"               : 0.005,
    "lambda_l1"        : 0.0001,
    "tau"              : 0.01,
    "n_epochs"         : 50,
    "patience"         : 10,
    "batch_size"       : 4,
    "max_train_samples": 500,
    "max_eval_samples" : 100,
    "max_seq_len"      : 128,
}

# ══════════════════════════════════════════════════════════════════════════
# AUTO-RESOLVE
# ══════════════════════════════════════════════════════════════════════════
def _slug(mid): return mid.split("/")[-1].lower()

MODEL_SLUG  = _slug(EXPERIMENT["model_id"])
LANG_SLUG   = EXPERIMENT["language"].lower().replace(" ", "_")
WIDTH_SLUG  = EXPERIMENT["width"].lower()
BASE_DIR    = Path(f"/content/drive/MyDrive/sae_outputs/{MODEL_SLUG}/{WIDTH_SLUG}/{LANG_SLUG}")

model_registry = SAE_REGISTRY[EXPERIMENT["model_id"]]
width_config   = model_registry[EXPERIMENT["width"]]
data_config    = DATASET_REGISTRY[EXPERIMENT["language"]]

EMOTION_CLASSES = data_config["emotion_classes"]
EMOTION_TO_IDX  = {e: i for i, e in enumerate(EMOTION_CLASSES)}
LAYER_EMOTION_MAP = None #MODEL_EMOTION_LAYER.get(EXPERIMENT["model_id"]) # <--change this to None to use single-layer

# ── SAE ID resolver ──────────────────────────────────────────────────────
def get_sae_id(layer):
    """Resolve the full SAE ID for a given layer using the active width config."""
    fmt = width_config["sae_id_fmt"]
    if "{l0}" not in fmt:
        return fmt.format(layer=layer)
    l0 = width_config["layer_l0"].get(layer, width_config["default_l0"])
    if l0 is None:
        raise ValueError(f"No L0 value for layer {layer}. "
                         f"Check HF repo and add to layer_l0 dict.")
    return fmt.format(layer=layer, l0=l0)

# ── Mode & layer routing ─────────────────────────────────────────────────
if LAYER_EMOTION_MAP is not None:
    MODE = "multi-layer"
    UNIQUE_LAYERS = sorted(LAYER_EMOTION_MAP.keys())
    EMO_TO_LAYER = {}
    for layer, emos in LAYER_EMOTION_MAP.items():
        for e in emos:
            EMO_TO_LAYER[e] = layer
else:
    MODE = "single-layer"
    dl = model_registry.get("default_layer", 0)
    UNIQUE_LAYERS = [dl]
    EMO_TO_LAYER = {e: dl for e in EMOTION_CLASSES}
    LAYER_EMOTION_MAP = {dl: EMOTION_CLASSES}

# ── Output directory tag ─────────────────────────────────────────────────
if MODE == "multi-layer" and EXPERIMENT["custom_sae"]:
    SP2_TAG = "sp2_multilayer_custom_sae"
elif MODE == "multi-layer" and not EXPERIMENT["custom_sae"]:
    SP2_TAG = "sp2_multilayer"
else:
    SP2_TAG = f"sp2_layer{UNIQUE_LAYERS[0]}"

SP2_DIR = BASE_DIR / EXPERIMENT["cand_selection"] / SP2_TAG
SP2_DIR.mkdir(parents=True, exist_ok=True)

# ── Source paths per layer ───────────────────────────────────────────────
if EXPERIMENT["cand_selection"] == "classifier-based":
  SP1 = "classifier-based/sp1_classifier"

else:
  SP1 = "semantic-based/sp1_semantic"
SP1_PATHS      = {layer: BASE_DIR / SP1 / f"layer_{layer}" / "f_cand.json" for layer in UNIQUE_LAYERS}
CUST_SAE_PATHS = {layer: BASE_DIR / "cust_sae" / f"layer_{layer}" / "W_dec.pt" for layer in UNIQUE_LAYERS}
print(f"candidate paths: {SP1_PATHS}")

# ── Populate CONFIG ──────────────────────────────────────────────────────
CONFIG["model_id"]        = EXPERIMENT["model_id"]
CONFIG["language"]        = EXPERIMENT["language"]
CONFIG["emotion_classes"] = EMOTION_CLASSES
CONFIG["width"]           = EXPERIMENT["width"]
CONFIG["custom_sae"]      = EXPERIMENT["custom_sae"]
CONFIG["sae_release"]     = width_config["sae_release"]

# ── Print summary ────────────────────────────────────────────────────────
print("=" * 60)
print(f"  Mode       : {MODE}")
print(f"  Model      : {EXPERIMENT['model_id']}")
print(f"  Width      : {EXPERIMENT['width']}")
print(f"  Custom SAE : {EXPERIMENT['custom_sae']}")
print(f"  Language   : {EXPERIMENT['language']}")
print(f"  SAE release: {width_config['sae_release']}")
print(f"  Layers     : {UNIQUE_LAYERS}")
print(f"  Output     : {SP2_DIR}")
if width_config.get("note"):
    print(f"  Note       : {width_config['note']}")
print("=" * 60)

print(f"\nEmotion -> Layer routing:")
for e in EMOTION_CLASSES:
    print(f"  {e:<10} -> Layer {EMO_TO_LAYER[e]}")

print(f"\nSAE IDs per layer:")
for layer in UNIQUE_LAYERS:
    if EXPERIMENT["custom_sae"]:
        p = CUST_SAE_PATHS[layer]
        status = "OK" if p.exists() else "MISSING"
        print(f"  Layer {layer:>2}: [custom] {p}  {status}")
    else:
        sae_id = get_sae_id(layer)
        p = SP1_PATHS[layer]
        status = "OK" if p.exists() else "MISSING"
        print(f"  Layer {layer:>2}: {sae_id}  |  SP1: {status}")


# ── CELL 8 ──
def load_candidates(path):
    with open(path) as f:
        ch = f.read(1); f.seek(0)
        cands = json.load(f) if ch == "[" else [json.loads(l) for l in f if l.strip()]
    for feat in cands:
        if "index" not in feat: feat["index"] = int(feat["id"].split(":")[-1])
        else: feat["index"] = int(feat["index"])
    return cands

# Load candidates per layer
CANDS = {}     # {layer: list of feature dicts}
INDICES = {}   # {layer: list of int indices}

for layer in UNIQUE_LAYERS:
    cands = load_candidates(SP1_PATHS[layer])
    CANDS[layer] = cands
    INDICES[layer] = [f["index"] for f in cands]
    print(f"  Layer {layer:>2}: {len(cands)} candidates, max index = {max(INDICES[layer])}")

total_cand = sum(len(v) for v in CANDS.values())
print(f"\n  Total candidates across layers: {total_cand}")

# ── CELL 10 ──
def load_emotion_dataset(cfg):
    load_kwargs = {"path": cfg["hf_dataset_id"]}
    if cfg["hf_subset"]: load_kwargs["name"] = cfg["hf_subset"]
    ds = load_dataset(**load_kwargs)
    label_names = cfg["emotion_classes"]
    emotion_cols = [e for e in label_names if e != "neutral"]
    def to_df(split, mx):
        sp = ds[split]
        if mx: sp = sp.select(range(min(mx, len(sp))))
        df = sp.to_pandas()
        df["neutral"] = (df[emotion_cols].sum(axis=1) == 0).astype(int)
        rows = []
        for _, row in df.iterrows():
            for emo in label_names:
                rows.append({"text": row["text"], "emotion_query": emo, "label": int(row[emo])})
        out = pd.DataFrame(rows)
        print(f"    {split}: {len(df)} texts x {len(label_names)} = {len(out)} pairs")
        return out
    if "validation" in ds and "dev" not in ds: ds["dev"] = ds["validation"]
    av = set(ds.keys())
    if "train" in av and "test" in av: t, e = "train", "test"
    elif "train" in av and "dev" in av: t, e = "train", "dev"
    else: t, e = "dev", "test"
    return to_df(t, CONFIG["max_train_samples"]), to_df(e, CONFIG["max_eval_samples"]), label_names

train_df, eval_df, LABEL_NAMES = load_emotion_dataset(data_config)
display(train_df.head(14))
print(f"\nLabel balance:")
display(train_df.groupby("emotion_query")["label"].value_counts().unstack(fill_value=0))

# ── CELL 12 ──
tokenizer = AutoTokenizer.from_pretrained(EXPERIMENT["model_id"])
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    EXPERIMENT["model_id"], torch_dtype=torch.float16,
    device_map="auto", attn_implementation="eager",
)
model.eval(); model.to(DEVICE)
for p in model.parameters(): p.requires_grad = False

d_model = model.config.hidden_size
n_layers = model.config.num_hidden_layers
print(f"d_model: {d_model}, n_layers: {n_layers}")

yes_token_id = tokenizer.encode("yes", add_special_tokens=False)[0]
no_token_id  = tokenizer.encode("no",  add_special_tokens=False)[0]
print(f"yes={yes_token_id}, no={no_token_id}")

# ── CELL 14 ──
V_CAND = {}    # {layer: Tensor (n_cand, d_model)}

if EXPERIMENT["custom_sae"]:
    print("Loading custom SAE decoders ...")
    INDICES = {}

    CONFIG.update({
        "lr"         : 0.005,
        "lambda_l1"  : 0.0001,
        "n_epochs"   : 50,
        "patience"   : 10,
    })

    for layer in UNIQUE_LAYERS:
        W_dec = torch.load(CUST_SAE_PATHS[layer]).to(DEVICE)
        W_dec_T = W_dec.T    # (d_sae, d_model)

        d_sae = W_dec_T.shape[0]
        print(f"  Layer {layer}: d_sae={d_sae}")

        #==== Option A: ALL features =====#
        # INDICES[layer] = list(range(d_sae))
        # V_CAND[layer] = W_dec_T[INDICES[layer]].to(DEVICE)
        # CONFIG["lambda_l1"] = 0.00005

        #===== Option B: Top-N most active ====#
        top_n = 2000
        feature_info_path = CUST_SAE_PATHS[layer].parent / "feature_info.json"
        if feature_info_path.exists():
            with open(feature_info_path) as f:
                fi = json.load(f)
            top_indices = [r["index"] for r in sorted(fi["top_features"], key=lambda x: x["freq"], reverse=True)]
            if len(top_indices) < top_n:
                recorded = set(top_indices)
                extras = [i for i in range(d_sae) if i not in recorded]
                top_indices = top_indices + extras[:top_n - len(top_indices)]
            INDICES[layer] = top_indices[:top_n]
        else:
            print(f"  No feature_info.json for layer {layer}, using first {top_n} features")
            INDICES[layer] = list(range(top_n))

        V_CAND[layer] = W_dec_T[INDICES[layer]].to(DEVICE)
        print(f"  Layer {layer}: V_CAND {V_CAND[layer].shape}")
        del W_dec, W_dec_T

else:
    print(f"Loading SAE decoders (release: {width_config['sae_release']}, width: {EXPERIMENT['width']}) ...")

    for layer in UNIQUE_LAYERS:
        sae_id = get_sae_id(layer)
        print(f"  Layer {layer}: {sae_id} ...")

        sae, _, _ = SAE.from_pretrained(
            release=width_config["sae_release"], sae_id=sae_id, device=DEVICE,
        )
        W_dec = sae.W_dec.detach().clone().to(torch.float16)
        V_CAND[layer] = W_dec[INDICES[layer]].to(DEVICE)
        print(f"    V_cand[{layer}]: {V_CAND[layer].shape}")
        del sae, W_dec

torch.cuda.empty_cache()
print("\nAll SAE decoders loaded.")


# ── CELL 17 ──
PROMPT_TEMPLATE = 'Is the emotion "{emotion}" present in the following text? Answer only yes or no.\nText: "{text}"\nAnswer:'
def build_prompt(text, emotion): return PROMPT_TEMPLATE.format(text=text, emotion=emotion)

# Build layer index for each sample: which layer should steer this sample?
EMO_TO_LAYER_IDX = {e: UNIQUE_LAYERS.index(EMO_TO_LAYER[e]) for e in EMOTION_CLASSES}

def prepare_batches(df, tokenizer, batch_size, max_seq_len):
    prompts = [build_prompt(r["text"], r["emotion_query"]) for _, r in df.iterrows()]
    labels  = torch.tensor(df["label"].values, dtype=torch.float32)
    emo_ids = torch.tensor([EMOTION_TO_IDX[e] for e in df["emotion_query"]], dtype=torch.long)
    # Which layer each sample routes to (as index into UNIQUE_LAYERS)
    layer_ids = torch.tensor([EMO_TO_LAYER_IDX[e] for e in df["emotion_query"]], dtype=torch.long)

    batches = []
    for i in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[i:i+batch_size], return_tensors="pt",
                        padding=True, truncation=True, max_length=max_seq_len).to(DEVICE)
        batches.append((
            enc.input_ids, enc.attention_mask,
            labels[i:i+batch_size].to(DEVICE),
            emo_ids[i:i+batch_size].to(DEVICE),
            layer_ids[i:i+batch_size].to(DEVICE),
        ))
    print(f"  {len(batches)} batches of size {batch_size}")
    return batches

train_batches = prepare_batches(train_df, tokenizer, CONFIG["batch_size"], CONFIG["max_seq_len"])
eval_batches  = prepare_batches(eval_df,  tokenizer, CONFIG["batch_size"], CONFIG["max_seq_len"])

# ── CELL 19 ──
class MultiLayerSteeringContext:
    """
    Registers hooks at multiple layers. Each sample is steered only at
    the layer assigned to its emotion query.

    alpha_dict : {layer_idx_in_UNIQUE: nn.Parameter (n_emo_at_layer, n_cand)}
    V_dict     : {layer_idx_in_UNIQUE: Tensor (n_cand, d_model)}
    """
    def __init__(self, model, alpha_dict, V_dict, layer_ids, emo_ids):
        self.model      = model
        self.alpha_dict = alpha_dict
        self.V_dict     = V_dict
        self.layer_ids  = layer_ids    # (batch,) index into UNIQUE_LAYERS
        self.emo_ids    = emo_ids      # (batch,) global emotion index
        self.handles    = []

    def __enter__(self):
        for ul_idx, layer_j in enumerate(UNIQUE_LAYERS):
            nxt = layer_j + 1
            target = self.model.model.layers[nxt] if nxt < n_layers else self.model.model.norm
            handle = target.register_forward_pre_hook(self._make_hook(ul_idx, layer_j))
            self.handles.append(handle)
        return self

    def __exit__(self, *a):
        for h in self.handles: h.remove()
        self.handles = []

    def _make_hook(self, ul_idx, layer_j):
        # Which emotions route to this layer?
        emos_at_layer = LAYER_EMOTION_MAP[layer_j]
        emo_global_ids = [EMOTION_TO_IDX[e] for e in emos_at_layer]
        alpha = self.alpha_dict[ul_idx]    # (n_emo_at_layer, n_cand)
        V     = self.V_dict[ul_idx]        # (n_cand, d_model)

        def hook_fn(module, args):
            h = args[0]    # (batch, seq, d_model)
            bs = h.size(0)
            delta = torch.zeros(bs, h.size(-1), device=h.device, dtype=h.dtype)

            for local_idx, emo_gid in enumerate(emo_global_ids):
                mask = (self.emo_ids == emo_gid)
                if mask.sum() == 0: continue
                alpha_pos = F.softplus(alpha[local_idx])
                d_emo = torch.matmul(alpha_pos, V).to(dtype=h.dtype)
                delta[mask] = d_emo

            return (h + delta.unsqueeze(1),) + args[1:]
        return hook_fn


def forward_steered(ids, mask, model, alpha_dict, V_dict, layer_ids, emo_ids):
    with MultiLayerSteeringContext(model, alpha_dict, V_dict, layer_ids, emo_ids):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=ids, attention_mask=mask)
    sl = mask.sum(dim=1)-1; bi = torch.arange(ids.size(0), device=ids.device)
    return (out.logits[bi, sl, yes_token_id] - out.logits[bi, sl, no_token_id]).float()


def forward_unsteered(ids, mask, model):
    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = model(input_ids=ids, attention_mask=mask)
    sl = mask.sum(dim=1)-1; bi = torch.arange(ids.size(0), device=ids.device)
    return (out.logits[bi, sl, yes_token_id] - out.logits[bi, sl, no_token_id]).float()

print("Multi-layer steering ready.")

# ── CELL 21 ──
def train_sp2(model, V_CAND, INDICES, train_batches, eval_batches, cfg):
    # Build alpha parameters per layer
    alpha_dict = {}    # {ul_idx: nn.Parameter}
    V_dict = {}        # {ul_idx: Tensor}

    for ul_idx, layer in enumerate(UNIQUE_LAYERS):
        n_emo  = len(LAYER_EMOTION_MAP[layer])
        n_cand = len(INDICES[layer])
        alpha_dict[ul_idx] = nn.Parameter(torch.full((n_emo, n_cand), 0.01, device=DEVICE))
        V_dict[ul_idx] = V_CAND[layer]
        print(f"  Layer {layer:>2}: alpha ({n_emo}, {n_cand}) for {LAYER_EMOTION_MAP[layer]}")

    all_params = list(alpha_dict.values())
    # optimizer = Adam([{"params": all_params, "lr": cfg["lr"]}])

    #==better optimizer===#

    optimizer = Adam([{"params": all_params, "lr": cfg["lr"]}])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )



    n_pos = sum(int((l > 0.5).sum()) for _, _, l, _, _ in train_batches)
    n_neg = sum(int((l <= 0.5).sum()) for _, _, l, _, _ in train_batches)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    lam = cfg["lambda_l1"]; epochs = cfg["n_epochs"]; pat = cfg.get("patience", 10)
    history = {"train_loss": [], "eval_loss": [], "alpha_nnz": [], "alpha_l1": []}
    best_loss = float("inf")
    best_alpha = {k: v.detach().cpu().clone() for k, v in alpha_dict.items()}
    best_ep = 0; no_imp = 0

    total_params = sum(a.numel() for a in all_params)
    print(f"\n{'='*60}")
    print(f"  Training: {epochs} epochs, lr={cfg['lr']}, lam={lam}")
    print(f"  Total alpha params: {total_params}")
    print(f"  pos_weight: {pos_weight.item():.2f}, patience: {pat}")
    print(f"{'='*60}\n")

    for ep in range(epochs):
        ep_loss = 0.0; nb = 0
        for ids, mask, labels, emo_ids, layer_ids in train_batches:
            optimizer.zero_grad()
            scores = forward_steered(ids, mask, model, alpha_dict, V_dict, layer_ids, emo_ids)
            l1 = sum(F.softplus(a).sum() for a in all_params)
            loss = loss_fn(scores, labels) + lam * l1
            loss.backward(); optimizer.step()
            ep_loss += loss.item(); nb += 1
        tl = ep_loss / max(nb, 1)

        el = 0.0; ne = 0
        with torch.no_grad():
            for ids, mask, labels, emo_ids, layer_ids in eval_batches:
                scores = forward_steered(ids, mask, model, alpha_dict, V_dict, layer_ids, emo_ids)
                el += loss_fn(scores, labels).item(); ne += 1
        el = el / max(ne, 1)

        with torch.no_grad():
            all_sp = torch.cat([F.softplus(a).flatten() for a in all_params])
            nnz = int((all_sp > cfg["tau"]).sum()); l1v = float(all_sp.sum())

        history["train_loss"].append(tl); history["eval_loss"].append(el)
        history["alpha_nnz"].append(nnz); history["alpha_l1"].append(l1v)

        if el < best_loss:
            best_loss = el
            best_alpha = {k: v.detach().cpu().clone() for k, v in alpha_dict.items()}
            best_ep = ep + 1; no_imp = 0; mk = " * best"
        else:
            no_imp += 1; mk = f" ({no_imp}/{pat})"

        scheduler.step(el)


        current_lr = optimizer.param_groups[0]["lr"]
        # print(f"  Epoch {ep+1:>2}/{epochs}  |  train={tl:.4f}  eval={el:.4f}  "
        #       f"nnz={nnz:>4}  L1={l1v:.4f}{mk}")
        print(f"  Epoch {ep+1:>2}/{epochs}  |  train={tl:.4f}  eval={el:.4f}  "
              f"nnz={nnz:>4}  L1={l1v:.4f}  lr={current_lr:.6f}{mk}")

        if no_imp >= pat:
            print(f"\n  Early stopping at epoch {ep+1}"); break

    print(f"\n  Restoring best from epoch {best_ep} (eval_loss={best_loss:.4f})")
    return best_alpha, V_dict, history

# ── CELL 22 ──
alpha_star, V_dict, train_history = train_sp2(
    model, V_CAND, INDICES, train_batches, eval_batches, CONFIG
)

# ── CELL 24 ──
print("== Per-Emotion Feature Selection ==\n")
for ul_idx, layer in enumerate(UNIQUE_LAYERS):
    emos = LAYER_EMOTION_MAP[layer]
    asp = F.softplus(alpha_star[ul_idx]).numpy()
    #cands = CANDS[layer]
    for local_idx, emo in enumerate(emos):
        row = asp[local_idx]
        sel = (row > CONFIG["tau"]).sum()
        top = row.argsort()[-3:][::-1]
        print(f"  {emo:<10} (Layer {layer:>2}) |C*|={sel:>3}")
        for j in top:
            # print(f"    [{INDICES[layer][j]:>5}] sp={row[j]:.4f}  {cands[j].get("description","")[:55]}")
            print(f"    [{INDICES[layer][j]:>5}] sp={row[j]:.4f}")
        print()

# ── CELL 26 ──
try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(train_history["train_loss"], label="Train")
    axes[0].plot(train_history["eval_loss"], label="Eval")
    axes[0].legend(); axes[0].set_title("Loss")
    axes[1].plot(train_history["alpha_nnz"], color="purple"); axes[1].set_title("Active features")
    axes[2].plot(train_history["alpha_l1"], color="green"); axes[2].set_title("L1 norm")
    plt.tight_layout(); plt.show()
except ImportError: pass

# ── CELL 28 ──
print("Collecting scores ...\n")
all_su, all_ss, all_lb = [], [], []
# Move alpha_star to device for forward pass
alpha_dev = {k: v.to(DEVICE) for k, v in alpha_star.items()}

with torch.no_grad():
    for ids, mask, labels, emo_ids, layer_ids in eval_batches:
        all_su.append(forward_unsteered(ids, mask, model).cpu())
        all_ss.append(forward_steered(ids, mask, model, alpha_dev, V_dict, layer_ids, emo_ids).cpu())
        all_lb.append(labels.cpu())

all_su = torch.cat(all_su).numpy()
all_ss = torch.cat(all_ss).numpy()
all_lb = torch.cat(all_lb).numpy()

thresholds = np.arange(-5.0, 5.0, 0.1)
sweep = [{"t": round(t,1),
          "f1s": f1_score(all_lb, (all_ss>t).astype(float), average="macro", zero_division=0),
          "f1u": f1_score(all_lb, (all_su>t).astype(float), average="macro", zero_division=0),
          "nys": int((all_ss>t).sum()), "nyu": int((all_su>t).sum())} for t in thresholds]
sdf = pd.DataFrame(sweep)
BTS = sdf.loc[sdf["f1s"].idxmax(), "t"]
BTU = sdf.loc[sdf["f1u"].idxmax(), "t"]
print(f"Steered best:   t={BTS:+.1f}  F1={sdf.loc[sdf['f1s'].idxmax(), 'f1s']:.4f}")
print(f"Unsteered best: t={BTU:+.1f}  F1={sdf.loc[sdf['f1u'].idxmax(), 'f1u']:.4f}")
print(f"Actual positives: {int(all_lb.sum())} / {len(all_lb)}")

try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(sdf["t"], sdf["f1s"], label="Steered", color="#7F77DD")
    axes[0].plot(sdf["t"], sdf["f1u"], label="Unsteered", color="#E57373")
    axes[0].axvline(BTS, color="#7F77DD", linestyle=":"); axes[0].axvline(BTU, color="#E57373", linestyle=":")
    axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Macro F1"); axes[0].legend()
    axes[1].plot(sdf["t"], sdf["nys"], color="#7F77DD", label="Steered")
    axes[1].plot(sdf["t"], sdf["nyu"], color="#E57373", label="Unsteered")
    axes[1].axhline(int(all_lb.sum()), color="green", linestyle="--", label="actual")
    axes[1].legend(); plt.tight_layout(); plt.show()
except ImportError: pass

# ── CELL 30 ──
emotions = eval_df["emotion_query"].values[:len(all_lb)]

def eval_at(scores, labels, emotions, thresh):
    preds = (scores > thresh).astype(float)
    pe = {}
    for emo in EMOTION_CLASSES:
        m = (emotions == emo)
        if m.sum() > 0:
            pe[emo] = {"f1": f1_score(labels[m], preds[m], zero_division=0),
                       "acc": accuracy_score(labels[m], preds[m])}
    return {"acc": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro", zero_division=0),
            "prec": precision_score(labels, preds, average="macro", zero_division=0),
            "rec": recall_score(labels, preds, average="macro", zero_division=0),
            "pe": pe, "ny": int(preds.sum())}

steered_opt   = eval_at(all_ss, all_lb, emotions, BTS)
unsteered_opt = eval_at(all_su, all_lb, emotions, BTU)

comp = pd.DataFrame({
    "Metric": ["Accuracy", "Macro-F1", "Precision", "Recall"],
    "Unsteered": [unsteered_opt["acc"], unsteered_opt["f1"], unsteered_opt["prec"], unsteered_opt["rec"]],
    "Steered": [steered_opt["acc"], steered_opt["f1"], steered_opt["prec"], steered_opt["rec"]],
})
comp["D"] = comp["Steered"] - comp["Unsteered"]
print(f"Steered (t={BTS:+.1f}) vs Unsteered (t={BTU:+.1f})\n")
display(comp.style.format({"Unsteered": "{:.4f}", "Steered": "{:.4f}", "D": "{:+.4f}"}))

print("\n-- Per-Emotion (with steering layer) --")
er = []
for emo in EMOTION_CLASSES:
    u = unsteered_opt["pe"].get(emo, {}); s = steered_opt["pe"].get(emo, {})
    er.append({"Emotion": emo, "Layer": EMO_TO_LAYER[emo],
               "F1_u": u.get("f1",0), "F1_s": s.get("f1",0),
               "D": s.get("f1",0) - u.get("f1",0)})
edf = pd.DataFrame(er)
display(edf.style.format({"F1_u": "{:.4f}", "F1_s": "{:.4f}", "D": "{:+.4f}"}))

# ── CELL 32 ──
for sc, th, nm in [(all_ss, BTS, "Steered"), (all_su, BTU, "Unsteered")]:
    p = (sc > th).astype(float)
    f0 = f1_score(all_lb, p, pos_label=0, zero_division=0)
    f1 = f1_score(all_lb, p, pos_label=1, zero_division=0)
    print(f"  {nm} (t={th:+.1f}): yes={int(p.sum())} no={int((1-p).sum())} "
          f"F1(no)={f0:.4f} F1(yes)={f1:.4f} Macro={(f0+f1)/2:.4f}")

try:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, sc, th, ti in [(axes[0], all_su, BTU, "Unsteered"), (axes[1], all_ss, BTS, "Steered")]:
        ax.hist(sc[all_lb==0], bins=40, alpha=0.6, label="no", color="#E57373")
        ax.hist(sc[all_lb==1], bins=40, alpha=0.6, label="yes", color="#64B5F6")
        ax.axvline(th, color="black", linestyle="--"); ax.set_title(ti); ax.legend()
    diff = all_ss - all_su
    axes[2].hist(diff[all_lb==0], bins=40, alpha=0.6, label="no", color="#E57373")
    axes[2].hist(diff[all_lb==1], bins=40, alpha=0.6, label="yes", color="#64B5F6")
    axes[2].axvline(0, color="black", linestyle="--"); axes[2].set_title("Score shift"); axes[2].legend()
    plt.tight_layout(); plt.show()
except ImportError: pass

# ── CELL 34 ──
# Serialize alpha_dict: {ul_idx: tensor} -> {layer: list}
alpha_serialized = {}
for ul_idx, layer in enumerate(UNIQUE_LAYERS):
    alpha_serialized[str(layer)] = alpha_star[ul_idx].tolist()

save_dict = {
    "mode"                  : MODE,
    "alpha_per_layer"       : alpha_serialized,
    "indices_per_layer"     : {str(l): INDICES[l] for l in UNIQUE_LAYERS},
    "layer_emotion_map"     : {str(l): LAYER_EMOTION_MAP[l] for l in UNIQUE_LAYERS},
    "unique_layers"         : UNIQUE_LAYERS,
    "emotion_classes"       : EMOTION_CLASSES,
    "tau"                   : CONFIG["tau"],
    "lambda_l1"             : CONFIG["lambda_l1"],
    "lr"                    : CONFIG["lr"],
    "n_epochs"              : CONFIG["n_epochs"],
    "model_id"              : EXPERIMENT["model_id"],
    "language"              : EXPERIMENT["language"],
    "best_thresh_steered"   : float(BTS),
    "best_thresh_unsteered" : float(BTU),
}
with open(SP2_DIR / "alpha_star.json", "w") as f:
    json.dump(save_dict, f, indent=2)
print(f"Saved alpha* -> {SP2_DIR / 'alpha_star.json'}")

with open(SP2_DIR / "train_history.json", "w") as f:
    json.dump(train_history, f, indent=2)
print(f"Saved history -> {SP2_DIR / 'train_history.json'}")

# Save per-layer delta vectors
with torch.no_grad():
    deltas = {}
    for ul_idx, layer in enumerate(UNIQUE_LAYERS):
        asp = F.softplus(alpha_star[ul_idx].to(DEVICE)).to(dtype=V_CAND[layer].dtype)
        deltas[layer] = torch.matmul(asp, V_CAND[layer]).cpu()
    torch.save(deltas, SP2_DIR / "delta_star.pt")
print(f"Saved deltas -> {SP2_DIR / 'delta_star.pt'}")

# ── CELL 36 ──
print("\n== SP-2 Summary ==")
print(f"  Mode:       {MODE}")
print(f"  Model:      {EXPERIMENT['model_id']}")
print(f"  Language:   {EXPERIMENT['language']}")
print(f"  Layers:     {UNIQUE_LAYERS}")
print(f"  Epochs ran: {len(train_history['train_loss'])}")
print(f"  Threshold:  steered={BTS:+.1f}  unsteered={BTU:+.1f}")
print(f"  Steered F1:   {steered_opt['f1']:.4f}")
print(f"  Unsteered F1: {unsteered_opt['f1']:.4f}")
print(f"  Improvement:  {steered_opt['f1'] - unsteered_opt['f1']:+.4f}")
print(f"  Output dir:   {SP2_DIR}")

print(f"\n  Per-emotion:")
for _, r in edf.iterrows():
    print(f"    {r['Emotion']:<10} L{r['Layer']:>2}  F1: {r['F1_u']:.4f} -> {r['F1_s']:.4f} ({r['D']:+.4f})")

print(f"\n  Output files:")
for f in sorted(SP2_DIR.iterdir()): print(f"    {f.name}")


################################################################################
# FILE: sp2_sae_analysis.ipynb
################################################################################

# ── CELL 3 ──
import os; os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import json, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from google.colab import runtime
from sae_lens import SAE
import torch, torch.nn as nn, torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from sklearn.metrics import f1_score

import warnings; warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# ── CELL 5 ──
EXPERIMENT = {
    "model_id"   : "google/gemma-2-9b-it",
    "language"   : "indonesia",
    "custom_sae" : False,
    "width"      : "131k",
}

MODEL_EMOTION_LAYER = {
    "google/gemma-2-2b": {
        1  : ["disgust"],
        6  : ["sadness"],
        7  : ["anger", "neutral"],
        8  : ["surprise"],
        11 : ["joy"],
        19 : ["fear"],
    },
    "google/gemma-2-9b-it": {
        9  : ["anger", "surprise", "neutral"],
        20 : ["fear", "joy", "sadness"],
        31 : ["disgust"],
    },
    "google/gemma-2-9b": {
        3  : ["anger"],
        8  : ["surprise"],
        11 : ["joy"],
        13 : ["fear"],
        14 : ["sadness"],
        18 : ["neutral"],
        35 : ["disgust"],
    },
}

DATASET_REGISTRY = {
    "indonesia": {
        "hf_dataset_id"   : "brighter-dataset/BRIGHTER-emotion-categories",
        "hf_subset"       : "ind",
        "text_column"     : "text",
        "emotion_classes" : ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"],
    },
}

EVAL_CONFIG = {
    "max_eval_samples" : 100,
    "max_seq_len"      : 128,
    "batch_size"       : 4,
}

SAE_REGISTRY = {
    "google/gemma-2-2b": {
        "16k": {
            "sae_release" : "gemma-scope-2b-pt-res-canonical",
            "sae_id_fmt"  : "layer_{layer}/width_16k/canonical",
            "default_l0"  : None,
            "layer_l0"    : {},
            "note"        : "Canonical pick (L0 ~100). Available at all 26 layers.",
        },
        "65k": {
            "sae_release" : "gemma-scope-2b-pt-res",
            "sae_id_fmt"  : "layer_{layer}/width_65k/average_l0_{l0}",
            "default_l0"  : 107,
            "layer_l0"    : {1: 121, 6: 107, 7: 107, 8: 111, 11: 70, 19: 115},
            "note"        : "Available at all 26 layers.",
        },
        "default_layer" : 7,
    },
    "google/gemma-2-9b": {
        "16k": {
            "sae_release" : "gemma-scope-9b-pt-res-canonical",
            "sae_id_fmt"  : "layer_{layer}/width_16k/canonical",
            "default_l0"  : None,
            "layer_l0"    : {},
            "note"        : "Canonical pick. Available at all 42 layers.",
        },
        "131k": {
            "sae_release" : "gemma-scope-9b-pt-res",
            "sae_id_fmt"  : "layer_{layer}/width_131k/average_l0_{l0}",
            "default_l0"  : 121,
            "layer_l0"    : {3: 103, 8: 129, 11: 88, 13: 99, 14: 105, 18: 113, 35: 94},
            "note"        : "Available at subset of layers.",
        },
        "default_layer" : 16,
    },
    "google/gemma-2-9b-it": {
        "16k": {
            "sae_release" : "gemma-scope-9b-it-res",
            "sae_id_fmt"  : "layer_{layer}/width_16k/average_l0_{l0}",
            "default_l0"  : 91,
            "layer_l0"    : {9: 88, 20: 91, 31: 76},
            "note"        : "IT-specific SAEs. Only layers 9, 20, 31.",
        },
        "131k": {
            "sae_release" : "gemma-scope-9b-it-res",
            "sae_id_fmt"  : "layer_{layer}/width_131k/average_l0_{l0}",
            "default_l0"  : 81,
            "layer_l0"    : {9: 121, 20: 81, 31: 109},
            "note"        : "Using BASE model SAEs. Transfers well per Google's report.",
        },
        "default_layer" : 20,
    },
}

# ══════════════════════════════════════════════════════════════════════════
# AUTO-RESOLVE
# ══════════════════════════════════════════════════════════════════════════
def _slug(mid): return mid.split("/")[-1].lower()

MODEL_SLUG = _slug(EXPERIMENT["model_id"])
LANG_SLUG  = EXPERIMENT["language"].lower().replace(" ", "_")
WIDTH_SLUG = EXPERIMENT["width"].lower()
BASE_DIR   = Path(f"/content/drive/MyDrive/sae_outputs/{MODEL_SLUG}/{WIDTH_SLUG}/{LANG_SLUG}")

model_registry = SAE_REGISTRY[EXPERIMENT["model_id"]]
width_config   = model_registry[EXPERIMENT["width"]]
data_config    = DATASET_REGISTRY[EXPERIMENT["language"]]

EMOTION_CLASSES = data_config["emotion_classes"]
EMOTION_TO_IDX  = {e: i for i, e in enumerate(EMOTION_CLASSES)}

# ── SAE ID resolver ──────────────────────────────────────────────────────
def get_sae_id(layer):
    fmt = width_config["sae_id_fmt"]
    if "{l0}" not in fmt:
        return fmt.format(layer=layer)
    l0 = width_config["layer_l0"].get(layer, width_config["default_l0"])
    if l0 is None:
        raise ValueError(f"No L0 value for layer {layer}.")
    return fmt.format(layer=layer, l0=l0)

# ── Layer-emotion routing ────────────────────────────────────────────────
LAYER_EMOTION_MAP = MODEL_EMOTION_LAYER.get(EXPERIMENT["model_id"])

if LAYER_EMOTION_MAP is not None:
    UNIQUE_LAYERS = sorted(LAYER_EMOTION_MAP.keys())
    EMO_TO_LAYER = {}
    for layer, emos in LAYER_EMOTION_MAP.items():
        for e in emos:
            EMO_TO_LAYER[e] = layer
else:
    dl = model_registry.get("default_layer", 0)
    UNIQUE_LAYERS = [dl]
    EMO_TO_LAYER = {e: dl for e in EMOTION_CLASSES}
    LAYER_EMOTION_MAP = {dl: EMOTION_CLASSES}

# ── Paths ────────────────────────────────────────────────────────────────
CUST_SAE_PATHS = {layer: BASE_DIR / "cust_sae" / f"layer_{layer}" / "W_dec.pt" for layer in UNIQUE_LAYERS}

if EXPERIMENT["custom_sae"]:
    SP2_DIR = BASE_DIR / "sp2_multilayer_custom_sae"
else:
    SP2_DIR = BASE_DIR / "sp2_multilayer"

# ── Detect available SP-2 outputs ────────────────────────────────────────
sp2_dirs = list(BASE_DIR.glob("sp2_*"))
print(f"Available SP-2 outputs:")
for d in sp2_dirs:
    has = (d / "alpha_star.json").exists()
    print(f"  {d.name}  {'OK' if has else 'no alpha_star.json'}")

print(f"\nAnalyzing: {SP2_DIR}")
print(f"Exists: {SP2_DIR.exists()}")
print(f"Model: {EXPERIMENT['model_id']}")
print(f"Width: {EXPERIMENT['width']}")
print(f"Custom SAE: {EXPERIMENT['custom_sae']}")
print(f"SAE release: {width_config['sae_release']}")
print(f"Layers: {UNIQUE_LAYERS}")
if width_config.get("note"):
    print(f"Note: {width_config['note']}")

# ── CELL 6 ──
# EXPERIMENT = {
#     "model_id"   : "google/gemma-2-2b",
#     "language"   : "indonesia",
#     "custom_sae" : False,
#     "width"      : "65k",
# }


# DATASET_REGISTRY = {
#     "indonesia": {
#         "hf_dataset_id"   : "brighter-dataset/BRIGHTER-emotion-categories",
#         "hf_subset"       : "ind",
#         "text_column"     : "text",
#         "emotion_classes" : ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"],
#     },
# }

# EVAL_CONFIG = {
#     "max_eval_samples" : 100,
#     "max_seq_len"      : 128,
#     "batch_size"       : 4,
# }


# SAE_REGISTRY = {
#     "google/gemma-2-2b": {
#         "16k": {
#             "sae_release" : "gemma-scope-2b-pt-res-canonical",
#             "sae_id_fmt"  : "layer_{layer}/width_16k/canonical",
#             "default_l0"  : None,
#             "layer_l0"    : {},
#             "note"        : "Canonical pick (L0 ~100). Available at all 26 layers.",
#         },
#         "65k": {
#             "sae_release" : "gemma-scope-2b-pt-res",
#             "sae_id_fmt"  : "layer_{layer}/width_65k/average_l0_{l0}",
#             "default_l0"  : 107,
#             "layer_l0"    : {1: 121, 6: 107, 7: 107, 8: 111, 11: 70, 19: 115},
#             "note"        : "Available at all 26 layers.",
#         },
#         "default_layer" : 7,
#     },
#     "google/gemma-2-9b": {
#         "16k": {
#             "sae_release" : "gemma-scope-9b-pt-res-canonical",
#             "sae_id_fmt"  : "layer_{layer}/width_16k/canonical",
#             "default_l0"  : None,
#             "layer_l0"    : {},
#             "note"        : "Canonical pick. Available at all 42 layers.",
#         },
#         "131k": {
#             "sae_release" : "gemma-scope-9b-pt-res",
#             "sae_id_fmt"  : "layer_{layer}/width_131k/average_l0_{l0}",
#             "default_l0"  : 121,
#             "layer_l0"    : {3: 103, 8: 129, 11: 88, 13: 99, 14: 105, 18: 113, 35: 94},
#             "note"        : "Available at subset of layers.",
#         },
#         "default_layer" : 16,
#     },
#     "google/gemma-2-9b-it": {
#         "16k": {
#             "sae_release" : "gemma-scope-9b-it-res",
#             "sae_id_fmt"  : "layer_{layer}/width_16k/average_l0_{l0}",
#             "default_l0"  : 91,
#             "layer_l0"    : {9: 88, 20: 91, 31: 76},
#             "note"        : "IT-specific SAEs. Only layers 9, 20, 31.",
#         },
#         "131k": {
#             "sae_release" : "gemma-scope-9b-it-res",
#             "sae_id_fmt"  : "layer_{layer}/width_131k/average_l0_{l0}",
#             "default_l0"  : 81,
#             "layer_l0"    : {9: 121, 20: 81, 31: 109},
#             "note"        : "Using BASE model SAEs. Transfers well per Google's report.",
#         },
#         "default_layer" : 20,
#     },
# }


# def _slug(mid): return mid.split("/")[-1].lower()

# MODEL_SLUG  = _slug(EXPERIMENT["model_id"])
# LANG_SLUG   = EXPERIMENT["language"].lower().replace(" ", "_")
# WIDTH_SLUG  = EXPERIMENT["width"].lower()
# BASE_DIR    = Path(f"/content/drive/MyDrive/sae_outputs/{MODEL_SLUG}/{WIDTH_SLUG}/{LANG_SLUG}")

# # sae_config  = SAE_REGISTRY[EXPERIMENT["model_id"]]
# model_registry = SAE_REGISTRY[EXPERIMENT["model_id"]]
# width_config   = model_registry[EXPERIMENT["width"]]
# data_config    = DATASET_REGISTRY[EXPERIMENT["language"]]


# # ── SAE ID resolver ──────────────────────────────────────────────────────
# def get_sae_id(layer):
#     """Resolve the full SAE ID for a given layer using the active width config."""
#     fmt = width_config["sae_id_fmt"]
#     if "{l0}" not in fmt:
#         return fmt.format(layer=layer)
#     l0 = width_config["layer_l0"].get(layer, width_config["default_l0"])
#     if l0 is None:
#         raise ValueError(f"No L0 value for layer {layer}. "
#                          f"Check HF repo and add to layer_l0 dict.")
#     return fmt.format(layer=layer, l0=l0)

# if EXPERIMENT["custom_sae"]:
#   SAE_DIR    = Path(f"{BASE_DIR}/cust_sae/")
#   # Auto-detect SP-2 output directories
#   sp2_dirs = list(BASE_DIR.glob("sp2_*"))
#   print(f"Available SP-2 outputs:")
#   for d in sp2_dirs:
#       has = (d / "alpha_star.json").exists()
#       print(f"  {d.name}  {'OK' if has else 'no alpha_star.json'}")

#   # Select which to analyze
#   SP2_DIR = BASE_DIR / "sp2_multilayer_custom_sae"    # ← change if needed
#   print(f"\nAnalyzing: {SP2_DIR}")
# else:
#   SP2_DIR = BASE_DIR / "sp2_multilayer"    # change if needed
#   SAE_RELEASE = sae_config["sae_release"]
#   print(f"Analyzing: {SP2_DIR}")
#   print(f"Exists: {SP2_DIR.exists()}")


# # Emotion classes
# EMOTION_CLASSES = DATASET_CONFIG["emotion_classes"]
# EMOTION_TO_IDX  = {e: i for i, e in enumerate(EMOTION_CLASSES)}

# ── CELL 8 ──
with open(SP2_DIR / "alpha_star.json") as f:
    save_dict = json.load(f)

MODE = save_dict.get("mode", "unknown")
UNIQUE_LAYERS = save_dict["unique_layers"]
LAYER_EMOTION_MAP = {int(k): v for k, v in save_dict["layer_emotion_map"].items()}

EMO_TO_LAYER = {}
for layer, emos in LAYER_EMOTION_MAP.items():
    for e in emos:
        EMO_TO_LAYER[e] = layer

alpha_star = {}
for ul_idx, layer in enumerate(UNIQUE_LAYERS):
    alpha_star[ul_idx] = torch.tensor(save_dict["alpha_per_layer"][str(layer)])

INDICES = {int(k): v for k, v in save_dict["indices_per_layer"].items()}

saved_tau = save_dict.get("tau", 0.01)
saved_lam = save_dict.get("lambda_l1", 0.0001)

with open(SP2_DIR / "train_history.json") as f:
    train_history = json.load(f)

print(f"Mode: {MODE}")
print(f"Layers: {UNIQUE_LAYERS}")
print(f"Epochs: {len(train_history['train_loss'])}")
print(f"tau={saved_tau}, lambda={saved_lam}")
print(f"\nLayer-Emotion routing:")
for layer, emos in LAYER_EMOTION_MAP.items():
    ul_idx = UNIQUE_LAYERS.index(layer)
    n_cand = alpha_star[ul_idx].shape[-1]
    print(f"  Layer {layer:>2}: {emos}  ({n_cand} features)")

# ── CELL 10 ──
tl = train_history["train_loss"]; el = train_history["eval_loss"]
print(f"Train: {tl[0]:.4f} -> {tl[-1]:.4f}  ({tl[0]-tl[-1]:.4f} reduction)")
print(f"Eval:  {el[0]:.4f} -> {el[-1]:.4f}  ({el[0]-el[-1]:.4f} reduction)")

# Check overfitting
best_eval = min(el); best_epoch = el.index(best_eval) + 1
final_eval = el[-1]
print(f"\nBest eval: {best_eval:.4f} at epoch {best_epoch}")
if final_eval > best_eval * 1.05:
    print(f"WARNING: Final eval {final_eval:.4f} is {(final_eval/best_eval-1)*100:.1f}% worse than best -> overfitting")
else:
    print(f"No significant overfitting")

# Per-layer alpha stats
print(f"\nAlpha stats per layer:")
for ul_idx, layer in enumerate(UNIQUE_LAYERS):
    a = alpha_star[ul_idx]
    if a.dim() == 2:
        asp = F.softplus(a).numpy()
        for li, emo in enumerate(LAYER_EMOTION_MAP[layer]):
            sel = (asp[li] > saved_tau).sum()
            print(f"  L{layer:>2} {emo:<10} nnz={sel:>4}/{asp.shape[1]}  range=[{asp[li].min():.4f}, {asp[li].max():.4f}]")
    else:
        asp = F.softplus(a).numpy()
        sel = (asp > saved_tau).sum()
        print(f"  L{layer:>2} nnz={sel:>4}/{len(asp)}  range=[{asp.min():.4f}, {asp.max():.4f}]")

try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(tl, label="Train"); axes[0].plot(el, label="Eval")
    axes[0].axvline(best_epoch-1, color="green", linestyle="--", alpha=0.3, label=f"best epoch {best_epoch}")
    axes[0].legend(); axes[0].set_title("Loss")
    axes[1].plot(train_history.get("alpha_nnz", []), color="purple"); axes[1].set_title("Active features")
    axes[2].plot(train_history.get("alpha_l1", []), color="green"); axes[2].set_title("L1 norm")
    plt.tight_layout(); plt.show()
except ImportError: pass

# ── CELL 12 ──
tokenizer = AutoTokenizer.from_pretrained(EXPERIMENT["model_id"])
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

model_cfg = {"d_model": 2304}  # gemma-2-2b; change if different model
load_kw = {"pretrained_model_name_or_path": EXPERIMENT["model_id"],
           "device_map": "auto", "attn_implementation": "eager",
           "torch_dtype": torch.float16}
model = AutoModelForCausalLM.from_pretrained(**load_kw)
model.eval()
for p in model.parameters(): p.requires_grad = False
n_layers = model.config.num_hidden_layers
d_model = model.config.hidden_size
print(f"Model: d={d_model}, layers={n_layers}")

yes_token_id = tokenizer.encode("yes", add_special_tokens=False)[0]
no_token_id  = tokenizer.encode("no",  add_special_tokens=False)[0]
print(f"yes={yes_token_id}, no={no_token_id}")

# ── CELL 13 ──
V_CAND = {}

if EXPERIMENT["custom_sae"]:
    print(f"Loading custom SAE decoders ...")
    for layer in UNIQUE_LAYERS:
        wdec_path = CUST_SAE_PATHS[layer]
        if wdec_path.exists():
            W_dec = torch.load(wdec_path, weights_only=True)
            W_dec_T = W_dec.T    # (d_sae, d_model)

            layer_int = int(layer)
            if layer_int in INDICES and len(INDICES[layer_int]) > 0:
                V_CAND[layer] = W_dec_T[INDICES[layer_int]].to(torch.float16).to(DEVICE)
            else:
                # Fallback: use all features
                n_feat = W_dec_T.shape[0]
                INDICES[layer_int] = list(range(n_feat))
                V_CAND[layer] = W_dec_T.to(torch.float16).to(DEVICE)
                print(f"    No INDICES for layer {layer}, using all {n_feat} features")

            print(f"  Layer {layer:>2}: V_CAND {V_CAND[layer].shape}")
            del W_dec, W_dec_T
        else:
            print(f"  Layer {layer:>2}: W_dec NOT FOUND at {wdec_path}")

else:
    print(f"Loading SAE decoders (release: {width_config['sae_release']}, width: {EXPERIMENT['width']}) ...")
    for layer in UNIQUE_LAYERS:
        sae_id = get_sae_id(layer)
        print(f"  Layer {layer:>2}: {sae_id} ...")

        sae, _, _ = SAE.from_pretrained(
            release=width_config["sae_release"], sae_id=sae_id, device=DEVICE,
        )
        W = sae.W_dec.detach().clone().to(torch.float16)

        layer_int = int(layer)
        if layer_int in INDICES and len(INDICES[layer_int]) > 0:
            V_CAND[layer] = W[INDICES[layer_int]].to(DEVICE)
        else:
            # Fallback: use all features
            INDICES[layer_int] = list(range(W.shape[0]))
            V_CAND[layer] = W.to(DEVICE)
            print(f"    No INDICES for layer {layer}, using all {W.shape[0]} features")

        print(f"    V_CAND[{layer}]: {V_CAND[layer].shape}")
        del sae, W

torch.cuda.empty_cache()
print("\nAll SAE decoders loaded.")

# ── CELL 14 ──
# # Load custom SAE decoder weights
# V_CAND = {}    # {layer: Tensor (n_cand, d_model)}

# if EXPERIMENT["custom_sae"]:
#   for layer in UNIQUE_LAYERS:
#       wdec_path = SAE_DIR/f"layer_{layer}/W_dec.pt"
#       if wdec_path.exists():
#           W_dec = torch.load(wdec_path, weights_only=True)
#           W_dec_T = W_dec.T

#           # Debug: confirm key exists
#           print(f"  Layer {layer} (type={type(layer)})  INDICES keys: {list(INDICES.keys())[:7]}")

#           # Force integer key lookup
#           layer_int = int(layer)
#           V_CAND[layer] = W_dec_T[INDICES[layer_int]].to(torch.float16).to(DEVICE)
#           print(f"  Layer {layer}: V_CAND {V_CAND[layer].shape}")
#           del W_dec, W_dec_T
#       else:
#           print(f"  Layer {layer}: W_dec NOT FOUND at {wdec_path}")
# else:
#   for layer in UNIQUE_LAYERS:
#     if EXPERIMENT["model_id"] == "google/gemma-2-2b":
#         sae_id = f"layer_{layer}/width_16k/canonical"
#     else:
#         sae_id = sae_config["layers"][layer]
#     sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id, device=DEVICE)
#     W = sae.W_dec.detach().clone().to(torch.float16)
#     V_CAND[layer] = W[INDICES[layer]].to(DEVICE)
#     print(f"  Layer {layer:>2}: V_cand {V_CAND[layer].shape}")
#     del sae, W
# torch.cuda.empty_cache()

# ── CELL 15 ──
# Dataset
cfg = data_config
lk = {"path": cfg["hf_dataset_id"]}
if cfg["hf_subset"]: lk["name"] = cfg["hf_subset"]
ds = load_dataset(**lk)
if "validation" in ds and "dev" not in ds: ds["dev"] = ds["validation"]
av = set(ds.keys())
if "train" in av and "test" in av: esp = "test"
elif "train" in av and "dev" in av: esp = "dev"
else: esp = "test"
split = ds[esp].select(range(min(EVAL_CONFIG["max_eval_samples"], len(ds[esp]))))
df = split.to_pandas()
emotion_cols = [e for e in EMOTION_CLASSES if e != "neutral"]
df["neutral"] = (df[emotion_cols].sum(axis=1) == 0).astype(int)
for e in EMOTION_CLASSES: df[e] = df[e].astype(int)
print(f"Eval: {len(df)} texts")

# Build binary pairs (same as old SP-2 binary approach)
PROMPT_TEMPLATE = 'Is the emotion "{emotion}" present in the following text? Answer only yes or no.\nText: "{text}"\nAnswer:'
def build_prompt(text, emotion): return PROMPT_TEMPLATE.format(text=text, emotion=emotion)

eval_rows = []
for _, row in df.iterrows():
    for emo in EMOTION_CLASSES:
        eval_rows.append({"text": row["text"], "emotion_query": emo, "label": int(row[emo])})
eval_df = pd.DataFrame(eval_rows)
print(f"Eval pairs: {len(eval_df)}")

# Batches
prompts = [build_prompt(r["text"], r["emotion_query"]) for _, r in eval_df.iterrows()]
labels_t = torch.tensor(eval_df["label"].values, dtype=torch.float32)
emo_ids_t = torch.tensor([EMOTION_TO_IDX[e] for e in eval_df["emotion_query"]], dtype=torch.long)

eval_batches = []
bs = EVAL_CONFIG["batch_size"]
for i in range(0, len(prompts), bs):
    enc = tokenizer(prompts[i:i+bs], return_tensors="pt", padding=True,
                    truncation=True, max_length=EVAL_CONFIG["max_seq_len"]).to(DEVICE)
    eval_batches.append((enc.input_ids, enc.attention_mask,
                         labels_t[i:i+bs].to(DEVICE), emo_ids_t[i:i+bs].to(DEVICE)))
print(f"Batches: {len(eval_batches)}")

# Scoring function
def forward_unsteered(ids, mask):
    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = model(input_ids=ids, attention_mask=mask)
    sl = mask.sum(dim=1)-1; bi = torch.arange(ids.size(0), device=DEVICE)
    return (out.logits[bi, sl, yes_token_id] - out.logits[bi, sl, no_token_id]).float()

def forward_with_layers(ids, mask, active_layers, emo_ids_batch=None):
    """Steer with only the specified layers active."""
    if not active_layers:
        return forward_unsteered(ids, mask)

    handles = []
    for ul_idx, layer_j in enumerate(UNIQUE_LAYERS):
        if layer_j not in active_layers: continue
        a = alpha_star[ul_idx].to(DEVICE)
        V = V_CAND[layer_j]

        if a.dim() == 2 and emo_ids_batch is not None:
            # Emotion-conditioned: per-sample delta
            emos_at = LAYER_EMOTION_MAP[layer_j]
            emo_gids = [EMOTION_TO_IDX[e] for e in emos_at]
            def make_hook_cond(a_l, V_l, gids):
                def fn(module, args):
                    h = args[0]
                    delta = torch.zeros(h.size(0), h.size(-1), device=h.device, dtype=h.dtype)
                    for li, eid in enumerate(gids):
                        m = (emo_ids_batch == eid)
                        if m.sum() == 0: continue
                        ap = F.softplus(a_l[li]).to(dtype=V_l.dtype)
                        delta[m] = torch.matmul(ap, V_l).to(dtype=h.dtype)
                    return (h + delta.unsqueeze(1),) + args[1:]
                return fn
            hook_fn = make_hook_cond(a, V, emo_gids)
        else:
            # Shared alpha: same delta for all samples
            def make_hook_shared(a_l, V_l):
                def fn(module, args):
                    h = args[0]
                    ap = F.softplus(a_l).to(dtype=V_l.dtype)
                    delta = torch.matmul(ap, V_l).to(dtype=h.dtype)
                    return (h + delta.unsqueeze(0).unsqueeze(0),) + args[1:]
                return fn
            hook_fn = make_hook_shared(a, V)

        nxt = layer_j + 1
        target = model.model.layers[nxt] if nxt < n_layers else model.model.norm
        handles.append(target.register_forward_pre_hook(hook_fn))

    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = model(input_ids=ids, attention_mask=mask)
    for h in handles: h.remove()
    sl = mask.sum(dim=1)-1; bi = torch.arange(ids.size(0), device=DEVICE)
    return (out.logits[bi, sl, yes_token_id] - out.logits[bi, sl, no_token_id]).float()

print("Helpers ready.")

# ── CELL 17 ──
configs = {"Unsteered": []}
for layer in UNIQUE_LAYERS:
    configs[f"L{layer} only"] = [layer]
configs["All layers"] = list(UNIQUE_LAYERS)

print(f"Configurations to test: {list(configs.keys())}\n")

all_scores = {}   # {config_name: numpy array of scores}
all_labels = None

for name, active in configs.items():
    print(f"  {name} ...", end="", flush=True)
    scores_list = []
    labels_list = []
    with torch.no_grad():
        for ids, mask, labels, emo_ids in eval_batches:
            s = forward_with_layers(ids, mask, active, emo_ids).cpu()
            scores_list.append(s)
            labels_list.append(labels.cpu())
    all_scores[name] = torch.cat(scores_list).numpy()
    if all_labels is None:
        all_labels = torch.cat(labels_list).numpy()
    print(f" done ({len(all_scores[name])} scores)")

emotions_arr = eval_df["emotion_query"].values[:len(all_labels)]
print(f"\nTotal: {len(all_labels)} predictions per config")

# ── CELL 19 ──
print("== Threshold Sweep ==\n")

thresholds = np.arange(-5.0, 5.0, 0.1)
config_results = {}

for name in configs:
    sc = all_scores[name]
    best_f1 = 0; best_t = 0
    for t in thresholds:
        f = f1_score(all_labels, (sc > t).astype(float), average="macro", zero_division=0)
        if f > best_f1: best_f1 = f; best_t = t
    config_results[name] = {"macro_f1": best_f1, "threshold": best_t}
    print(f"  {name:<16} t={best_t:+.1f}  F1={best_f1:.4f}")

# Find best config
base_f1 = config_results["Unsteered"]["macro_f1"]
best_name = max(config_results, key=lambda n: config_results[n]["macro_f1"])
print(f"\n  Baseline:  {base_f1:.4f}")
print(f"  Best:      {best_name} -> {config_results[best_name]['macro_f1']:.4f} ({config_results[best_name]['macro_f1']-base_f1:+.4f})")

try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    for name in configs:
        sc = all_scores[name]
        f1s = [f1_score(all_labels, (sc>t).astype(float), average="macro", zero_division=0) for t in thresholds]
        style = {"linewidth": 2, "alpha": 0.9} if name in ["Unsteered", "All layers"] else {"linewidth": 1, "alpha": 0.6}
        plt.plot(thresholds, f1s, label=f"{name} ({config_results[name]['macro_f1']:.4f})", **style)
    plt.xlabel("Threshold"); plt.ylabel("Macro F1")
    plt.title("F1 vs Threshold per Configuration"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.show()
except ImportError: pass

# ── CELL 21 ──
print("== Per-Emotion F1 (at each config's optimal threshold) ==\n")

# Header
header = f"{'Config':<16}"
for e in EMOTION_CLASSES: header += f"  {e[:5]:>6}"
header += f"  {'Macro':>6}"
print(header)
print("-" * len(header))

per_emo_table = []
for name in configs:
    sc = all_scores[name]
    bt = config_results[name]["threshold"]
    preds = (sc > bt).astype(float)

    row = {"Config": name}
    per_emo_f1s = []
    line = f"{name:<16}"
    for emo in EMOTION_CLASSES:
        m = (emotions_arr == emo)
        f1 = f1_score(all_labels[m], preds[m], zero_division=0)
        row[emo] = f1
        per_emo_f1s.append(f1)
        line += f"  {f1:>6.4f}"
    macro = config_results[name]["macro_f1"]
    row["Macro"] = macro
    line += f"  {macro:>6.4f}"
    per_emo_table.append(row)
    print(line)

pe_df = pd.DataFrame(per_emo_table)

# Delta vs baseline
print(f"\n== Delta vs Unsteered ==\n")
base_row = pe_df[pe_df["Config"] == "Unsteered"].iloc[0]
header2 = f"{'Config':<16}"
for e in EMOTION_CLASSES: header2 += f"  {e[:5]:>7}"
header2 += f"  {'Macro':>7}"
print(header2)
print("-" * len(header2))

for _, row in pe_df.iterrows():
    if row["Config"] == "Unsteered": continue
    line = f"{row['Config']:<16}"
    for emo in EMOTION_CLASSES:
        d = row[emo] - base_row[emo]
        line += f"  {d:>+7.4f}"
    d_macro = row["Macro"] - base_row["Macro"]
    line += f"  {d_macro:>+7.4f}"
    print(line)

# ── CELL 23 ──
try:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart: macro F1 per config
    names = list(configs.keys())
    macros = [config_results[n]["macro_f1"] for n in names]
    colors = ["#999" if n == "Unsteered"
              else ("#2ECC71" if config_results[n]["macro_f1"] > base_f1 else "#E74C3C")
              for n in names]
    axes[0].bar(names, macros, color=colors)
    axes[0].axhline(base_f1, color="black", linestyle="--", alpha=0.5, label="baseline")
    axes[0].set_ylabel("Macro F1"); axes[0].set_title("Macro F1 by Configuration")
    axes[0].tick_params(axis="x", rotation=30); axes[0].legend()

    # Heatmap: per-emotion delta
    steered_names = [n for n in names if n != "Unsteered"]
    delta_matrix = np.zeros((len(steered_names), len(EMOTION_CLASSES)))
    for i, name in enumerate(steered_names):
        for j, emo in enumerate(EMOTION_CLASSES):
            delta_matrix[i, j] = pe_df[pe_df["Config"]==name].iloc[0][emo] - base_row[emo]

    im = axes[1].imshow(delta_matrix, aspect="auto", cmap="RdYlGn", vmin=-0.2, vmax=0.2)
    axes[1].set_xticks(range(len(EMOTION_CLASSES)))
    axes[1].set_xticklabels([e[:5] for e in EMOTION_CLASSES], rotation=45)
    axes[1].set_yticks(range(len(steered_names)))
    axes[1].set_yticklabels(steered_names)
    axes[1].set_title("Per-Emotion F1 Delta vs Unsteered")
    plt.colorbar(im, ax=axes[1], label="Delta F1")

    # Annotate cells
    for i in range(len(steered_names)):
        for j in range(len(EMOTION_CLASSES)):
            axes[1].text(j, i, f"{delta_matrix[i,j]:+.3f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if abs(delta_matrix[i,j]) > 0.1 else "black")

    plt.tight_layout(); plt.show()
except ImportError: pass

# ── CELL 25 ──
try:
    n_configs = len(configs)
    fig, axes = plt.subplots(1, n_configs, figsize=(5*n_configs, 4))
    if n_configs == 1: axes = [axes]

    for i, name in enumerate(configs):
        sc = all_scores[name]; bt = config_results[name]["threshold"]
        axes[i].hist(sc[all_labels==0], bins=40, alpha=0.6, label="no", color="#E57373")
        axes[i].hist(sc[all_labels==1], bins=40, alpha=0.6, label="yes", color="#64B5F6")
        axes[i].axvline(bt, color="black", linestyle="--")
        axes[i].set_title(f"{name}\nt={bt:+.1f}")
        axes[i].legend(fontsize=7)
    plt.suptitle("Score Distributions"); plt.tight_layout(); plt.show()

    # Shift histograms vs unsteered
    sc_base = all_scores["Unsteered"]
    steered_names = [n for n in configs if n != "Unsteered"]
    fig, axes = plt.subplots(1, len(steered_names), figsize=(5*len(steered_names), 4))
    if len(steered_names) == 1: axes = [axes]
    for i, name in enumerate(steered_names):
        diff = all_scores[name] - sc_base
        axes[i].hist(diff[all_labels==0], bins=40, alpha=0.6, label="true=no", color="#E57373")
        axes[i].hist(diff[all_labels==1], bins=40, alpha=0.6, label="true=yes", color="#64B5F6")
        axes[i].axvline(0, color="black", linestyle="--")
        axes[i].set_title(f"{name} shift"); axes[i].legend(fontsize=7)
    plt.suptitle("Score Shift vs Unsteered (discriminative = different colors different sides)")
    plt.tight_layout(); plt.show()
except ImportError: pass

# ── CELL 27 ──
print("== Analysis & Recommendation ==\n")

# 1. Best overall config
print(f"1. BEST CONFIGURATION: {best_name}")
print(f"   F1={config_results[best_name]['macro_f1']:.4f} vs baseline {base_f1:.4f} ({config_results[best_name]['macro_f1']-base_f1:+.4f})")

# 2. Single-layer comparison
print(f"\n2. SINGLE-LAYER RANKING:")
single_layers = [(n, config_results[n]["macro_f1"]) for n in configs if "only" in n]
single_layers.sort(key=lambda x: x[1], reverse=True)
for rank, (name, f1) in enumerate(single_layers):
    d = f1 - base_f1
    verdict = "HELPS" if d > 0.005 else "HURTS" if d < -0.005 else "neutral"
    marker = " <-- BEST SINGLE" if rank == 0 else ""
    print(f"   {name:<16} F1={f1:.4f} ({d:+.4f}) {verdict}{marker}")

best_single = single_layers[0][0]
best_single_f1 = single_layers[0][1]

# 3. All-layers vs best single
all_f1 = config_results["All layers"]["macro_f1"]
print(f"\n3. ALL LAYERS vs BEST SINGLE:")
print(f"   All layers:  {all_f1:.4f}")
print(f"   {best_single}: {best_single_f1:.4f}")
if best_single_f1 > all_f1 + 0.005:
    print(f"   -> Single layer WINS. Multi-layer adds interference.")
elif all_f1 > best_single_f1 + 0.005:
    print(f"   -> All layers WINS. Multi-layer synergy.")
else:
    print(f"   -> Roughly equal.")

# 4. Score shift analysis
print(f"\n4. SHIFT ANALYSIS:")
for name in ["All layers", best_single]:
    diff = all_scores[name] - all_scores["Unsteered"]
    yes_shift = diff[all_labels==1].mean()
    no_shift  = diff[all_labels==0].mean()
    print(f"   {name:<16} yes_shift={yes_shift:+.4f}  no_shift={no_shift:+.4f}", end="")
    if abs(yes_shift - no_shift) < 0.01:
        print(" -> UNIFORM")
    elif yes_shift > 0 and no_shift < 0:
        print(" -> DISCRIMINATIVE (good!)")
    elif yes_shift > no_shift:
        print(" -> Partially discriminative")
    else:
        print(" -> Wrong direction")

# 5. Per-emotion winners
print(f"\n5. BEST CONFIG PER EMOTION:")
for emo in EMOTION_CLASSES:
    best_emo_f1 = 0; best_emo_cfg = ""
    for name in configs:
        f1 = pe_df[pe_df["Config"]==name].iloc[0][emo]
        if f1 > best_emo_f1: best_emo_f1 = f1; best_emo_cfg = name
    d = best_emo_f1 - base_row[emo]
    print(f"   {emo:<10} {best_emo_cfg:<16} F1={best_emo_f1:.4f} ({d:+.4f})")

# 6. Summary
print(f"\n6. SUGGESTED NEXT STEPS:")
d_best = config_results[best_name]["macro_f1"] - base_f1
if d_best > 0.02:
    print(f"   Custom SAE steering works (+{d_best:.4f}). Use config: {best_name}")
elif d_best > 0:
    print(f"   Small improvement (+{d_best:.4f}). Try:")
    print(f"     (a) More training data for SAE (increase max_texts)")
    print(f"     (b) Lower lr with more epochs")
    print(f"     (c) Reduce to fewer features (top 500 by activation)")
else:
    print(f"   No improvement. Custom SAE features may need more training data")
    print(f"   or a different corpus (mc4/CulturaX instead of BRIGHTER).")

torch.cuda.empty_cache()

# ── CELL 29 ──
if EXPERIMENT['custom_sae']:
  diag_dir = BASE_DIR / "diagnostics/cust_sae"
else:
  diag_dir = BASE_DIR / "diagnostics/neuronpedia_sae"

diag_dir.mkdir(parents=True, exist_ok=True)

diag_out = {
    "configs": {name: {"macro_f1": config_results[name]["macro_f1"],
                       "threshold": config_results[name]["threshold"]}
                for name in configs},
    "per_emotion": pe_df.to_dict(orient="records"),
    "best_config": best_name,
    "best_macro": config_results[best_name]["macro_f1"],
    "baseline_macro": base_f1,
}
with open(diag_dir / "custom_sae_analysis.json", "w") as f:
    json.dump(diag_out, f, indent=2)
pe_df.to_csv(diag_dir / "custom_sae_per_emotion.csv", index=False)
print(f"Saved to {diag_dir}")
runtime.unassign()
