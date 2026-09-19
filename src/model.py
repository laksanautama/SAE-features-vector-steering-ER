"""
Model loading utilities. Handles 4-bit quantization and tokenizer setup.
"""

import gc
import shutil
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.config import should_use_4bit


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(model_id: str, force_4bit: bool = None):
    """
    Load model and tokenizer. Handles 4-bit quantization automatically.

    Returns:
        (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = force_4bit if force_4bit is not None else should_use_4bit(model_id)

    load_kw = {
        "pretrained_model_name_or_path": model_id,
        "device_map": "auto",
        "attn_implementation": "eager",
    }

    if use_4bit:
        print(f"  Loading {model_id} in 4-bit quantization ...")
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        print(f"  Loading {model_id} in float16 ...")
        load_kw["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(**load_kw)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"  d_model={d_model}, n_layers={n_layers}")

    return model, tokenizer


def free_model(model, tokenizer=None):
    """Delete model (and optionally tokenizer) and clear all GPU memory."""
    del model
    if tokenizer is not None:
        del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("  Model freed from GPU memory.")


def get_token_ids(tokenizer) -> tuple:
    """Get yes/no token IDs for binary classification."""
    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("no", add_special_tokens=False)[0]
    return yes_id, no_id


def clear_hf_cache(keep_datasets=True):
    """
    Clear HuggingFace cache to free disk space on quota-limited HPC.

    Removes:
      - HF hub cache       (model weights, SAE files — the big ones)
      - sae_lens cache     (if present)

    Keeps:
      - HF datasets cache  (reused across runs, unless keep_datasets=False)
      - HF login token

    Uses HuggingFace's own path resolution, which respects:
      HF_HUB_CACHE  → exact hub cache path
      HF_HOME       → {HF_HOME}/hub
      default       → ~/.cache/huggingface/hub

    Call this after free_model() at the end of each (model, width) scenario.
    """
    from huggingface_hub import constants as hf_constants

    # 1. Hub cache — resolve the real path the HF libraries actually use
    hub_dir = Path(hf_constants.HF_HUB_CACHE)
    if hub_dir.exists():
        size_mb = sum(f.stat().st_size for f in hub_dir.rglob("*") if f.is_file()) / (1024 ** 2)
        shutil.rmtree(hub_dir, ignore_errors=True)
        print(f"  Cleared HF hub cache ({hub_dir}): ~{size_mb:.0f} MB freed")

    # 2. Datasets cache (optional)
    if not keep_datasets:
        # datasets library stores its cache location in HF_DATASETS_CACHE or under HF_HOME
        import os
        ds_dir = Path(os.environ.get(
            "HF_DATASETS_CACHE",
            hub_dir.parent / "datasets"   # sibling of hub/
        ))
        if ds_dir.exists():
            size_mb = sum(f.stat().st_size for f in ds_dir.rglob("*") if f.is_file()) / (1024 ** 2)
            shutil.rmtree(ds_dir, ignore_errors=True)
            print(f"  Cleared HF datasets cache ({ds_dir}): ~{size_mb:.0f} MB freed")

    # 3. sae_lens cache — check common locations + under HF_HOME
    hf_home = hub_dir.parent  # e.g. ~/.cache/huggingface or $HF_HOME
    for candidate in [
        hf_home / "sae_lens",
        hf_home / "saelens",
        Path.home() / ".cache" / "sae_lens",
        Path.home() / ".cache" / "saelens",
    ]:
        if candidate.exists():
            size_mb = sum(f.stat().st_size for f in candidate.rglob("*") if f.is_file()) / (1024 ** 2)
            shutil.rmtree(candidate, ignore_errors=True)
            print(f"  Cleared sae_lens cache ({candidate}): ~{size_mb:.0f} MB freed")

    gc.collect()
