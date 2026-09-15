"""
Model loading utilities. Handles 4-bit quantization and tokenizer setup.
"""

import gc
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
