"""
SAE loading utilities. Handles both pre-trained (Neuronpedia) and custom SAEs.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sae_lens import SAE

from src.config import get_sae_id, get_width_config


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_pretrained_sae_decoder(model_id: str, width: str, layer: int,
                                 indices: list = None):
    """
    Load W_dec from a pre-trained SAE (Neuronpedia/Gemma Scope).

    Returns:
        V_cand: Tensor (n_features, d_model)
    """
    wcfg = get_width_config(model_id, width)
    sae_id = get_sae_id(model_id, width, layer)
    print(f"    SAE: {sae_id}")

    sae, _, _ = SAE.from_pretrained(
        release=wcfg["sae_release"], sae_id=sae_id, device=DEVICE,
    )
    W_dec = sae.W_dec.detach().clone().to(torch.float16)

    if indices is not None:
        V_cand = W_dec[indices].to(DEVICE)
    else:
        V_cand = W_dec.to(DEVICE)

    print(f"    V_cand: {V_cand.shape}")
    del sae, W_dec
    torch.cuda.empty_cache()
    return V_cand


def load_custom_sae_decoder(wdec_path: Path, indices: list = None):
    """
    Load W_dec from a custom-trained SAE.

    Returns:
        V_cand: Tensor (n_features, d_model)
    """
    W_dec = torch.load(wdec_path, weights_only=True)
    W_dec_T = W_dec.T  # (d_sae, d_model)

    if indices is not None:
        V_cand = W_dec_T[indices].to(torch.float16).to(DEVICE)
    else:
        V_cand = W_dec_T.to(torch.float16).to(DEVICE)

    print(f"    V_cand: {V_cand.shape}")
    del W_dec, W_dec_T
    torch.cuda.empty_cache()
    return V_cand


def load_candidates(path: Path) -> list:
    """Load f_cand.json (works for both semantic and activation-based)."""
    with open(path) as f:
        ch = f.read(1)
        f.seek(0)
        cands = json.load(f) if ch == "[" else [json.loads(l) for l in f if l.strip()]
    for feat in cands:
        if "index" not in feat:
            feat["index"] = int(feat["id"].split(":")[-1])
        else:
            feat["index"] = int(feat["index"])
    return cands


def load_sae_encoder(model_id: str, width: str, layer: int, d_model: int):
    """
    Load SAE encoder for activation extraction.

    Returns:
        (sae_encoder, sae_b_dec) — nn.Linear + bias tensor
    """
    wcfg = get_width_config(model_id, width)
    sae_id = get_sae_id(model_id, width, layer)
    sae_obj, _, _ = SAE.from_pretrained(
        release=wcfg["sae_release"], sae_id=sae_id, device=DEVICE,
    )
    d_sae = sae_obj.cfg.d_sae

    sae_encoder = nn.Linear(d_model, d_sae, bias=True).to(DEVICE)
    with torch.no_grad():
        sae_encoder.weight.data = sae_obj.W_enc.detach().clone().float().T
        if hasattr(sae_obj, "b_enc"):
            sae_encoder.bias.data = sae_obj.b_enc.detach().clone().float()
        else:
            sae_encoder.bias.data.zero_()

    sae_b_dec = sae_obj.b_dec.detach().clone().float().to(DEVICE)
    del sae_obj
    torch.cuda.empty_cache()

    return sae_encoder, sae_b_dec, d_sae
