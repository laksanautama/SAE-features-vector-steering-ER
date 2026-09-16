"""
Configuration loading and resolution utilities.
All paths, registries, and experiment settings are resolved here.
"""

import os
import yaml
from pathlib import Path
from typing import Optional


_hf_logged_in = False


def hf_login():
    """
    Authenticate with HuggingFace Hub once per process.

    Resolution order:
      1. HF_TOKEN environment variable  (preferred — no secrets in files)
      2. hf_token field in experiment.yaml
      3. Existing huggingface-cli login cache (~/.cache/huggingface/token)

    Call this at the top of every script's main(). Subsequent calls are no-ops.
    """
    global _hf_logged_in
    if _hf_logged_in:
        return

    from huggingface_hub import login, get_token

    # 1. Env variable
    token = os.environ.get("HF_TOKEN")

    # 2. YAML fallback
    if not token:
        exp = load_experiment_config()
        token = exp.get("hf_token")
        # Ignore placeholder
        if token and token.startswith("YOUR_"):
            token = None

    # 3. Existing cache
    if not token:
        cached = get_token()
        if cached:
            print("  HF auth: using cached token from huggingface-cli login")
            _hf_logged_in = True
            return

    if not token:
        raise RuntimeError(
            "No HuggingFace token found. Either:\n"
            "  • Set the HF_TOKEN environment variable, or\n"
            "  • Paste your token in config/experiment.yaml under hf_token, or\n"
            "  • Run `huggingface-cli login` first."
        )

    login(token=token, add_to_git_credential=False)
    _hf_logged_in = True
    print("  HF auth: logged in successfully")


def _find_config_dir() -> Path:
    """Find the config/ directory relative to the project root."""
    # Try relative to this file first
    here = Path(__file__).resolve().parent.parent / "config"
    if here.exists():
        return here
    # Try current working directory
    cwd = Path.cwd() / "config"
    if cwd.exists():
        return cwd
    raise FileNotFoundError("Cannot find config/ directory")


def load_yaml(name: str) -> dict:
    """Load a YAML file from the config/ directory."""
    path = _find_config_dir() / name
    with open(path) as f:
        return yaml.safe_load(f)


def load_experiment_config() -> dict:
    return load_yaml("experiment.yaml")


def load_model_registry() -> dict:
    return load_yaml("models.yaml")


def load_dataset_registry() -> dict:
    return load_yaml("datasets.yaml")


def model_slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def get_model_config(model_id: str) -> dict:
    """Get model-specific config (d_model, n_layers, etc.)."""
    registry = load_model_registry()
    return registry["models"][model_id]


def get_width_config(model_id: str, width: str) -> dict:
    """Get width-specific SAE config (release, id format, l0 values)."""
    model_cfg = get_model_config(model_id)
    return model_cfg["sae_widths"][width]


def get_sae_id(model_id: str, width: str, layer: int) -> str:
    """Resolve the full SAE ID for a given model/width/layer."""
    wcfg = get_width_config(model_id, width)
    fmt = wcfg["sae_id_fmt"]
    if "{l0}" not in fmt:
        return fmt.format(layer=layer)
    l0 = wcfg.get("layer_l0", {}).get(layer, wcfg.get("default_l0"))
    if l0 is None:
        raise ValueError(f"No L0 value for {model_id} width={width} layer={layer}. "
                         f"Check config/models.yaml and add to layer_l0.")
    return fmt.format(layer=layer, l0=l0)


def get_dataset_config(language: str) -> dict:
    """Get dataset config for a language."""
    registry = load_dataset_registry()
    return registry["languages"][language]


def get_layer_emotion_map(model_id: str) -> Optional[dict]:
    """Get layer-emotion mapping from probing. Returns None if not available."""
    registry = load_model_registry()
    maps = registry.get("layer_emotion_maps", {})
    lem = maps.get(model_id)
    if lem is None:
        return None
    # Ensure integer keys
    return {int(k): v for k, v in lem.items()}


def resolve_layers(model_id: str, mode: str) -> tuple:
    """
    Resolve which layers to use based on mode.

    Returns:
        (unique_layers, layer_emotion_map, emo_to_layer)
    """
    data_reg = load_dataset_registry()
    model_cfg = get_model_config(model_id)

    if mode == "multi":
        lem = get_layer_emotion_map(model_id)
        if lem is None:
            raise ValueError(f"No layer_emotion_map for {model_id} in config/models.yaml")
        unique_layers = sorted(lem.keys())
        emo_to_layer = {}
        for layer, emos in lem.items():
            for e in emos:
                emo_to_layer[e] = layer
        return unique_layers, lem, emo_to_layer
    else:
        # Single-layer: use default_layer for all emotions
        dl = model_cfg.get("default_layer", 0)
        # Get emotion classes from first available language (caller should pass this)
        return [dl], None, None


def resolve_output_dir(base_dir: str, model_id: str, width: str, language: str) -> Path:
    """Resolve the output directory for a specific experiment."""
    slug = model_slug(model_id)
    lang = language.lower().replace(" ", "_")
    return Path(base_dir) / slug / width / lang


def resolve_sae_explns_dir(base_dir: str, model_id: str, width: str) -> Path:
    """
    Resolve the SAE explanations directory.

    Explanations are model+width specific, NOT language specific:
        {output_dir}/{model_slug}/{width}/sae_explanations/
    """
    slug = model_slug(model_id)
    return Path(base_dir) / slug / width / "sae_explanations"


def resolve_sp1_dir(base_dir: Path, cand_method: str) -> tuple:
    """
    Resolve SP-1 subdirectory based on candidate method.

    Returns:
        (sp1_parent, sp1_subdir) — e.g. ("classifier-based", "sp1_classifier")
    """
    if cand_method == "classifier-based":
        return "classifier-based", "sp1_classifier"
    else:
        return "semantic-based", "sp1_semantic"


def should_use_4bit(model_id: str) -> bool:
    """Check if model should use 4-bit quantization."""
    exp_cfg = load_experiment_config()
    force_list = exp_cfg.get("gpu", {}).get("force_4bit_models", [])
    if model_id in force_list:
        return True
    model_cfg = get_model_config(model_id)
    return model_cfg.get("use_4bit", False)