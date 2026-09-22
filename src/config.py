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


_probing_cache = {}

def load_probing_results(language: str) -> dict:
    """
    Load optional per-language probing overrides from
    config/probing_results/{language}.yaml.

    These override models.yaml when present. If the file doesn't exist,
    models.yaml values are used as-is (the normal workflow of editing
    models.yaml directly).
    """
    if language in _probing_cache:
        return _probing_cache[language]

    path = _find_config_dir() / "probing_results" / f"{language}.yaml"
    if not path.exists():
        print(f"  [config] No probing override at {path} — using models.yaml")
        _probing_cache[language] = {}
        return {}

    data = yaml.safe_load(open(path)) or {}
    print(f"  [config] Loaded probing override: {path}")
    if "default_layers" in data:
        print(f"           default_layers: {data['default_layers']}")
    if "layer_emotion_maps" in data:
        models_listed = list(data["layer_emotion_maps"].keys())
        print(f"           layer_emotion_maps for: {models_listed}")
    _probing_cache[language] = data
    return data


def model_slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def get_model_config(model_id: str, language: str = None) -> dict:
    """
    Get model-specific config.

    Primary source: models.yaml (always read).
    Optional override: probing_results/{language}.yaml overrides default_layer
    when present.
    """
    registry = load_model_registry()
    cfg = dict(registry["models"][model_id])  # shallow copy

    if language:
        probing = load_probing_results(language)
        dl = probing.get("default_layers", {}).get(model_id)
        if dl is not None:
            cfg["default_layer"] = dl

    return cfg


def get_width_config(model_id: str, width: str, language: str = None) -> dict:
    """
    Get width-specific SAE config.

    Primary source: models.yaml layer_l0 (always read).
    Optional override: probing_results/{language}.yaml layer_l0_overrides
    are merged on top when present.
    """
    model_cfg = get_model_config(model_id)  # base config, no language
    wcfg = dict(model_cfg["sae_widths"][width])  # shallow copy
    base_l0 = dict(wcfg.get("layer_l0") or {})

    if language:
        probing = load_probing_results(language)
        overrides = (probing.get("layer_l0_overrides", {})
                     .get(model_id, {}).get(width, {}))
        for layer, l0 in overrides.items():
            base_l0[int(layer)] = l0

    wcfg["layer_l0"] = base_l0
    return wcfg


def get_sae_id(model_id: str, width: str, layer: int, language: str = None) -> str:
    """Resolve the full SAE ID for a given model/width/layer."""
    wcfg = get_width_config(model_id, width, language=language)
    fmt = wcfg["sae_id_fmt"]
    if "{l0}" not in fmt:
        return fmt.format(layer=layer)
    l0 = wcfg.get("layer_l0", {}).get(layer, wcfg.get("default_l0"))
    if l0 is None:
        raise ValueError(
            f"No L0 value for {model_id} width={width} layer={layer}. "
            f"Add it to layer_l0 in config/models.yaml, or run:\n"
            f"  python scripts/discover_l0.py --model {model_id} --width {width} --layers {layer}")
    return fmt.format(layer=layer, l0=l0)


def get_dataset_config(language: str) -> dict:
    """Get dataset config for a language."""
    registry = load_dataset_registry()
    return registry["languages"][language]


def get_layer_emotion_map(model_id: str, language: str = None) -> Optional[dict]:
    """
    Get layer-emotion mapping.

    Primary source: models.yaml layer_emotion_maps.
    Optional override: probing_results/{language}.yaml takes priority when present.
    """
    # Check optional override first
    if language:
        probing = load_probing_results(language)
        lem = probing.get("layer_emotion_maps", {}).get(model_id)
        if lem is not None:
            return {int(k): v for k, v in lem.items()}

    # Primary source: models.yaml
    registry = load_model_registry()
    maps = registry.get("layer_emotion_maps", {})
    lem = maps.get(model_id)
    if lem is None:
        return None
    return {int(k): v for k, v in lem.items()}


def resolve_layers(model_id: str, mode: str, language: str = None) -> tuple:
    """
    Resolve which layers to use based on mode and language.

    Returns:
        (unique_layers, layer_emotion_map, emo_to_layer)
    """
    model_cfg = get_model_config(model_id, language=language)

    if mode == "multi":
        lem = get_layer_emotion_map(model_id, language=language)
        if lem is None:
            raise ValueError(
                f"No layer_emotion_map for {model_id}. "
                f"Add it to config/models.yaml or config/probing_results/{language}.yaml"
            )
        unique_layers = sorted(lem.keys())
        emo_to_layer = {}
        for layer, emos in lem.items():
            for e in emos:
                emo_to_layer[e] = layer
        return unique_layers, lem, emo_to_layer
    else:
        dl = model_cfg.get("default_layer", 0)
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
    """
    Check if model should use 4-bit quantization.

    Resolution order:
      1. gpu.use_4bit in experiment.yaml — master switch.
         If False, 4-bit is OFF for ALL models (force list ignored).
         If True (default), continue to step 2.
      2. gpu.force_4bit_models — if the model is listed here, use 4-bit.
      3. models.yaml per-model use_4bit field — final fallback.
    """
    exp_cfg = load_experiment_config()
    gpu_cfg = exp_cfg.get("gpu", {})

    # Master switch — if globally disabled, no model uses 4-bit
    if not gpu_cfg.get("use_4bit", True):
        return False

    # Force list — these models always use 4-bit (when master switch is on)
    force_list = gpu_cfg.get("force_4bit_models", [])
    if model_id in force_list:
        return True

    # Per-model default from models.yaml
    model_cfg = get_model_config(model_id)
    return model_cfg.get("use_4bit", False)