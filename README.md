# Emotion SAE Steering

Cultural feature selection for emotion recognition in low-resource languages using SAE steering vectors.

## Project Structure

```
emotion-sae-steering/
├── config/
│   ├── experiment.yaml      # Which models, widths, modes to run
│   ├── models.yaml           # Model registry, SAE IDs, layer-emotion maps
│   └── datasets.yaml         # Dataset configs, hyperparameters per module
├── src/
│   ├── config.py              # Config loading & resolution
│   ├── data.py                # Dataset loading
│   ├── model.py               # Model loading (with 4-bit support)
│   ├── sae.py                 # SAE loading (pre-trained & custom)
│   ├── steering.py            # Steering context managers
│   ├── prompt.py              # Prompt templates & response parsing
│   ├── evaluation.py          # Metrics & threshold sweep
│   └── utils.py               # Plotting & path utilities
├── scripts/
│   ├── run_layer_probing.py   # Module 1: Layer probing
│   ├── run_sp1_semantic.py    # Module 2: Semantic feature selection
│   ├── run_sp1_classifier.py  # Module 3: Classifier-based feature selection
│   ├── run_sp2_optimization.py # Module 4: Coefficient optimisation
│   └── run_analysis.py        # Module 5: Analysis & ablation
├── outputs/                   # Auto-created results directory
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Modules 1-4: Language-only parameter

These modules iterate over all models and widths defined in `config/experiment.yaml`:

```bash
# 1. Layer probing (no width needed — probes at all layers)
python scripts/run_layer_probing.py --language indonesia

# 2. Semantic feature selection (SP-1)
python scripts/run_sp1_semantic.py --language indonesia

# 3. Classifier-based feature selection
python scripts/run_sp1_classifier.py --language indonesia

# 4. Coefficient optimisation (SP-2)
python scripts/run_sp2_optimization.py --language indonesia
```

### Module 5: Analysis (requires specific parameters)

```bash
python scripts/run_analysis.py \
    --language indonesia \
    --model google/gemma-2-2b \
    --width 16k \
    --mode multi \
    --cand_method semantic
```

## Output Directory Structure

```
outputs/
└── gemma-2-2b/
    ├── probing/indonesia/
    │   ├── probe_results.csv
    │   ├── probing_summary.json
    │   └── plots/
    │       ├── probing_f1.png
    │       └── probing_heatmap.png
    ├── 16k/indonesia/
    │   ├── sp1/layer_7/f_cand.json
    │   ├── sp1_activation/layer_7/f_cand.json
    │   ├── semantic/sp2_multilayer/
    │   │   ├── alpha_star.json
    │   │   ├── train_history.json
    │   │   ├── plots/
    │   │   │   ├── training_curves.png
    │   │   │   ├── threshold_sweep.png
    │   │   │   └── score_distributions.png
    │   │   └── analysis/
    │   │       ├── ablation_results.json
    │   │       └── plots/
    │   └── activation-based/sp2_multilayer/
    │       └── ...
    └── 65k/indonesia/
        └── ...
```

## Configuration

### Adding a new language

Edit `config/datasets.yaml`:

```yaml
languages:
  chinese:
    hf_dataset_id: brighter-dataset/BRIGHTER-emotion-categories
    hf_subset: zho
    text_column: text
    emotion_classes: [anger, disgust, fear, joy, sadness, surprise, neutral]
```

Then run: `python scripts/run_layer_probing.py --language chinese`

### Adding a new model

Edit `config/models.yaml` with the model's d_model, n_layers, SAE widths, and layer_l0 values.

### Changing hyperparameters

Edit `config/datasets.yaml` under the `sp2:`, `sp1:`, `probing:`, or `activation_selection:` sections.
