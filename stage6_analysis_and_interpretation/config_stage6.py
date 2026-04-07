"""
config_stage6.py
================
Central configuration for Stage 6: Analysis & Interpretation.

This stage is purely analytical — it consumes outputs from Stages 2–5
and produces dissertation-ready figures, tables, and statistical tests.

SETUP:
1. Ensure Stages 3, 4, 5 are complete (all outputs exist).
2. Update paths below to match your project layout.
3. Update NORM_MEAN / NORM_STD if not already set.
"""

import os
from pathlib import Path

# ===========================================================================
# PATHS
# ===========================================================================

PROJECT_ROOT = Path(__file__).parent.parent

# ── Dataset paths ──────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
PV_ROOT = DATA_DIR/ "plantvillage_raw"
PD_ROOT = DATA_DIR / "plantdoc_raw"

# ── Stage 2 outputs ────────────────────────────────────────────────────────
ST2_OUT = PROJECT_ROOT / "stage2_dataset_preparation" / "outputs"
SPLITS_DIR       = ST2_OUT / "splits"
PREPROCESS_DIR   = ST2_OUT / "preprocessing"
ANALYSIS_DIR  = ST2_OUT / "analysis"

# ── Stage 3 outputs ────────────────────────────────────────────────────────
BASELINES_DIR    = PROJECT_ROOT / "stage3_baseline_establishment/outputs" / "baselines"

# ── Stage 4 outputs ────────────────────────────────────────────────────────
PRETRAIN_DIR     = PROJECT_ROOT / "stage4_leaf_jepa_pretraining/outputs" / "pretraining"

# ── Stage 5 outputs ────────────────────────────────────────────────────────
PEFT_RESULTS_DIR = PROJECT_ROOT / "stage5_peft_adaptation_experiments/outputs" / "peft"

# ── Stage 6 outputs ────────────────────────────────────────────────────────
ST6_OUT       = PROJECT_ROOT / "stage6_analysis_and_interpretation/outputs"
STAGE6_FIGURES   = ST6_OUT / "figures"
STAGE6_TABLES    = ST6_OUT / "tables"
STAGE6_DATA      = ST6_OUT / "data"

# ── Checkpoints ────────────────────────────────────────────────────────────
IJEPA_CHECKPOINT      = PROJECT_ROOT / "stage3_baseline_establishment/checkpoints" / "IN1K-vit.h.14-300e.pth.tar"
LEAF_JEPA_CHECKPOINT  = PROJECT_ROOT / "stage4_leaf_jepa_pretraining/outputs/checkpoints" / "leaf_jepa_encoder.pth"

# If S4_5 masking ablation was run:
LEAF_JEPA_STANDARD_MASKING_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "leaf_jepa_standard_masking.pth"  ## TODO: get standard masking pth generated and saved

# ===========================================================================
# IMMUTABLE CONSTANTS
# ===========================================================================

RANDOM_SEED      = 42
SUBSET_SEEDS     = [42, 123, 456]
LABEL_FRACTIONS  = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
IMAGE_SIZE       = 224
NUM_CLASSES      = 38

# ===========================================================================
# NORMALISATION — UPDATED FROM Stage 2 normalisation_stats.json
# ===========================================================================

NORM_MEAN = [0.466726, 0.488969, 0.41028]
NORM_STD  = [0.195034, 0.170282, 0.213409]

# ===========================================================================
# MODEL ARCHITECTURE
# ===========================================================================

VIT_MODEL_NAME   = "vit_huge_patch14_224"
EMBED_DIM        = 1280
PATCH_SIZE       = 14
NUM_PATCHES      = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 256
GRID_SIZE        = IMAGE_SIZE // PATCH_SIZE          # 16

# ===========================================================================
# ANALYSIS CONFIGURATION
# ===========================================================================

# ── Statistical testing ────────────────────────────────────────────────────
SIGNIFICANCE_ALPHA = 0.05
EFFECT_SIZE_THRESHOLDS = {"small": 0.2, "medium": 0.5, "large": 0.8}

# ── t-SNE / UMAP ──────────────────────────────────────────────────────────
TSNE_PERPLEXITY  = 30
TSNE_N_ITER      = 1000
TSNE_RANDOM_STATE = RANDOM_SEED
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1
PCA_COMPONENTS   = 50       # PCA reduction before t-SNE/UMAP
MAX_SAMPLES_EMBED = 2000    # cap for t-SNE (speed)

# ── Attention maps ─────────────────────────────────────────────────────────
ATTN_SAMPLES_PER_CATEGORY = 3
ATTN_ALPHA       = 0.45     # overlay opacity
ATTN_COLORMAP    = "inferno"
ATTN_TOP_K_PCT   = 0.25     # top 25% patches for IoU

# ── Pareto / radar ────────────────────────────────────────────────────────
RQ3_THRESHOLD_PCT = 2.0     # "within 2% of full fine-tune"

# ── Pathogen taxonomy ─────────────────────────────────────────────────────
# Mapping from PlantVillage class names to pathogen category.
# You may need to update this if your class_names ordering differs.
PATHOGEN_CATEGORIES = {
    "Fungal":    [],  # Fill from Stage 2 taxonomy
    "Bacterial": [],
    "Viral":     [],
    "Abiotic":   [],
    "Healthy":   [],
}

# ===========================================================================
# BASELINE & PEFT METHOD REGISTRY
# ===========================================================================

BASELINE_IDS   = ["B1", "B2", "B3", "B4", "B5"]
BASELINE_NAMES = {
    "B1": "ResNet-50",
    "B2": "EfficientNet-B3",
    "B3": "I-JEPA LP",
    "B4": "I-JEPA Full FT",
    "B5": "Leaf-JEPA LP",
}

PEFT_METHODS = ["LoRA", "Adapter", "VPT-Shallow", "VPT-Deep", "BitFit"]

# Colours for consistent plotting across all figures
METHOD_COLOURS = {
    "B1": "#e74c3c", "B2": "#e67e22", "B3": "#3498db",
    "B4": "#2c3e50", "B5": "#27ae60",
    "LoRA": "#9b59b6", "Adapter": "#1abc9c",
    "VPT-Shallow": "#f39c12", "VPT-Deep": "#d35400",
    "BitFit": "#7f8c8d",
    "Full FT": "#2c3e50", "Linear Probe": "#95a5a6",
}

# ===========================================================================
# WANDB
# ===========================================================================

WANDB_ENTITY  = "muh-haleef02-inform"
WANDB_PROJECT = "leaf-jepa-irp"


# ===========================================================================
# VERIFY
# ===========================================================================

def verify_config():
    """Quick verification of critical paths and settings."""
    print("=" * 60)
    print("  STAGE 6 CONFIG VERIFICATION")
    print("=" * 60)

    checks = [
        ("PLANTVILLAGE_DIR exists",   PV_ROOT.exists()),
        ("SPLITS_DIR exists",         SPLITS_DIR.exists()),
        ("BASELINES_DIR exists",      BASELINES_DIR.exists()),
        ("PEFT_RESULTS_DIR exists",   PEFT_RESULTS_DIR.exists()),
        ("IJEPA_CHECKPOINT exists",   IJEPA_CHECKPOINT.exists()),
        ("LEAF_JEPA_CHECKPOINT exists", LEAF_JEPA_CHECKPOINT.exists()),
        ("NORM_MEAN not default",     NORM_MEAN != [0.485, 0.456, 0.406]),
        ("WANDB_ENTITY set",          WANDB_ENTITY != "YOUR_USERNAME"),
    ]

    ok = 0
    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status}  {name}")
        if passed:
            ok += 1

    # Optional checks
    opt = [
        ("Standard masking checkpoint", LEAF_JEPA_STANDARD_MASKING_CHECKPOINT.exists()),
        ("Stage 2 analysis dir", ANALYSIS_DIR.exists() if ANALYSIS_DIR else False),
    ]
    print("\n  Optional:")
    for name, passed in opt:
        print(f"  {'✅' if passed else '⚠️'}  {name}")

    print(f"\n  {ok}/{len(checks)} required checks passed")

    # Create output dirs
    for d in [ST6_OUT, STAGE6_FIGURES, STAGE6_TABLES, STAGE6_DATA]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"\n  Output directories created at {ST6_OUT}")

    return ok == len(checks)


if __name__ == "__main__":
    verify_config()
