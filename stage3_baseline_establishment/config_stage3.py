"""
config_stage3.py
================
Central configuration for all Stage 3 baseline experiments.

SETUP INSTRUCTIONS:
1. Update NORM_MEAN and NORM_STD from outputs/preprocessing/normalisation_stats.json
2. Update IJEPA_CHECKPOINT to point to your downloaded checkpoint
3. Update WANDB_ENTITY to your WandB username
4. LEAF_JEPA_CHECKPOINT will be filled after Stage 4 completes
"""

import os
import json
from pathlib import Path

# ===========================================================================
# PATHS
# ===========================================================================
# PROJECT_ROOT = Path(os.getenv("LEAF_JEPA_ROOT", ".")).resolve()
PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR         = PROJECT_ROOT / "data"
PLANTVILLAGE_DIR = DATA_DIR / "plantvillage_raw"
PLANTDOC_DIR     = DATA_DIR / "plantdoc_processed"

OUTPUTS_DIR      = PROJECT_ROOT / "stage3_baseline_establishment/outputs"
SPLITS_DIR       = PROJECT_ROOT / "stage2_dataset_preparation/outputs/splits"
BASELINES_DIR    = OUTPUTS_DIR / "baselines"
FIGURES_DIR      = OUTPUTS_DIR / "figures"
CHECKPOINTS_DIR  = PROJECT_ROOT / "stage3_baseline_establishment/checkpoints"

# ===========================================================================
# CHECKPOINTS
# ===========================================================================
# DownloadED from: https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar
IJEPA_CHECKPOINT = CHECKPOINTS_DIR / "IN1K-vit.h.14-300e.pth.tar"

# ── UPDATE THIS after Stage 4 completes ──
LEAF_JEPA_CHECKPOINT = None  # e.g. CHECKPOINTS_DIR / "leaf_jepa_target_encoder.pth"

# ===========================================================================
# MODEL CONSTANTS
# ===========================================================================
VIT_MODEL_NAME = "vit_huge_patch14_224"  # timm model string
EMBED_DIM      = 1280                     # ViT-H/14 embedding dimension
NUM_CLASSES    = 38                       # PlantVillage class count

# ===========================================================================
# NORMALISATION — UPDATED FROM Stage 2 normalisation_stats.json
# ===========================================================================

NORM_MEAN = [0.466726, 0.488969, 0.41028]
NORM_STD  = [0.195034, 0.170282, 0.213409]

# ===========================================================================
# IMAGE PIPELINE
# ===========================================================================
IMAGE_RESIZE = 256
IMAGE_CROP   = 224

# ===========================================================================
# IMMUTABLE CONSTANTS
# ===========================================================================
RANDOM_SEED      = 42
SUBSET_SEEDS     = [42, 123, 456]
# LABEL_FRACTIONS  = [0.50, 1.00]
LABEL_FRACTIONS  = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15
TEST_RATIO       = 0.15

# ===========================================================================
# TRAINING HYPERPARAMETERS — CNN baselines (B1, B2)
# ===========================================================================
CNN_BATCH_SIZE   = 64 # reduce to 32 if OOM (local device)
CNN_LR           = 1e-4
CNN_HEAD_LR      = 1e-3
CNN_WEIGHT_DECAY = 0.01
CNN_EPOCHS       = 50 #2 (local device)
CNN_PATIENCE     = 7 #2 (local device)
CNN_GRAD_ACCUM   = 1

# ===========================================================================
# TRAINING HYPERPARAMETERS — I-JEPA linear probe (B3, B5)
# ===========================================================================
LP_BATCH_SIZE    = 256
LP_LR            = 1e-3
LP_WEIGHT_DECAY  = 0.0
LP_EPOCHS        = 100 #5 (local device)
LP_PATIENCE      = 10 #3 (local device)

# ===========================================================================
# TRAINING HYPERPARAMETERS — I-JEPA full fine-tune (B4)
# ===========================================================================
FT_BATCH_SIZE    = 16     # small due to 632M params
FT_LR            = 5e-5   # backbone LR
FT_HEAD_LR       = 1e-3
FT_WEIGHT_DECAY  = 0.05
FT_EPOCHS        = 25 #2 (local device)
FT_PATIENCE      = 5 #2 (local device)
FT_GRAD_ACCUM    = 4      # effective batch = 16 * 4 = 64
FT_LR_DECAY      = 0.70   # layerwise LR decay factor
FT_GRAD_CKPT     = True   # gradient checkpointing to save VRAM

# ===========================================================================
# WANDB
# ===========================================================================
WANDB_PROJECT = "leaf-jepa-irp"
WANDB_ENTITY  = "muh-haleef02-inform"

# ===========================================================================
# CLEANLAB
# ===========================================================================
CLEANLAB_QUALITY_THRESHOLD = 0.1  # samples below this are flagged as mislabelled

# ===========================================================================
# VERIFY CONFIG
# ===========================================================================
def verify_config():
    """Run basic sanity checks. Call this at the start of every notebook."""
    warnings = []
    errors   = []

    # Check normalisation stats are not ImageNet defaults
    if NORM_MEAN == [0.485, 0.456, 0.406] and NORM_STD == [0.229, 0.224, 0.225]:
        warnings.append(
            "NORM_MEAN/NORM_STD are still ImageNet defaults! "
            "Update from outputs/preprocessing/normalisation_stats.json"
        )

    # Check checkpoint exists
    if not IJEPA_CHECKPOINT.exists():
        errors.append(
            f"I-JEPA checkpoint not found at {IJEPA_CHECKPOINT}. "
            "Download from https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar"
        )

    # Check splits directory
    if not SPLITS_DIR.exists():
        errors.append(f"Splits directory not found at {SPLITS_DIR}. Run Stage 2 first.")

    # Check WandB entity
    if WANDB_ENTITY is None:
        warnings.append("WANDB_ENTITY is None — set to your WandB username.")

    # Check PlantVillage exists
    if not PLANTVILLAGE_DIR.exists():
        errors.append(f"PlantVillage dataset not found at {PLANTVILLAGE_DIR}")

    # Print results
    for w in warnings:
        print(f"  \u26a0\ufe0f  WARNING: {w}")
    for e in errors:
        print(f"  \u274c  ERROR: {e}")

    if not warnings and not errors:
        print("  \u2705  All config checks passed.")
    elif errors:
        print(f"\n  {len(errors)} error(s) must be fixed before running.")
    else:
        print(f"\n  {len(warnings)} warning(s) — experiments will run but results may not be optimal.")

    return len(errors) == 0
