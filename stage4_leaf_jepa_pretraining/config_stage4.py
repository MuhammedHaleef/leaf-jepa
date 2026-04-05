"""
config_stage4.py
================
Central configuration for Stage 4: Leaf-JEPA Domain Pretraining.

SETUP INSTRUCTIONS (in order):
1. Copy config_stage3.py values for NORM_MEAN/NORM_STD, IJEPA_CHECKPOINT, WANDB_ENTITY
2. Set PT_EPOCHS to match your compute budget (50 = minimum, 150 = recommended)
3. Optionally enable ENABLE_BIASED_MASKING (True = novel contribution, False = ablation baseline)
4. After Stage 4 completes, update LEAF_JEPA_CHECKPOINT in config_stage3.py with the
   path produced by S4_6_checkpoint_and_export.ipynb

Do NOT modify RANDOM_SEED, SUBSET_SEEDS, LABEL_FRACTIONS, IMAGE_CROP/RESIZE, NORM_MEAN/NORM_STD,
PATCH_SIZE, or PLANTDOC_DIR. These are immutable constants fixed in Stage 2.
"""

import os
from pathlib import Path

# ===========================================================================
# PATHS
# ===========================================================================

PROJECT_ROOT = Path(__file__).parent.parent

DATA_ROOT        = PROJECT_ROOT / "data"                # need to change to processed data dir
OUTPUT_DIR = PROJECT_ROOT / "stage4_leaf_jepa_pretraining/outputs"
SPLITS_DIR       = PROJECT_ROOT / "stage2_dataset_preparation/outputs" / "splits"
PREPROC_DIR      = PROJECT_ROOT / "stage2_dataset_preparation/outputs" / "preprocessing"
BASELINES_DIR    = PROJECT_ROOT / "stage3_baseline_establishment/outputs" / "baselines"
FIGURES_DIR      = PROJECT_ROOT / "stage4_leaf_jepa_pretraining/outputs" / "figures"

# Stage 4 specific outputs
PRETRAIN_DIR     = OUTPUT_DIR / "pretraining"      # loss curves, checkpoints
PRETRAIN_CKPT_DIR= OUTPUT_DIR / "checkpoints" / "stage4"       # epoch checkpoints
LEAF_JEPA_DIR    = OUTPUT_DIR / "checkpoints"                  # final exported encoder

PLANTVILLAGE_DIR = DATA_ROOT / "plantvillage_raw"
PLANTDOC_DIR     = DATA_ROOT / "plantdoc_raw"

# ===========================================================================
# CHECKPOINTS
# ===========================================================================

# Source checkpoint (Meta I-JEPA, ViT-H/14, 300 epochs ImageNet-1K)
# Download: https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar
IJEPA_CHECKPOINT = PROJECT_ROOT / "stage3_baseline_establishment/checkpoints" / "IN1K-vit.h.14-300e.pth.tar"

# Output: best Leaf-JEPA encoder (written by S4_6_checkpoint_and_export.ipynb)
# After Stage 4, copy this path into config_stage3.py as LEAF_JEPA_CHECKPOINT
LEAF_JEPA_CHECKPOINT = LEAF_JEPA_DIR / "leafjepa-vit-h14-best.pth"

# ===========================================================================
# IMMUTABLE CONSTANTS — DO NOT CHANGE (fixed in Stage 2)
# ===========================================================================

RANDOM_SEED     = 42
SUBSET_SEEDS    = [42, 123, 456]
LABEL_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

NUM_CLASSES      = 38
IMAGE_RESIZE     = 256
IMAGE_CROP       = 224
PATCH_SIZE       = 14          # ViT-H/14 patch size

# NORMALISATION: Updated from outputs/preprocessing/normalisation_stats.json
NORM_MEAN = [0.465809, 0.487659, 0.409572]
NORM_STD = [0.19489, 0.169946, 0.213739]

# ===========================================================================
# MODEL ARCHITECTURE
# ===========================================================================

VIT_MODEL_NAME  = "vit_huge_patch14_224"   # timm model name
VIT_EMBED_DIM   = 1280                      # ViT-H/14 embedding dim
VIT_NUM_HEADS   = 16                        # ViT-H/14 attention heads
VIT_DEPTH       = 32                        # ViT-H/14 transformer blocks
VIT_LIGHT_NAME  = "vit_base_patch16_224"   # VRAM-constrained fallback (embed_dim=768)
VIT_LIGHT_DIM   = 768

# Number of patches per spatial axis at IMAGE_CROP=224, PATCH_SIZE=14
NUM_PATCHES     = (IMAGE_CROP // PATCH_SIZE) ** 2   # 16x16 = 256 patches for ViT-H/14

# ===========================================================================
# PREDICTOR ARCHITECTURE
# ===========================================================================
# Re-initialised from scratch — never loaded from Meta checkpoint.
# Narrow by design: forces encoder to learn real abstractions (not shortcuts).

PRED_DEPTH       = 4      # transformer layers
PRED_EMBED_DIM   = 256    # hidden dimension (narrow bottleneck)
PRED_NUM_HEADS   = 4      # attention heads
PRED_MLP_RATIO   = 4.0    # FFN expansion ratio
PRED_DROPOUT     = 0.1

# ===========================================================================
# PRETRAINING HYPERPARAMETERS
# ===========================================================================

# --- Training duration ---
PT_EPOCHS           = 50 #150    # Recommended: 100–200. Minimum: 50 (document as limitation)
PT_BATCH_SIZE       = 16 #128    # Reduce to 64 if CUDA OOM; 32 if using ViT-H on 16GB GPU
PT_ACCUMULATE_GRAD  = 1      # Gradient accumulation steps (increase if small batch)

# --- Learning rates (layer-wise) ---
PT_LR_HEAD          = 3e-4   # Predictor (training from scratch)
PT_LR_ENCODER_TOP   = 1e-4   # Encoder layers 10–12 (standard adaptation)
PT_LR_ENCODER_MID   = 1e-5   # Encoder layers 5–9  (slow adaptation)
PT_LR_ENCODER_BOT   = 0.0    # Encoder layers 1–4  (FROZEN)
PT_WEIGHT_DECAY     = 0.04
PT_WARMUP_EPOCHS    = 10     # Linear LR warmup before cosine decay

# --- EMA (Exponential Moving Average) ---
EMA_TAU_START       = 0.996  # Initial target encoder momentum
EMA_TAU_END         = 0.999  # Final target encoder momentum (cosine schedule)

# --- Masking parameters ---
# Context block
PT_CONTEXT_SCALE    = (0.85, 1.0)   # Context block covers 85–100% of patches
PT_CONTEXT_RATIO    = (0.75, 1.5)   # Context block aspect ratio range

# Target blocks
PT_NUM_TARGET_BLOCKS = 4             # Number of target blocks per image
PT_TARGET_SCALE      = (0.15, 0.20) # Each target covers 15–20% of patches
PT_TARGET_RATIO      = (0.75, 1.5)  # Target aspect ratio range

# Disease-region-biased masking (novel contribution)
# Set True for the main run. Set False for ablation baseline (S4_AB_masking_ablation.ipynb)
ENABLE_BIASED_MASKING   = True
SALIENCY_BIAS_STRENGTH  = 5.0       # Temperature for saliency-weighted sampling (higher = more biased)
HEALTHY_HUE_CENTER      = 0.3153 #0.30      # calculated hue of healthy leaf (green in [0,1] HSV)
HEALTHY_HUE_SIGMA       = 0.690 #0.10      # Width of healthy hue distribution

# --- Loss ---
PT_LOSS             = "smooth_l1"   # "mse" or "smooth_l1" (smooth_l1 more robust to outliers)

# ===========================================================================
# PERIODIC LINEAR PROBE MONITORING
# ===========================================================================
# Every LP_MONITOR_INTERVAL epochs, freeze encoder and train a quick linear head
# to measure representation quality directly (not just pretraining loss).
# This is the primary convergence criterion — pretraining stops when LP score plateaus.

LP_MONITOR_INTERVAL = 25    # Run linear probe every N epochs
LP_MONITOR_EPOCHS   = 5     # Epochs for each monitoring linear probe
LP_MONITOR_FRAC     = 0.2   # Fraction of training data for quick LP training

# ===========================================================================
# CHECKPOINTING
# ===========================================================================

CKPT_SAVE_INTERVAL  = 25    # Save checkpoint every N epochs
# Best checkpoint is selected by LP_monitor_f1 (not pretraining loss)

# ===========================================================================
# MIXED PRECISION & DATALOADER
# ===========================================================================

USE_AMP      = True
NUM_WORKERS  = 0  #4
PIN_MEMORY   = True

# ===========================================================================
# WANDB
# ===========================================================================

WANDB_PROJECT = "leaf-jepa-irp"
WANDB_ENTITY  = "muh-haleef02"

def wandb_pretrain_run_name(extra: str = "") -> str:
    base = f"LeafJEPA-pretrain-vit-h14-{PT_EPOCHS}e"
    return f"{base}-{extra}" if extra else base

# ===========================================================================
# LAYER GROUPS (used by get_layerwise_optimizer)
# Indices refer to ViT-H/14 transformer blocks (0-indexed, 32 total)
# ===========================================================================

FROZEN_LAYERS   = list(range(0,  4))   # blocks 0–3:  frozen completely
LOW_LR_LAYERS   = list(range(4,  9))   # blocks 4–8:  PT_LR_ENCODER_MID
STD_LR_LAYERS   = list(range(9, 32))   # blocks 9–31: PT_LR_ENCODER_TOP

# ===========================================================================
# SANITY CHECK
# ===========================================================================

def verify_config():
    """Call at start of each Stage 4 notebook."""
    import json
    issues = []

    if not SPLITS_DIR.exists():
        issues.append(f"SPLITS_DIR not found: {SPLITS_DIR}  (run Stage 2 first)")

    norm_path = PREPROC_DIR / "normalisation_stats.json"
    if norm_path.exists():
        if abs(NORM_MEAN[0] - 0.4611) < 1e-4:
            print("⚠️  WARNING: NORM_MEAN still appears to be placeholder value.")
            print("   Copy values from normalisation_stats.json into config_stage4.py")
    else:
        issues.append(f"normalisation_stats.json not found: {norm_path}")

    if not IJEPA_CHECKPOINT.exists():
        issues.append(f"I-JEPA checkpoint not found: {IJEPA_CHECKPOINT}")
        issues.append("  Download: https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar")

    if WANDB_ENTITY == "YOUR_WANDB_USERNAME":
        issues.append("WANDB_ENTITY not set in config_stage4.py")

    if issues:
        print("❌ CONFIG ISSUES FOUND:")
        for i in issues:
            print(f"   • {i}")
        return False
    else:
        print("✅ Stage 4 config verified.")
        return True


if __name__ == "__main__":
    verify_config()
