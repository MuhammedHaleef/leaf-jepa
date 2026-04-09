"""
config_stage5.py
================
Central configuration for all Stage 5 PEFT Adaptation Experiments.

SETUP INSTRUCTIONS:
1. Update NORM_MEAN and NORM_STD from outputs/preprocessing/normalisation_stats.json
2. Update LEAF_JEPA_CHECKPOINT to your Stage 4 exported encoder
3. Update IJEPA_CHECKPOINT to the downloaded Meta checkpoint
4. Update WANDB_ENTITY to your WandB username
5. Adjust BATCH_SIZE if CUDA OOM (ViT-H/14 + PEFT typically fits in 16GB at bs=32)

Do NOT modify RANDOM_SEED, SUBSET_SEEDS, LABEL_FRACTIONS, or split ratios.
"""

import os
import json
from pathlib import Path

# ===========================================================================
# PATHS
# ===========================================================================

PROJECT_ROOT = Path(__file__).parent.parent

# Dataset roots
DATA_DIR = PROJECT_ROOT / "data"
PV_ROOT = DATA_DIR/ "plantvillage_raw"
PD_ROOT = DATA_DIR / "plantdoc_raw"

# Stage 2 outputs (read-only)
ST2_OUT = PROJECT_ROOT / "stage2_dataset_preparation" / "outputs"
SPLITS_DIR       = ST2_OUT / "splits"
PREPROCESS_DIR   = ST2_OUT / "preprocessing"
NORM_STATS_PATH  = ST2_OUT / "preprocessing/normalisation_stats.json"
CLASS_WEIGHTS_PATH = ST2_OUT / "analysis/plantvillage_class_weights.json"
TAXONOMY_PATH    = ST2_OUT / "taxonomy.json"

# Stage 3 outputs (read-only — reference baselines)
ST3_OUT = PROJECT_ROOT / "stage3_baseline_establishment" / "outputs"
BASELINE_DIR     = ST3_OUT / "baselines"

# Checkpoints
IJEPA_CHECKPOINT      = PROJECT_ROOT / "stage3_baseline_establishment/checkpoints" / "IN1K-vit.h.14-300e.pth.tar"
ST4_OUT = PROJECT_ROOT / "stage4_leaf_jepa_pretraining" / "outputs"
LEAF_JEPA_CHECKPOINT   = ST4_OUT / "checkpoints" / "leafjepa-vit-h14-best.pth"
# ^^^ Update after Stage 4 export — path printed by S4_6_checkpoint_export.ipynb

# Stage 5 outputs
OUTPUT_DIR = PROJECT_ROOT / "stage5_peft_adaptation_experiments/outputs"
PEFT_DIR    = OUTPUT_DIR / "peft"
FIGURES_DIR = OUTPUT_DIR /  "figures"
CKPT_DIR    = OUTPUT_DIR / "peft" / "checkpoints"

# ===========================================================================
# IMMUTABLE CONSTANTS
# ===========================================================================

RANDOM_SEED     = 42
SUBSET_SEEDS    = [42, 123, 456]
LABEL_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
NUM_CLASSES     = 38
IMAGE_RESIZE    = 256
IMAGE_CROP      = 224

# ===========================================================================
# NORMALISATION — UPDATED FROM Stage 2 normalisation_stats.json
# ===========================================================================

NORM_MEAN = [0.466726, 0.488969, 0.41028]
NORM_STD  = [0.195034, 0.170282, 0.213409]

# ===========================================================================
# MODEL ARCHITECTURE
# ===========================================================================

VIT_MODEL_NAME = "vit_huge_patch14_224"
VIT_EMBED_DIM  = 1280
VIT_DEPTH      = 32       # Number of transformer blocks in ViT-H/14
VIT_PATCH_SIZE = 14

# ===========================================================================
# TRAINING HYPERPARAMETERS (shared across ALL PEFT methods)
# ===========================================================================

BATCH_SIZE          = 32
NUM_WORKERS         = 4
MAX_EPOCHS          = 50
EARLY_STOP_PATIENCE = 10
EARLY_STOP_METRIC   = "val_macro_f1"
EARLY_STOP_MODE     = "max"

OPTIMIZER           = "AdamW"
WEIGHT_DECAY        = 0.01
SCHEDULER           = "cosine"
WARMUP_FRACTION     = 0.10    # Linear warmup for first 10% of steps

USE_AMP             = True    # Mixed precision (fp16)
GRADIENT_CLIP       = 1.0     # Max grad norm (None to disable)

# ===========================================================================
# PER-METHOD LEARNING RATES (determined by ST5.3 LR sweep)
# Update these after running the LR sweep in S1_method_comparison.ipynb
# ===========================================================================

LR_SWEEP_VALUES = [1e-4, 3e-4, 1e-3]   # Candidates for the LR sweep

# Best LR per method — fill in after LR sweep completes
BEST_LR = {
    "lora":         3e-4,    # Placeholder — update after LR sweep
    "adapter":      3e-4,    # Placeholder — update after LR sweep
    "vpt_shallow":  3e-4,    # Placeholder — update after LR sweep
    "vpt_deep":     3e-4,    # Placeholder — update after LR sweep
    "bitfit":       1e-3,    # Placeholder — update after LR sweep
    "full_ft":      1e-4,    # From Stage 3 B4
    "linear_probe": 1e-3,    # From Stage 3 B3/B5
}

# ===========================================================================
# PEFT METHOD CONFIGURATIONS
# ===========================================================================

# Set 1: Main comparison sweep
LORA_RANKS            = [4, 8, 16]
ADAPTER_DIMS          = [8, 16, 64]
VPT_SHALLOW_LENGTHS   = [1, 5, 10, 50]
VPT_DEEP_LENGTH       = 50

# Set 4: Extended sensitivity sweep
LORA_RANKS_EXTENDED   = [1, 2, 4, 8, 16, 32]
ADAPTER_DIMS_EXTENDED = [4, 8, 16, 32, 64, 128]
VPT_LENGTHS_EXTENDED  = [1, 2, 5, 10, 20, 50, 100]

# LoRA specifics
LORA_ALPHA    = None   # None => alpha = rank (standard default)
LORA_DROPOUT  = 0.0
LORA_TARGETS  = ["qkv"]  # Which projection(s) to inject into
# timm ViT uses a fused "qkv" linear — we split into Q, V internally

# ===========================================================================
# CROSS-DOMAIN (Set 3)
# ===========================================================================

KNN_K_VALUES = [5, 10, 20]
PD_MIN_SAMPLES = 10   # Exclude PlantDoc classes with < this many samples

# ===========================================================================
# LAYER ABLATION (Set 5)
# ===========================================================================

# ViT-H/14 has blocks 0–31
LAYER_GROUPS = {
    "early":  list(range(0, 11)),    # Blocks 0–10
    "middle": list(range(11, 21)),   # Blocks 11–20
    "late":   list(range(21, 32)),   # Blocks 21–31
    "all":    list(range(0, 32)),    # All blocks
}

# ===========================================================================
# WANDB
# ===========================================================================

WANDB_ENTITY  = "muh-haleef02-inform"
WANDB_PROJECT = "leaf-jepa-irp"

WANDB_GROUPS = {
    "set1": "Set1-MethodComparison",
    "set2": "Set2-LabelEfficiency",
    "set3": "Set3-CrossDomain",
    "set4": "Set4-Sensitivity",
    "set5": "Set5-LayerAblation",
    "set6": "Set6-Combinations",
    "lr_sweep": "Set1-LRSweep",
}

# ===========================================================================
# HELPER: build WandB run name
# ===========================================================================

def wandb_run_name(method: str, hp_tag: str, fraction: float, seed: int) -> str:
    """Generate consistent WandB run name.
    
    Examples:
        S5-LoRA-r8-frac1.00-seed42
        S5-VPT-shallow-p50-frac0.10-seed123
        S5-Adapter-d64-frac0.05-seed456
    """
    return f"S5-{method}-{hp_tag}-frac{fraction:.2f}-seed{seed}"

# ===========================================================================
# VERIFICATION
# ===========================================================================

def verify_config():
    """Run pre-flight checks. Call at the start of every notebook."""
    print("=" * 60)
    print("Stage 5 Configuration Verification")
    print("=" * 60)
    
    checks_passed = 0
    checks_total  = 0
    
    def check(name, condition, detail=""):
        nonlocal checks_passed, checks_total
        checks_total += 1
        status = "✅" if condition else "❌"
        if condition:
            checks_passed += 1
        print(f"  {status}  {name}")
        if detail and not condition:
            print(f"       → {detail}")
    
    # Dataset paths
    check("PlantVillage root exists", PV_ROOT.exists(),
          f"Expected: {PV_ROOT}")
    check("PlantDoc root exists", PD_ROOT.exists(),
          f"Expected: {PD_ROOT}")
    
    # Stage 2 outputs
    check("Splits directory exists", SPLITS_DIR.exists(),
          f"Expected: {SPLITS_DIR}")
    check("Normalisation stats exist", NORM_STATS_PATH.exists(),
          f"Expected: {NORM_STATS_PATH}")
    
    # Normalisation sanity check
    if NORM_STATS_PATH.exists():
        stats = json.loads(NORM_STATS_PATH.read_text())
        expected_mean = stats.get("mean", stats.get("channel_means", NORM_MEAN))
        expected_std  = stats.get("std", stats.get("channel_stds", NORM_STD))
        mean_match = all(abs(a - b) < 1e-4 for a, b in zip(NORM_MEAN, expected_mean))
        std_match  = all(abs(a - b) < 1e-4 for a, b in zip(NORM_STD, expected_std))
        check("NORM_MEAN matches JSON", mean_match,
              f"config={NORM_MEAN}, json={expected_mean}")
        check("NORM_STD matches JSON", std_match,
              f"config={NORM_STD}, json={expected_std}")
    
    # Checkpoints
    check("I-JEPA checkpoint exists", IJEPA_CHECKPOINT.exists(),
          f"Download from: https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar")
    check("Leaf-JEPA checkpoint exists", LEAF_JEPA_CHECKPOINT.exists(),
          f"Run Stage 4 first, then update LEAF_JEPA_CHECKPOINT in config_stage5.py")
    
    # Stage 3 baselines
    check("Baselines directory exists", BASELINE_DIR.exists(),
          f"Expected: {BASELINE_DIR}")
    
    # WandB
    check("WANDB_ENTITY set", WANDB_ENTITY != "your-username",
          "Update WANDB_ENTITY in config_stage5.py")
    
    # Summary
    print(f"\n  {checks_passed}/{checks_total} checks passed")
    if checks_passed < checks_total:
        print("  ⚠️  Fix failing checks before running experiments!")
    else:
        print("  🚀  All checks passed — ready to run Stage 5")
    print("=" * 60)
    return checks_passed == checks_total


if __name__ == "__main__":
    verify_config()
