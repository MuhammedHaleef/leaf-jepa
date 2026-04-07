"""
config_stage2.py
=========
Central configuration for Leaf-JEPA Stage 2: Dataset Preparation.

All paths, constants, random seeds, and dataset parameters are defined here.

"""

import os
from pathlib import Path

# ─── REPRODUCIBILITY ──────────────────────────────────────────────────────────
# A single fixed seed governs ALL random operations: splits, subsets, sampling.
RANDOM_SEED = 42

# ─── PROJECT PATHS ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"


ST2_DIR = PROJECT_ROOT / "stage2_dataset_preparation"
OUTPUTS_DIR  = ST2_DIR /  "outputs"
ANALYSIS_OUT = OUTPUTS_DIR / "analysis"
SPLITS_DIR   = OUTPUTS_DIR / "splits"
PREPROC_OUT   = OUTPUTS_DIR / "preprocessing"
FRACTIONS_DIR = SPLITS_DIR / "fractions"
NSHOT_DIR = SPLITS_DIR / "nshots"
AUG_OUT      = OUTPUTS_DIR / "augmentation"
LOGS_DIR     = OUTPUTS_DIR / "logs"
QUALITY_OUT  = OUTPUTS_DIR / "quality_verification"
TAXONOMY_FILE = OUTPUTS_DIR / "taxonomy.json"



# Raw dataset locations — update these to match your local filesystem
PLANTVILLAGE_RAW_DIR = DATA_DIR / "plantvillage_raw"
PLANTDOC_RAW_DIR     = DATA_DIR / "plantdoc_raw"

# Processed dataset locations (after preprocessing pipeline)
PLANTVILLAGE_PROC_DIR = DATA_DIR / "plantvillage_processed"
PLANTDOC_PROC_DIR     = DATA_DIR / "plantdoc_processed"

FIGURE_DPI  = 150
VALID_EXT   = {".jpg", ".jpeg", ".png"}

# ─── IMAGE PREPROCESSING ──────────────────────────────────────────────────────
# Justified in Subtask 4 (preprocessing pipeline).
# Resolution: 256→224 centre-crop (standard ViT preprocessing protocol).
# used by I-JEPA (Assran et al., 2023) and consistent with our backbone.

IMAGE_RESIZE   = 256   # Resize shorter edge to this before crop
IMAGE_CROP     = 224   # Final crop size — matches ViT patch grid expectations
IMAGE_CHANNELS = 3

# Normalisation statistics — computed from PlantVillage TRAINING SET ONLY.
# Using ImageNet stats (mean=[0.485,0.456,0.406]) is INCORRECT for plant data
# as the distribution is dominated by greens, not balanced RGB.
NORM_MEAN_PLACEHOLDER = [0.466726, 0.488969, 0.41028]
NORM_STD_PLACEHOLDER  = [0.195034, 0.170282, 0.213409]

# ─── DATASET SPLITS ───────────────────────────────────────────────────────────
# Stratified split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Label efficiency fractions for RQ3 / RQ5 experiments
LABEL_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]

# Number of random seeds for label efficiency subsets.
# Each fraction is created with  different seeds to allow mean±std reporting.
N_SEEDS = 3
SUBSET_SEEDS = [42, 123, 456]
NSHOT_VALUES    = [5, 10, 20]

# ─── DATASET CONSTANTS ────────────────────────────────────────────────────────
# PlantVillage: 38 classes across 14 crop species, ~54,305 images (lab conditions)
PLANTVILLAGE_EXPECTED_CLASSES = 38
PLANTVILLAGE_EXPECTED_MIN_IMAGES = 50000  # Sanity check threshold

# PlantDoc: 27 classes across 13 crop species, ~2,569 images (field conditions)
PLANTDOC_EXPECTED_CLASSES = 27  # actual 28
PLANTDOC_EXPECTED_MIN_IMAGES = 2000

# ─── QUALITY VERIFICATION ─────────────────────────────────────────────────────
# Near-duplicate detection threshold using perceptual hashing (pHash).
# Hamming distance ≤ PHASH_THRESHOLD means images are considered near-duplicates.
# Value of 10 is conservative — catches visually identical images but not
# distinct images of the same disease class.
PHASH_THRESHOLD = 10

# Hash size for perceptual hashing (higher = more precise, slower)
PHASH_HASH_SIZE = 16

# Minimum images per class to consider a class valid for training
MIN_PER_CLASS   = 50  #10

# ─── TAXONOMY ─────────────────────────────────────────────────────────────────
# Five top-level pathogen categories used by Leaf-JEPA.
# Aligned with EPPO (European and Mediterranean Plant Protection Organization)
# and USDA AMS classification standards.
PATHOGEN_CATEGORIES = [
    "Fungal",
    "Bacterial",
    "Viral",
    "Abiotic",
    "Healthy",
]

# ─── ANALYSIS ─────────────────────────────────────────────────────────────────
# Number of sample images to visually inspect per class during dataset analysis
VISUAL_INSPECT_SAMPLES = 16

# Figure DPI for saved analysis plots
FIGURE_DPI = 150

# ─── LOGGING ──────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
