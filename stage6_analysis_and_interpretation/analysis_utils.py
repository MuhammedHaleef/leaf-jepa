"""
analysis_utils.py
=================
Shared utilities for Stage 6: Analysis & Interpretation.

Provides:
  - Result loading & aggregation
  - Statistical testing (Wilcoxon, paired t-test, Cohen's d, bootstrap CI)
  - Embedding extraction & dimensionality reduction (t-SNE, UMAP, PCA)
  - Attention map extraction & IoU computation
  - Confusion matrix differencing
  - Pareto frontier computation
  - Label efficiency analysis (AULEC, crossover)
  - Plotting utilities (consistent styling across all Stage 6 figures)
"""

import json
import warnings
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from scipy import stats as sp_stats
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, silhouette_score,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from stage2_dataset_preparation.outputs.augmentation.transforms import (
    get_pretrain_transform, get_eval_transform, get_finetune_transform
)

# ── Attempt UMAP import (optional) ────────────────────────────────────────
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️  umap-learn not installed; UMAP plots will be skipped.")

# ===========================================================================
# GLOBAL PLOT STYLE
# ===========================================================================

def set_plot_style():
    """Consistent plot style across all Stage 6 figures."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "figure.autolayout": True
        # "savefig.bbox_inches": "tight",
    })

set_plot_style()


# ===========================================================================
# SEED & DEVICE
# ===========================================================================

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ===========================================================================
# RESULT LOADING
# ===========================================================================

def load_json(path):
    """Load a JSON file, return dict or list."""
    with open(path) as f:
        return json.load(f)

def save_json(data, path):
    """Save data as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")

def load_baseline_results(baselines_dir, baseline_ids=None):
    """
    Load aggregate results for all baselines.
    Returns dict: {baseline_id: {seed_42: macro_f1, seed_123: ..., mean: ..., std: ...}}
    """
    baselines_dir = Path(baselines_dir)
    if baseline_ids is None:
        baseline_ids = ["B1", "B2", "B3", "B4", "B5"]

    results = {}
    for bid in baseline_ids:
        agg_path = baselines_dir / f"{bid}_aggregate.json"
        if agg_path.exists():
            agg = load_json(agg_path)
            results[bid] = agg
        else:
            print(f"  ⚠️  {agg_path} not found — {bid} will be missing from analysis")
    return results

def load_peft_results(peft_dir):
    """
    Load Stage 5 PEFT experiment results.
    Expects a structured directory with per-method result JSONs.
    Returns dict: {method_name: {config: ..., seeds: {42: macro_f1, ...}, mean: ..., std: ...}}
    """
    peft_dir = Path(peft_dir)
    results = {}

    # Try loading the main comparison summary
    summary_path = peft_dir / "S1_method_comparison_summary.json"
    if summary_path.exists():
        results["S1"] = load_json(summary_path)

    # Label efficiency
    le_path = peft_dir / "S2_label_efficiency_results.json"
    if le_path.exists():
        results["S2"] = load_json(le_path)

    # Cross-domain
    cd_path = peft_dir / "S3_cross_domain_results.json"
    if cd_path.exists():
        results["S3"] = load_json(cd_path)

    # HP sensitivity
    hp_path = peft_dir / "S4_hp_sensitivity_results.json"
    if hp_path.exists():
        results["S4"] = load_json(hp_path)

    # Also try individual method files
    for f in peft_dir.glob("*_results.json"):
        key = f.stem
        if key not in results:
            results[key] = load_json(f)

    return results

def collect_macro_f1_per_seed(results_dict, method_key="macro_f1"):
    """
    From a results dict, extract per-seed macro-F1 as a numpy array.
    Handles various JSON formats from Stages 3 and 5.
    """
    if isinstance(results_dict, dict):
        # Try common formats
        if "seeds" in results_dict:
            return np.array(list(results_dict["seeds"].values()))
        if "per_seed" in results_dict:
            return np.array([r.get(method_key, r.get("macro_f1", 0))
                            for r in results_dict["per_seed"]])
        if "macro_f1_mean" in results_dict and "macro_f1_std" in results_dict:
            # Reconstruct approximate per-seed from mean/std (fallback)
            m = results_dict["macro_f1_mean"]
            s = results_dict["macro_f1_std"]
            return np.array([m - s, m, m + s])
        if method_key in results_dict and isinstance(results_dict[method_key], list):
            return np.array(results_dict[method_key])
    return None


# ===========================================================================
# STATISTICAL TESTING
# ===========================================================================

def paired_wilcoxon(scores_a, scores_b):
    """
    Wilcoxon signed-rank test for paired samples.
    Returns: (statistic, p_value)
    Falls back gracefully when n is too small.
    """
    diff = np.array(scores_a) - np.array(scores_b)
    if np.all(diff == 0):
        return 0.0, 1.0
    try:
        stat, p = sp_stats.wilcoxon(diff, alternative="two-sided")
        return float(stat), float(p)
    except ValueError:
        # Too few samples for Wilcoxon
        return np.nan, np.nan

def paired_ttest(scores_a, scores_b):
    """
    Paired t-test.
    Returns: (t_statistic, p_value)
    """
    try:
        t, p = sp_stats.ttest_rel(scores_a, scores_b)
        return float(t), float(p)
    except Exception:
        return np.nan, np.nan

def cohens_d_paired(scores_a, scores_b):
    """
    Cohen's d for paired samples.
    d = mean(diff) / std(diff)
    """
    diff = np.array(scores_a) - np.array(scores_b)
    if diff.std() == 0:
        return 0.0
    return float(diff.mean() / diff.std())

def effect_size_label(d):
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"

def bootstrap_ci(scores, n_bootstrap=10000, ci=0.95, seed=42):
    """
    Bootstrap confidence interval for the mean.
    Returns: (lower, upper)
    """
    rng = np.random.RandomState(seed)
    scores = np.array(scores)
    means = np.array([
        rng.choice(scores, size=len(scores), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(means, alpha * 100)), float(np.percentile(means, (1 - alpha) * 100))

def pairwise_significance_matrix(method_scores, method_names, test="wilcoxon"):
    """
    Compute pairwise significance matrix.

    Args:
        method_scores: dict {method_name: np.array of per-seed scores}
        method_names: list of method names (defines ordering)
        test: "wilcoxon" or "ttest"

    Returns:
        p_matrix: DataFrame of p-values
        d_matrix: DataFrame of Cohen's d
    """
    n = len(method_names)
    p_mat = np.ones((n, n))
    d_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            a = method_scores[method_names[i]]
            b = method_scores[method_names[j]]
            if a is None or b is None:
                continue

            if test == "wilcoxon":
                _, p = paired_wilcoxon(a, b)
            else:
                _, p = paired_ttest(a, b)

            d = cohens_d_paired(a, b)
            p_mat[i, j] = p
            p_mat[j, i] = p
            d_mat[i, j] = d
            d_mat[j, i] = -d

    p_df = pd.DataFrame(p_mat, index=method_names, columns=method_names)
    d_df = pd.DataFrame(d_mat, index=method_names, columns=method_names)
    return p_df, d_df

def plot_significance_matrix(p_df, d_df, save_path, alpha=0.05):
    """Plot significance heatmap with effect sizes annotated."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # P-value heatmap
    mask = np.eye(len(p_df), dtype=bool)
    p_display = p_df.copy()
    p_display.values[mask] = np.nan

    cmap_p = sns.color_palette(["#27ae60", "#f39c12", "#e74c3c"], as_cmap=True)
    sns.heatmap(p_display, annot=True, fmt=".3f", mask=mask, ax=axes[0],
                cmap="RdYlGn_r", vmin=0, vmax=0.2, linewidths=0.5,
                annot_kws={"fontsize": 8})
    axes[0].set_title("P-Values (Pairwise Significance)")

    # Effect size heatmap
    d_display = d_df.copy()
    d_display.values[mask] = np.nan
    sns.heatmap(d_display, annot=True, fmt=".2f", mask=mask, ax=axes[1],
                cmap="RdBu", center=0, vmin=-3, vmax=3, linewidths=0.5,
                annot_kws={"fontsize": 8})
    axes[1].set_title("Cohen's d (Effect Size)")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ===========================================================================
# DATASET FOR FEATURE EXTRACTION
# ===========================================================================

class SimpleImageDataset(Dataset):
    """Loads images from a list of (path, label) pairs."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def get_eval_transform(norm_mean, norm_std, image_size=224):
    """Deterministic evaluation transform."""
    # return transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=norm_mean, std=norm_std),
    # ])
    return get_eval_transform()


def load_split_data(splits_dir, split="test"):
    """
    Load image paths and labels from Stage 2 split JSON.
    Returns: (image_paths, labels, class_names)
    """
    split_path = Path(splits_dir) / f"{split}.json"
    data = load_json(split_path)
    class_names = data.get("class_names", [])
    samples = data.get("samples", data.get("data", []))

    image_paths = []
    labels = []
    for s in samples:
        if isinstance(s, dict):
            image_paths.append(s["path"])
            labels.append(s["label"])
        elif isinstance(s, (list, tuple)):
            image_paths.append(s[0])
            labels.append(s[1])

    return image_paths, labels, class_names


# ===========================================================================
# ENCODER LOADING
# ===========================================================================

def load_ijepa_encoder(checkpoint_path, device="cpu"):
    """
    Load an I-JEPA encoder (generic or Leaf-JEPA) from checkpoint.
    Returns the model in eval mode.
    """
    import timm

    model = timm.create_model(
        "vit_huge_patch14_224",
        pretrained=False,
        num_classes=0,
        global_pool="avg",
        no_embed_class=True,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if "target_encoder" in ckpt:
        state_dict = ckpt["target_encoder"]
    elif "encoder" in ckpt:
        state_dict = ckpt["encoder"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Clean keys
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        if k == "norm.weight":
            k = "fc_norm.weight"
        elif k == "norm.bias":
            k = "fc_norm.bias"
        cleaned[k] = v
    cleaned.pop("cls_token", None)

    msg = model.load_state_dict(cleaned, strict=False)
    if msg.missing_keys:
        missing_important = [k for k in msg.missing_keys if k != "cls_token"]
        if missing_important:
            print(f"  ⚠️  Missing keys: {missing_important[:5]}")
    if msg.unexpected_keys:
        print(f"  ⚠️  Unexpected keys: {msg.unexpected_keys[:5]}")

    model = model.to(device)
    model.eval()
    return model


# ===========================================================================
# FEATURE EXTRACTION
# ===========================================================================

@torch.no_grad()
def extract_features(model, dataloader, device="cpu", max_samples=None):
    """
    Extract features from a frozen encoder.
    Returns: (features: np.array [N, D], labels: np.array [N])
    """
    all_feats = []
    all_labels = []
    n = 0

    for images, labels in dataloader:
        images = images.to(device)
        feats = model(images)
        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))
        n += len(images)
        if max_samples and n >= max_samples:
            break

    feats = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    if max_samples and len(feats) > max_samples:
        feats = feats[:max_samples]
        labels = labels[:max_samples]

    return feats, labels


# ===========================================================================
# DIMENSIONALITY REDUCTION
# ===========================================================================

def compute_tsne(features, perplexity=30, n_iter=1000, seed=42, pca_dim=50):
    """PCA + t-SNE reduction to 2D."""
    if pca_dim and features.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=seed)
        features = pca.fit_transform(features)
        print(f"  PCA: {features.shape[1]}D → explained variance = {pca.explained_variance_ratio_.sum():.3f}")

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                random_state=seed, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(features)
    return coords

def compute_umap(features, n_neighbors=15, min_dist=0.1, seed=42, pca_dim=50):
    """PCA + UMAP reduction to 2D."""
    if not HAS_UMAP:
        print("  ⚠️  UMAP not available, skipping.")
        return None

    if pca_dim and features.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=seed)
        features = pca.fit_transform(features)

    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                   random_state=seed)
    coords = reducer.fit_transform(features)
    return coords


# ===========================================================================
# EMBEDDING ANALYSIS
# ===========================================================================

def compute_silhouette(features, labels, pca_dim=50, seed=42, sample_size=None):
    """Compute silhouette score with optional PCA pre-reduction."""
    if pca_dim and features.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=seed)
        features = pca.fit_transform(features)

    if sample_size and len(features) > sample_size:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(features), sample_size, replace=False)
        features = features[idx]
        labels = labels[idx]

    score = silhouette_score(features, labels)
    return float(score)

def compute_cluster_distances(features, labels, class_names=None):
    """
    Compute intra-class and inter-class distances.
    Returns dict with per-class intra distance and pairwise inter distances.
    """
    unique_labels = np.unique(labels)
    centroids = {}
    intra_distances = {}

    for lbl in unique_labels:
        mask = labels == lbl
        class_feats = features[mask]
        centroid = class_feats.mean(axis=0)
        centroids[lbl] = centroid

        # Mean pairwise L2 within class (sample if large)
        if len(class_feats) > 200:
            idx = np.random.choice(len(class_feats), 200, replace=False)
            class_feats = class_feats[idx]
        dists = np.linalg.norm(class_feats[:, None] - class_feats[None, :], axis=-1)
        intra_distances[int(lbl)] = float(dists[np.triu_indices_from(dists, k=1)].mean())

    # Inter-class centroid distances
    inter_distances = {}
    labels_list = sorted(centroids.keys())
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            d = float(np.linalg.norm(centroids[labels_list[i]] - centroids[labels_list[j]]))
            key = f"{labels_list[i]}_vs_{labels_list[j]}"
            inter_distances[key] = d

    return {
        "intra_class": intra_distances,
        "inter_class": inter_distances,
        "centroids": {int(k): v.tolist() for k, v in centroids.items()},
    }

def hard_pair_separation(features, labels, pair_indices, class_names=None):
    """
    For specific class pairs, compute centroid distance and separation ratio.
    pair_indices: list of (class_idx_a, class_idx_b)
    """
    results = []
    for a, b in pair_indices:
        feats_a = features[labels == a]
        feats_b = features[labels == b]
        if len(feats_a) == 0 or len(feats_b) == 0:
            continue

        centroid_a = feats_a.mean(axis=0)
        centroid_b = feats_b.mean(axis=0)
        inter_dist = float(np.linalg.norm(centroid_a - centroid_b))

        intra_a = np.linalg.norm(feats_a - centroid_a, axis=1).mean()
        intra_b = np.linalg.norm(feats_b - centroid_b, axis=1).mean()
        avg_intra = (intra_a + intra_b) / 2

        sep_ratio = inter_dist / avg_intra if avg_intra > 0 else float("inf")

        name_a = class_names[a] if class_names else str(a)
        name_b = class_names[b] if class_names else str(b)

        results.append({
            "class_a": name_a, "class_b": name_b,
            "idx_a": int(a), "idx_b": int(b),
            "centroid_distance": inter_dist,
            "avg_intra_distance": float(avg_intra),
            "separation_ratio": float(sep_ratio),
        })
    return results


# ===========================================================================
# ATTENTION MAP EXTRACTION
# ===========================================================================

@torch.no_grad()
def extract_attention_maps(model, images_tensor, device="cpu"):
    """
    Extract self-attention from the last transformer block.
    For I-JEPA (no CLS token), returns mean attention across all patches.

    Args:
        model: ViT model (timm)
        images_tensor: [B, 3, H, W]

    Returns:
        attn_maps: [B, num_patches] — per-patch attention importance
    """
    images_tensor = images_tensor.to(device)

    # Register hook on last attention block
    attn_weights = []

    def hook_fn(module, input, output):
        # timm Attention module stores attn after softmax
        # We need to access the attn_drop or compute it manually
        pass

    # Alternative: use model.blocks[-1].attn directly
    # Compute attention manually
    x = model.patch_embed(images_tensor)
    x = x + model.pos_embed
    x = model.pos_drop(x) if hasattr(model, 'pos_drop') else x
    x = model.patch_drop(x) if hasattr(model, 'patch_drop') else x
    x = model.norm_pre(x) if hasattr(model, 'norm_pre') else x

    # Pass through all blocks except the last
    for blk in model.blocks[:-1]:
        x = blk(x)

    # In the last block, extract attention
    last_block = model.blocks[-1]
    B, N, C = x.shape

    # Get QKV from last block's attention
    attn_module = last_block.attn
    qkv = attn_module.qkv(last_block.norm1(x))
    qkv = qkv.reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    scale = (C // attn_module.num_heads) ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)  # [B, num_heads, N, N]

    # Average across heads, then mean across source patches
    attn_mean = attn.mean(dim=1)  # [B, N, N]
    # Per-patch importance: mean attention received from all other patches
    patch_importance = attn_mean.mean(dim=1)  # [B, N]

    return patch_importance.cpu().numpy()

def attention_to_heatmap(attn_map, grid_size=16, image_size=224):
    """
    Reshape 1D attention vector to 2D heatmap and upscale.

    Args:
        attn_map: [num_patches] — flat attention importance
        grid_size: patch grid dimension (16 for 224/14)
        image_size: target resolution

    Returns:
        heatmap: [image_size, image_size] — upscaled attention map
    """
    heatmap = attn_map.reshape(grid_size, grid_size)

    # Upscale via bilinear interpolation
    heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).float()
    heatmap_up = torch.nn.functional.interpolate(
        heatmap_tensor, size=(image_size, image_size),
        mode="bilinear", align_corners=False
    )
    return heatmap_up.squeeze().numpy()

def compute_attention_iou(attn_map, saliency_map, top_k_pct=0.25):
    """
    IoU between high-attention patches and disease saliency regions.

    Args:
        attn_map: [num_patches] attention importance
        saliency_map: [num_patches] disease region saliency (from Stage 4)
        top_k_pct: fraction of patches to consider "high"

    Returns:
        iou: float
    """
    k = max(1, int(len(attn_map) * top_k_pct))

    attn_topk = set(np.argsort(attn_map)[-k:])
    sal_topk = set(np.argsort(saliency_map)[-k:])

    intersection = len(attn_topk & sal_topk)
    union = len(attn_topk | sal_topk)

    return intersection / union if union > 0 else 0.0

def plot_attention_comparison(images, attn_maps_dict, save_path,
                               grid_size=16, image_size=224,
                               alpha=0.45, cmap="inferno"):
    """
    Plot attention comparison grid: original | model_1 | model_2 | ...

    Args:
        images: list of PIL images or numpy arrays [N]
        attn_maps_dict: OrderedDict {model_name: list of [num_patches] arrays}
        save_path: path to save figure
    """
    n_images = len(images)
    n_models = len(attn_maps_dict)
    n_cols = 1 + n_models  # original + each model

    fig, axes = plt.subplots(n_images, n_cols, figsize=(4 * n_cols, 4 * n_images))
    if n_images == 1:
        axes = axes[np.newaxis, :]

    model_names = list(attn_maps_dict.keys())

    for i in range(n_images):
        # Original image
        img = images[i]
        if isinstance(img, Image.Image):
            img = np.array(img.resize((image_size, image_size)))
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original" if i == 0 else "", fontsize=10)
        axes[i, 0].axis("off")

        # Attention overlays
        for j, mname in enumerate(model_names):
            attn = attn_maps_dict[mname][i]
            heatmap = attention_to_heatmap(attn, grid_size, image_size)

            axes[i, j + 1].imshow(img)
            axes[i, j + 1].imshow(heatmap, cmap=cmap, alpha=alpha)
            axes[i, j + 1].set_title(mname if i == 0 else "", fontsize=10)
            axes[i, j + 1].axis("off")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ===========================================================================
# CONFUSION MATRIX ANALYSIS
# ===========================================================================

def compute_normalised_cm(y_true, y_pred, num_classes):
    """Row-normalised confusion matrix (recall-based)."""
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cm / row_sums

def confusion_difference(cm_a, cm_b):
    """
    Compute difference: cm_b - cm_a.
    Positive = confusions that INCREASED in B.
    Negative = confusions that DECREASED in B (improvements).
    """
    return cm_b - cm_a

def plot_confusion_diff(diff_cm, class_names, save_path, title="Confusion Difference"):
    """Plot differential confusion matrix with diverging colourmap."""
    fig, ax = plt.subplots(figsize=(18, 16))

    vmax = max(abs(diff_cm.min()), abs(diff_cm.max()))
    if vmax == 0:
        vmax = 1

    sns.heatmap(diff_cm, xticklabels=class_names, yticklabels=class_names,
                cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                ax=ax, linewidths=0.3, annot=False,
                cbar_kws={"label": "Change (+ = worse, − = better)"})
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(fontsize=6, rotation=90)
    plt.yticks(fontsize=6, rotation=0)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")

def top_confusion_changes(diff_cm, class_names, n=10):
    """
    Find top improved (most negative off-diagonal) and worsened (most positive)
    confusion pairs from a difference matrix.
    Returns: (improved: list of dicts, worsened: list of dicts)
    """
    rows, cols = np.where(~np.eye(len(diff_cm), dtype=bool))
    changes = []
    for r, c in zip(rows, cols):
        changes.append({
            "true_class": class_names[r],
            "pred_class": class_names[c],
            "change": float(diff_cm[r, c]),
        })

    changes.sort(key=lambda x: x["change"])
    improved = changes[:n]  # most negative = biggest improvement
    worsened = changes[-n:][::-1]  # most positive = biggest degradation

    return improved, worsened


# ===========================================================================
# PARETO FRONTIER
# ===========================================================================

def compute_pareto_frontier(points):
    """
    Compute Pareto-optimal points (minimize params, maximize F1).

    Args:
        points: list of (params, f1, name)

    Returns:
        pareto_points: list of (params, f1, name) on the frontier
    """
    # Sort by params ascending
    sorted_pts = sorted(points, key=lambda x: x[0])

    pareto = [sorted_pts[0]]
    max_f1 = sorted_pts[0][1]

    for pt in sorted_pts[1:]:
        if pt[1] > max_f1:
            pareto.append(pt)
            max_f1 = pt[1]

    return pareto

def plot_pareto(points, pareto_points, save_path, method_colours=None,
                rq3_threshold=None, full_ft_f1=None):
    """
    Pareto frontier: trainable params (log x) vs macro-F1 (linear y).

    Args:
        points: list of (params, f1, name)
        pareto_points: list of pareto-optimal points
        method_colours: dict {name: colour}
        rq3_threshold: if set, shade region within this % of full FT
        full_ft_f1: full fine-tune F1 for RQ3 threshold
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Scatter all points
    for params, f1, name in points:
        colour = method_colours.get(name, "#888888") if method_colours else "#3498db"
        ax.scatter(params, f1, c=colour, s=80, zorder=3, edgecolors="white", linewidths=0.5)
        ax.annotate(name, (params, f1), textcoords="offset points",
                   xytext=(8, 4), fontsize=8, color=colour)

    # Pareto frontier line
    if len(pareto_points) > 1:
        pp = sorted(pareto_points, key=lambda x: x[0])
        ax.plot([p[0] for p in pp], [p[1] for p in pp],
                "k--", alpha=0.4, linewidth=1.5, label="Pareto frontier")

    # RQ3 threshold
    if rq3_threshold and full_ft_f1:
        threshold = full_ft_f1 - rq3_threshold / 100.0
        ax.axhspan(threshold, full_ft_f1 + 0.005, alpha=0.08, color="green",
                   label=f"Within {rq3_threshold}% of Full FT")

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Pareto Frontier: Accuracy vs Parameter Efficiency")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ===========================================================================
# LABEL EFFICIENCY ANALYSIS
# ===========================================================================

def compute_aulec(fractions, f1_scores):
    """
    Area Under the Label Efficiency Curve (AULEC).
    Uses trapezoidal rule on (fraction, f1) pairs.
    """
    fractions = np.array(fractions)
    f1_scores = np.array(f1_scores)
    idx = np.argsort(fractions)
    return float(np.trapz(f1_scores[idx], fractions[idx]))

def find_crossover(fractions, f1_a, f1_b):
    """
    Find the approximate fraction where method B surpasses method A.
    Returns the fraction, or None if no crossover.
    """
    fractions = np.array(fractions)
    f1_a = np.array(f1_a)
    f1_b = np.array(f1_b)
    diff = f1_b - f1_a

    idx = np.argsort(fractions)
    fractions = fractions[idx]
    diff = diff[idx]

    for i in range(len(diff) - 1):
        if diff[i] <= 0 and diff[i + 1] > 0:
            # Linear interpolation
            frac = fractions[i] + (fractions[i + 1] - fractions[i]) * (-diff[i]) / (diff[i + 1] - diff[i])
            return float(frac)
    if diff[-1] > 0:
        return float(fractions[0])
    return None

def plot_label_efficiency(method_data, save_path, method_colours=None):
    """
    Overlay label efficiency curves for all methods.

    Args:
        method_data: dict {method_name: {fractions: [...], mean: [...], std: [...]}}
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for name, data in method_data.items():
        fracs = np.array(data["fractions"])
        means = np.array(data["mean"])
        stds = np.array(data.get("std", np.zeros_like(means)))
        colour = method_colours.get(name, None) if method_colours else None

        ax.plot(fracs, means, "o-", label=name, color=colour, markersize=5, linewidth=1.5)
        ax.fill_between(fracs, means - stds, means + stds, alpha=0.15, color=colour)

    ax.set_xlabel("Label Fraction")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Label Efficiency: All Methods")
    ax.legend(loc="lower right", ncol=2)
    ax.set_xlim(-0.02, 1.05)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ===========================================================================
# RADAR CHART
# ===========================================================================

def plot_radar(method_profiles, metric_names, save_path, method_colours=None):
    """
    Radar/spider chart comparing methods across multiple normalised metrics.

    Args:
        method_profiles: dict {method_name: [v1, v2, ...]} — all normalised 0-1
        metric_names: list of metric axis labels
    """
    n_metrics = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for name, values in method_profiles.items():
        vals = list(values) + [values[0]]
        colour = method_colours.get(name, None) if method_colours else None
        ax.plot(angles, vals, linewidth=1.5, label=name, color=colour)
        ax.fill(angles, vals, alpha=0.08, color=colour)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Method Profiles", y=1.08, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ===========================================================================
# t-SNE PLOTTING
# ===========================================================================

def plot_tsne_grid(coords_dict, labels, class_names, save_path,
                   silhouette_scores=None, figsize_per_panel=(6, 5)):
    """
    Multi-panel t-SNE progression figure.

    Args:
        coords_dict: OrderedDict {panel_name: [N, 2] coords}
        labels: [N] integer labels
        class_names: list of class names
    """
    n_panels = len(coords_dict)
    fig, axes = plt.subplots(1, n_panels, figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]))
    if n_panels == 1:
        axes = [axes]

    # Colour palette for classes
    n_classes = len(np.unique(labels))
    if n_classes <= 20:
        palette = plt.cm.tab20(np.linspace(0, 1, n_classes))
    else:
        palette = plt.cm.nipy_spectral(np.linspace(0, 1, n_classes))

    for ax, (name, coords) in zip(axes, coords_dict.items()):
        for c in range(n_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1],
                      c=[palette[c]], s=5, alpha=0.5, label=class_names[c] if c < 5 else None)

        title = name
        if silhouette_scores and name in silhouette_scores:
            title += f"\n(silhouette = {silhouette_scores[name]:.3f})"
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ===========================================================================
# HARD CLASS ANALYSIS
# ===========================================================================

def rank_classes_by_difficulty(per_class_f1_dict, class_names):
    """
    Rank all classes by average F1 across all methods.

    Args:
        per_class_f1_dict: dict {method_name: list of per-class F1}
        class_names: list of class names

    Returns:
        DataFrame ranked by avg F1 ascending (hardest first)
    """
    df = pd.DataFrame(per_class_f1_dict, index=class_names)
    df["avg_f1"] = df.mean(axis=1)
    df["std_f1"] = df.std(axis=1)
    df = df.sort_values("avg_f1", ascending=True)
    return df

def analyse_difficulty_factors(hard_classes_df, stage2_data=None):
    """
    Cross-reference hard classes with Stage 2 findings.
    Returns enriched DataFrame with hypothesised difficulty factors.
    """
    factors = []
    for cls_name in hard_classes_df.index:
        factor = {"class": cls_name, "avg_f1": hard_classes_df.loc[cls_name, "avg_f1"]}

        if stage2_data:
            # Check sample count
            if "class_counts" in stage2_data:
                count = stage2_data["class_counts"].get(cls_name, 0)
                factor["sample_count"] = count
                factor["is_rare"] = count < 500

            # Check background ratio
            if "background_ratios" in stage2_data:
                bg = stage2_data["background_ratios"].get(cls_name, 0)
                factor["background_ratio"] = bg
                factor["high_background"] = bg > 0.75

            # Check inter-class similarity
            if "top_similar_pairs" in stage2_data:
                for pair in stage2_data["top_similar_pairs"]:
                    if cls_name in (pair.get("class_a"), pair.get("class_b")):
                        factor["most_similar_to"] = (
                            pair["class_b"] if pair["class_a"] == cls_name else pair["class_a"]
                        )
                        factor["similarity_score"] = pair.get("similarity", 0)
                        break

        factors.append(factor)

    return pd.DataFrame(factors).set_index("class")


# ===========================================================================
# UTILITY: PRINT HELPERS
# ===========================================================================

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def print_result(key, value, indent=2):
    prefix = " " * indent
    if isinstance(value, float):
        print(f"{prefix}{key}: {value:.4f}")
    else:
        print(f"{prefix}{key}: {value}")
