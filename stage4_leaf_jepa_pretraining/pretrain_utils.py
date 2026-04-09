"""
pretrain_utils.py
=================
Shared utilities for Stage 4: Leaf-JEPA Pretraining.

Contains:
  - PlantVillagePretrainDataset
  - get_pretrain_transform()
  - MultiBlockMasking          — standard I-JEPA multi-block masking
  - DiseaseRegionBiasedMasking — novel contribution: colour-saliency weighted masking
  - SaliencyMap                — per-patch hue-deviation saliency
  - IJEPAPredictor             — narrow transformer predictor (re-init from scratch)
  - EMAUpdater                 — exponential moving average target encoder update
  - get_layerwise_optimizer()  — layer-wise LR assignment (freeze bot, low mid, std top)
  - pretrain_one_epoch()       — one full pretraining epoch with AMP
  - LinearProbeMonitor         — quick linear probe for representation quality tracking
  - save_checkpoint() / load_checkpoint()
  - plot_pretrain_curves()
  - set_seed()
"""

import os, json, time, random, copy, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stage2_dataset_preparation.outputs.augmentation.transforms import (
    get_pretrain_transform, get_eval_transform, get_finetune_transform
)

from torch.amp import autocast

from tqdm import tqdm

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================
# Dataset
# ============================================================

class PlantVillagePretrainDataset(Dataset):
    """
    Loads PlantVillage TRAINING split for self-supervised pretraining.
    Labels are never used during pretraining (SSL is unsupervised).
    Uses the same split CSV produced by Stage 2.

    IMPORTANT: PlantDoc must NEVER be included here.
    """

    def __init__(self, split_csv: Path, transform=None):
        df = pd.read_csv(split_csv)
        self.df        = df[df["split"] == "train"].reset_index(drop=True)
        self.transform = transform
        print(f"  PlantVillagePretrainDataset: {len(self.df):,} training images")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # Return image and label_idx (label used only for LP monitoring, not pretraining)
        return img, int(row["label_idx"])


# def get_pretrain_transform(norm_mean: List[float], norm_std: List[float],
#                             image_crop: int = 224, image_resize: int = 256):
#     """
#     Aggressive pretraining augmentation pipeline from Stage 2.
#     Hue jitter ≤ 0.05 — protects disease colour signal (immutable Stage 2 constraint).
#     """
#     return
#
#
# def get_eval_transform(norm_mean, norm_std, image_crop=224, image_resize=256):
#     return transforms.Compose([
#         transforms.Resize(image_resize,
#                           interpolation=transforms.InterpolationMode.BICUBIC),
#         transforms.CenterCrop(image_crop),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=norm_mean, std=norm_std),
#     ])


# ============================================================
# Patch utilities
# ============================================================

def get_num_patches(image_size: int, patch_size: int) -> Tuple[int, int]:
    """Returns (n_patches_h, n_patches_w)."""
    h = image_size // patch_size
    w = image_size // patch_size
    return h, w


def patches_to_mask(patch_indices: List[int], total_patches: int) -> torch.BoolTensor:
    """Convert list of patch indices to a boolean mask tensor."""
    mask = torch.zeros(total_patches, dtype=torch.bool)
    mask[patch_indices] = True
    return mask


# ============================================================
# Standard Multi-Block Masking (I-JEPA original)
# ============================================================

class MultiBlockMasking:
    """
    Generates context and target block masks following the original I-JEPA paper
    (Assran et al., CVPR 2023).

    Context: one large block covering ~85–100% of patches.
    Targets: N_target non-overlapping rectangular blocks.

    Returns:
        context_mask: BoolTensor (total_patches,) — True = context (visible)
        target_masks: List[BoolTensor] — each (total_patches,), True = target (to predict)
    """

    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 14,
                 num_target_blocks: int = 4,
                 context_scale: Tuple = (0.85, 1.0),
                 context_ratio: Tuple = (0.75, 1.5),
                 target_scale: Tuple  = (0.15, 0.20),
                 target_ratio: Tuple  = (0.75, 1.5)):
        self.H, self.W     = image_size // patch_size, image_size // patch_size
        self.total_patches = self.H * self.W
        self.num_target_blocks = num_target_blocks
        self.context_scale = context_scale
        self.context_ratio = context_ratio
        self.target_scale  = target_scale
        self.target_ratio  = target_ratio

    def _sample_block(self, scale_range, ratio_range) -> Tuple[int, int, int, int]:
        """Sample a random rectangular block (r0, c0, r1, c1)."""
        for _ in range(50):  # retry if block doesn't fit
            scale = random.uniform(*scale_range)
            ratio = random.uniform(*ratio_range)
            area  = int(self.total_patches * scale)
            h     = max(1, int(round(math.sqrt(area * ratio))))
            w     = max(1, int(round(math.sqrt(area / ratio))))
            if h <= self.H and w <= self.W:
                r0 = random.randint(0, self.H - h)
                c0 = random.randint(0, self.W - w)
                return r0, c0, r0 + h, c0 + w
        # fallback: small centre block
        h, w = max(1, self.H // 4), max(1, self.W // 4)
        r0, c0 = self.H // 4, self.W // 4
        return r0, c0, r0 + h, c0 + w

    def _block_to_indices(self, r0, c0, r1, c1) -> List[int]:
        indices = []
        for r in range(r0, r1):
            for c in range(c0, c1):
                indices.append(r * self.W + c)
        return indices

    def __call__(self, saliency: Optional[np.ndarray] = None) -> Tuple:
        """
        saliency: ignored in standard masking (used by DiseaseRegionBiasedMasking).
        Returns (context_mask, target_masks).
        """
        # Sample target blocks (no overlap)
        target_indices_all = set()
        target_masks = []
        for _ in range(self.num_target_blocks):
            for _ in range(20):  # retry to avoid overlap
                r0, c0, r1, c1 = self._sample_block(self.target_scale, self.target_ratio)
                block_idx = set(self._block_to_indices(r0, c0, r1, c1))
                if not block_idx & target_indices_all:
                    target_indices_all |= block_idx
                    target_masks.append(patches_to_mask(list(block_idx), self.total_patches))
                    break

        # Sample context block (large, may overlap targets — that's OK per I-JEPA paper)
        r0, c0, r1, c1 = self._sample_block(self.context_scale, self.context_ratio)
        context_mask = patches_to_mask(self._block_to_indices(r0, c0, r1, c1), self.total_patches)

        return context_mask, target_masks


# ============================================================
# Disease-Region-Biased Masking (Novel Contribution)
# ============================================================

class SaliencyMap:
    """
    Computes per-patch saliency as the hue-deviation from healthy leaf green.
    Patches with unusual hue (disease lesions are brownish/yellowish/dark)
    receive higher saliency scores and are preferentially sampled as target blocks.

    Operationalisation:
        saliency_i = 1 - exp(-((hue_i - healthy_hue_center)^2) / (2 * sigma^2))

    This is a domain-specific prior: healthy leaves cluster around hue ≈ 0.30 (HSV),
    while diseased patches deviate toward 0.05–0.20 (yellow/brown) or 0.50+ (purple lesions).
    """

    def __init__(self,
                 patch_size: int = 14,
                 image_size: int = 224,
                 healthy_hue_center=0.3153,
                 healthy_hue_sigma= 0.0690):
        self.patch_size         = patch_size
        self.image_size         = image_size
        self.H                  = image_size // patch_size
        self.W                  = image_size // patch_size
        self.healthy_hue_center = healthy_hue_center
        self.healthy_hue_sigma  = healthy_hue_sigma

    def __call__(self, img_tensor: torch.Tensor, norm_mean, norm_std) -> np.ndarray:
        """
        img_tensor: (3, H, W) normalised tensor.
        Returns saliency: (H//patch_size * W//patch_size,) numpy array, values in [0, 1].
        """
        # Denormalise
        mean = torch.tensor(norm_mean).view(3, 1, 1)
        std  = torch.tensor(norm_std).view(3, 1, 1)
        img_rgb = (img_tensor * std + mean).clamp(0, 1)  # (3, H, W)

        # Convert to HSV — approximate via max/min trick
        R, G, B = img_rgb[0], img_rgb[1], img_rgb[2]
        Cmax = torch.max(torch.stack([R, G, B], dim=0), dim=0).values
        Cmin = torch.min(torch.stack([R, G, B], dim=0), dim=0).values
        delta = Cmax - Cmin + 1e-8

        # Hue computation (simplified)
        hue = torch.zeros_like(R)
        mask_r = (Cmax == R)
        mask_g = (Cmax == G)
        mask_b = (Cmax == B)
        hue[mask_r] = ((G - B)[mask_r] / delta[mask_r]) % 6
        hue[mask_g] = ((B - R)[mask_g] / delta[mask_g]) + 2
        hue[mask_b] = ((R - G)[mask_b] / delta[mask_b]) + 4
        hue = (hue / 6.0).clamp(0, 1)  # normalise to [0, 1]

        # Per-patch mean hue (pool over patch_size x patch_size windows)
        H, W = self.H, self.W
        hue_patches = hue.unfold(0, self.patch_size, self.patch_size) \
                         .unfold(1, self.patch_size, self.patch_size)  # (H, W, ps, ps)
        hue_mean = hue_patches.reshape(H, W, -1).mean(dim=-1)  # (H, W)

        # Add saturation signal (low sat = yellowing)
        R2,G2,B2 = img_rgb[0],img_rgb[1],img_rgb[2]
        Cmax2 = torch.max(torch.stack([R2,G2,B2]),dim=0).values
        Cmin2 = torch.min(torch.stack([R2,G2,B2]),dim=0).values
        sat = (Cmax2 - Cmin2) / (Cmax2 + 1e-8)
        sat_patches = sat.unfold(0,self.patch_size,self.patch_size) \
            .unfold(1,self.patch_size,self.patch_size)
        sat_mean = sat_patches.reshape(H,W,-1).mean(-1)


        # Saliency = deviation from healthy hue
        deviation = (hue_mean - self.healthy_hue_center) ** 2
        saliency  = 1.0 - torch.exp(-deviation / (2 * self.healthy_hue_sigma ** 2))
        saliency  = saliency.numpy().flatten()   # (H*W,)

        # Ensure minimum saliency (every patch has some chance of being targeted)
        saliency = saliency + 0.05
        saliency = saliency / saliency.sum()

        # Low saturation = more salient (yellow/necrotic)
        sat_sal = 1.0 - sat_mean
        # Combine: weighted sum
        saliency_combined = 0.6 * saliency + 0.4 * sat_sal.numpy().flatten()

        return saliency_combined


class DiseaseRegionBiasedMasking(MultiBlockMasking):
    """
    Novel contribution: target blocks are sampled with probability proportional
    to per-patch saliency (hue deviation from healthy leaf distribution).

    Disease lesions (abnormal hue) receive higher saliency → more likely to be
    target blocks → encoder must learn to reason about disease-specific regions.

    Falls back to uniform sampling if saliency is None (handles early training
    before warm-start, or standard ablation mode).

    Reference: extends Assran et al. (2023) I-JEPA with domain-specific
    target block sampling. No prior work applies biased masking to plant
    disease SSL pretraining.
    """

    def __init__(self, *args, bias_strength: float = 3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias_strength = bias_strength  # temperature: higher = more biased

    def _sample_biased_block_centre(self,
                                     saliency: np.ndarray,
                                     block_h: int, block_w: int,
                                     exclude: set) -> Tuple[int, int]:
        """
        Sample a (row, col) centre for a target block, weighted by saliency.
        Ensures block fits within the grid and does not overlap excluded patches.
        """
        # Compute saliency weights on a valid centre grid
        valid_rows = self.H - block_h
        valid_cols = self.W - block_w
        if valid_rows <= 0 or valid_cols <= 0:
            return random.randint(0, max(0, self.H - block_h)), \
                   random.randint(0, max(0, self.W - block_w))

        # Average saliency over the would-be block footprint for each candidate centre
        weights = np.zeros((valid_rows + 1) * (valid_cols + 1))
        idx = 0
        centres = []
        for r0 in range(valid_rows + 1):
            for c0 in range(valid_cols + 1):
                block_patches = {(r0 + dr) * self.W + (c0 + dc)
                                  for dr in range(block_h) for dc in range(block_w)}
                if not block_patches & exclude:
                    patch_list = list(block_patches)
                    weights[idx] = saliency[patch_list].mean()
                else:
                    weights[idx] = 0.0
                centres.append((r0, c0))
                idx += 1

        weights = weights[:idx]
        # Apply temperature
        weights = np.power(weights, self.bias_strength)
        total = weights.sum()
        if total < 1e-10:
            # All blocked or zero saliency — fall back to uniform
            r0 = random.randint(0, valid_rows)
            c0 = random.randint(0, valid_cols)
        else:
            weights /= total
            chosen = np.random.choice(idx, p=weights)
            r0, c0 = centres[chosen]

        return r0, c0

    def __call__(self, saliency: Optional[np.ndarray] = None) -> Tuple:
        """
        saliency: (total_patches,) float array from SaliencyMap, or None for uniform.
        Returns (context_mask, target_masks).
        """
        if saliency is None:
            # Fall back to parent uniform sampling
            return super().__call__(saliency=None)

        # --- Sample target blocks biased by saliency ---
        target_indices_all = set()
        target_masks = []

        for _ in range(self.num_target_blocks):
            scale = random.uniform(*self.target_scale)
            ratio = random.uniform(*self.target_ratio)
            area  = max(1, int(self.total_patches * scale))
            h     = max(1, int(round(math.sqrt(area * ratio))))
            w     = max(1, int(round(math.sqrt(area / ratio))))
            h     = min(h, self.H)
            w     = min(w, self.W)

            r0, c0 = self._sample_biased_block_centre(saliency, h, w, target_indices_all)
            block_idx = set()
            for dr in range(h):
                for dc in range(w):
                    block_idx.add((r0 + dr) * self.W + (c0 + dc))

            target_indices_all |= block_idx
            target_masks.append(patches_to_mask(list(block_idx), self.total_patches))

        # --- Context block: large uniform block (same as standard I-JEPA) ---
        r0, c0, r1, c1 = self._sample_block(self.context_scale, self.context_ratio)
        context_mask = patches_to_mask(self._block_to_indices(r0, c0, r1, c1), self.total_patches)

        return context_mask, target_masks


# ============================================================
# Predictor Network (re-initialised from scratch)
# ============================================================

class PredictorAttentionBlock(nn.Module):
    """Single transformer block for the predictor."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1   = nn.LayerNorm(embed_dim)
        self.attn    = nn.MultiheadAttention(embed_dim, num_heads,
                                               dropout=dropout, batch_first=True)
        self.norm2   = nn.LayerNorm(embed_dim)
        ffn_dim      = int(embed_dim * mlp_ratio)
        self.ffn     = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x
        x = x + self.ffn(self.norm2(x))
        return x


class IJEPAPredictor(nn.Module):
    """
    Narrow transformer predictor for I-JEPA.

    Takes context encoder embeddings (token sequence) + positional tokens
    for target locations, and predicts the target encoder embeddings at
    those positions.

    Architecture (re-initialised, never loaded from Meta checkpoint):
      - Input projection: encoder_embed_dim → pred_embed_dim
      - N transformer blocks (default 4)
      - Output projection: pred_embed_dim → encoder_embed_dim

    The input projection down to pred_embed_dim (256 << 1280) creates the
    bottleneck that forces the encoder to learn compressed representations.
    """

    def __init__(self,
                 encoder_embed_dim: int = 1280,
                 pred_embed_dim: int    = 256,
                 num_patches: int       = 256,
                 num_heads: int         = 4,
                 depth: int             = 4,
                 mlp_ratio: float       = 4.0,
                 dropout: float         = 0.1):
        super().__init__()
        self.encoder_embed_dim = encoder_embed_dim
        self.pred_embed_dim    = pred_embed_dim
        self.num_patches       = num_patches

        # Project from encoder dim down to predictor dim
        self.input_proj  = nn.Linear(encoder_embed_dim, pred_embed_dim, bias=True)

        # Learnable positional tokens for target patch positions
        self.target_pos_embed = nn.Embedding(num_patches, pred_embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            PredictorAttentionBlock(pred_embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm        = nn.LayerNorm(pred_embed_dim)
        # Project back to encoder dim for MSE loss against target encoder output
        self.output_proj = nn.Linear(pred_embed_dim, encoder_embed_dim, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self,
                context_embeddings: torch.Tensor,
                target_patch_ids: torch.Tensor) -> torch.Tensor:
        """
        context_embeddings: (B, N_context, encoder_embed_dim)
        target_patch_ids:   (B, N_target) — patch indices to predict

        Returns predicted_embeddings: (B, N_target, encoder_embed_dim)
        """
        B, N_ctx, _ = context_embeddings.shape
        N_tgt = target_patch_ids.shape[1]

        # Project context to predictor dim
        ctx = self.input_proj(context_embeddings)   # (B, N_ctx, pred_dim)

        # Target positional tokens
        tgt_pos = self.target_pos_embed(target_patch_ids)  # (B, N_tgt, pred_dim)

        # Concatenate: [context tokens | target position queries]
        x = torch.cat([ctx, tgt_pos], dim=1)  # (B, N_ctx + N_tgt, pred_dim)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract only the target token predictions
        pred = x[:, N_ctx:, :]   # (B, N_tgt, pred_dim)

        # Project back to encoder dim
        pred = self.output_proj(pred)  # (B, N_tgt, encoder_embed_dim)

        return pred


# ============================================================
# EMA Target Encoder Updater
# ============================================================

class EMAUpdater:
    """
    Updates target encoder as Exponential Moving Average of context encoder.

    θ_target ← τ · θ_target + (1 − τ) · θ_context

    τ follows a cosine schedule from EMA_TAU_START to EMA_TAU_END over training.
    """

    def __init__(self, tau_start: float = 0.996, tau_end: float = 0.999,
                 total_steps: int = 1000):
        self.tau_start    = tau_start
        self.tau_end      = tau_end
        self.total_steps  = total_steps
        self.current_step = 0

    @property
    def tau(self) -> float:
        """Current EMA momentum (cosine schedule)."""
        progress = min(self.current_step / max(self.total_steps, 1), 1.0)
        return self.tau_end - (self.tau_end - self.tau_start) * (
            math.cos(math.pi * progress) + 1
        ) / 2

    @torch.no_grad()
    def update(self, context_encoder: nn.Module, target_encoder: nn.Module):
        """Perform one EMA update step."""
        tau = self.tau
        for ctx_param, tgt_param in zip(
            context_encoder.parameters(), target_encoder.parameters()
        ):
            tgt_param.data.mul_(tau).add_(ctx_param.data, alpha=1.0 - tau)
        self.current_step += 1
        return tau


# ============================================================
# Layer-wise Optimizer
# ============================================================

def get_layerwise_optimizer(context_encoder: nn.Module,
                              predictor: nn.Module,
                              frozen_layers: List[int],
                              low_lr_layers: List[int],
                              std_lr_layers: List[int],
                              lr_head: float    = 3e-4,
                              lr_mid: float     = 1e-5,
                              lr_top: float     = 1e-4,
                              weight_decay: float = 0.04) -> torch.optim.Optimizer:
    """
    Returns AdamW optimizer with layer-wise LR assignment.

    Predictor → lr_head (highest, training from scratch)
    Encoder blocks 9–31 → lr_top  (standard adaptation)
    Encoder blocks 4–8  → lr_mid  (slow adaptation)
    Encoder blocks 0–3  → FROZEN  (no gradient)
    Other encoder params (patch embed, pos embed, norm) → lr_mid

    Returns the optimizer and also mutates requires_grad on frozen layers.
    """
    param_groups = []

    # --- Predictor: full LR (training from scratch) ---
    param_groups.append({
        "params": list(predictor.parameters()),
        "lr": lr_head,
        "weight_decay": weight_decay,
        "name": "predictor",
    })

    # --- Encoder blocks ---
    if hasattr(context_encoder, "blocks"):
        for i, block in enumerate(context_encoder.blocks):
            if i in frozen_layers:
                for p in block.parameters():
                    p.requires_grad = False
            elif i in low_lr_layers:
                param_groups.append({
                    "params": [p for p in block.parameters() if p.requires_grad],
                    "lr": lr_mid,
                    "weight_decay": weight_decay,
                    "name": f"encoder_block_{i}",
                })
            else:  # std_lr_layers
                param_groups.append({
                    "params": [p for p in block.parameters() if p.requires_grad],
                    "lr": lr_top,
                    "weight_decay": weight_decay,
                    "name": f"encoder_block_{i}",
                })

    # --- Other encoder params (patch embed, pos embed, norm layers) ---
    block_params = set()
    if hasattr(context_encoder, "blocks"):
        for block in context_encoder.blocks:
            block_params.update(id(p) for p in block.parameters())

    other_params = [
        p for p in context_encoder.parameters()
        if id(p) not in block_params and p.requires_grad
    ]
    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": lr_mid,
            "weight_decay": weight_decay,
            "name": "encoder_other",
        })

    return torch.optim.AdamW(param_groups)


# ============================================================
# LR Scheduler with Warmup
# ============================================================

class WarmupCosineScheduler:
    """
    Linear warmup followed by cosine annealing.
    Applies to all param groups in optimizer.
    """

    def __init__(self, optimizer: torch.optim.Optimizer,
                 warmup_epochs: int, total_epochs: int):
        self.optimizer      = optimizer
        self.warmup_epochs  = warmup_epochs
        self.total_epochs   = total_epochs
        self.base_lrs       = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / max(self.warmup_epochs, 1)
        else:
            progress = (epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            factor = 0.5 * (1 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * factor

    def get_last_lr(self) -> List[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ============================================================
# Pretraining Loss
# ============================================================

def ijepa_loss(predicted: torch.Tensor, target: torch.Tensor,
               loss_type: str = "smooth_l1") -> torch.Tensor:
    """
    Computes pretraining loss between predicted and target embeddings.

    predicted: (B, N_target, embed_dim) — from predictor
    target:    (B, N_target, embed_dim) — from target_encoder (detached)
    loss_type: "mse" or "smooth_l1"
    """
    if loss_type == "mse":
        return F.mse_loss(predicted, target)
    elif loss_type == "smooth_l1":
        return F.smooth_l1_loss(predicted, target)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


# ============================================================
# Core pretraining step
# ============================================================

def pretrain_step(imgs: torch.Tensor,
                  context_encoder: nn.Module,
                  target_encoder: nn.Module,
                  predictor: nn.Module,
                  masking_fn,
                  saliency_fn: Optional,
                  optimizer: torch.optim.Optimizer,
                  scaler,
                  device: torch.device,
                  loss_type: str = "smooth_l1",
                  accumulate_steps: int = 1,
                  step_idx: int = 0) -> Dict:
    """
    Single pretraining step for one batch.

    Returns dict with loss and gradient norm.
    """
    B = imgs.shape[0]
    imgs = imgs.to(device)

    # Compute per-image saliency for biased masking
    if saliency_fn is not None:
        # Average saliency over batch (fast approximation)
        # For production, compute per-image; here per-batch for speed
        saliency = saliency_fn(imgs[0].cpu()).astype(np.float32)
    else:
        saliency = None

    # Generate masks (same mask applied to whole batch in this implementation)
    context_mask, target_masks = masking_fn(saliency)
    # context_mask: BoolTensor (total_patches,)
    # target_masks: List[BoolTensor (total_patches,)]

    total_patches = context_mask.shape[0]
    context_idx = context_mask.nonzero(as_tuple=False).squeeze(-1)  # (N_ctx,)

    # Merge all target patches for batched prediction
    target_idx_list = []
    for tm in target_masks:
        target_idx_list.append(tm.nonzero(as_tuple=False).squeeze(-1))
    target_idx = torch.cat(target_idx_list, dim=0)  # (N_tgt_total,)

    # Expand indices to batch
    context_idx_batch = context_idx.unsqueeze(0).expand(B, -1).to(device)  # (B, N_ctx)
    target_idx_batch  = target_idx.unsqueeze(0).expand(B, -1).to(device)   # (B, N_tgt)

    should_step = ((step_idx + 1) % accumulate_steps == 0)

    if scaler is not None:
        with autocast(device_type="cuda"):
            loss, grad_norm = _forward_loss(
                imgs, context_encoder, target_encoder, predictor,
                context_idx_batch, target_idx_batch, loss_type, device
            )
        scaler.scale(loss / accumulate_steps).backward()
        if should_step:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in list(context_encoder.parameters()) +
                 list(predictor.parameters()) if p.requires_grad], 1.0
            ).item()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    else:
        loss, grad_norm = _forward_loss(
            imgs, context_encoder, target_encoder, predictor,
            context_idx_batch, target_idx_batch, loss_type, device
        )
        (loss / accumulate_steps).backward()
        if should_step:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in list(context_encoder.parameters()) +
                 list(predictor.parameters()) if p.requires_grad], 1.0
            ).item()
            optimizer.step()
            optimizer.zero_grad()

    return {"loss": loss.item(), "grad_norm": grad_norm if should_step else 0.0}


def _forward_loss(imgs, context_encoder, target_encoder, predictor,
                   context_idx_batch, target_idx_batch, loss_type, device):
    """Internal forward pass for pretraining loss computation."""
    B = imgs.shape[0]

    # --- Context encoder forward (full image → all patch tokens) ---
    # timm ViT with global_pool='' returns patch tokens; with 'avg' returns CLS+avg
    # We need per-patch tokens for masking, so temporarily set pool mode.
    # The encoder is timm ViT — forward_features() returns (B, N_patches+1, D)
    # (first token is CLS). We take tokens[1:] for patch tokens.
    with torch.no_grad():
        # Target encoder — no grad (EMA, parameters not updated by backprop)
        all_target_tokens = _extract_patch_tokens(target_encoder, imgs)  # (B, N, D)
        target_embeddings = torch.gather(
            all_target_tokens, 1,
            target_idx_batch.unsqueeze(-1).expand(-1, -1, all_target_tokens.shape[-1])
        )  # (B, N_tgt, D)

    # Context encoder — with grad
    all_context_tokens = _extract_patch_tokens(context_encoder, imgs)   # (B, N, D)
    context_embeddings = torch.gather(
        all_context_tokens, 1,
        context_idx_batch.unsqueeze(-1).expand(-1, -1, all_context_tokens.shape[-1])
    )  # (B, N_ctx, D)

    # Predictor
    predicted_embeddings = predictor(context_embeddings, target_idx_batch)  # (B, N_tgt, D)

    # Loss
    loss = ijepa_loss(predicted_embeddings, target_embeddings.detach(), loss_type)
    return loss, 0.0   # grad_norm computed outside


def _extract_patch_tokens(encoder: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    """
    Extract per-patch tokens from a timm ViT encoder.
    timm forward_features() returns (B, N_patches+1, D) where index 0 is CLS.
    We return the patch tokens (indices 1:).
    """
    tokens = encoder.forward_features(imgs)  # (B, N+1, D) or (B, D) depending on pool
    if tokens.dim() == 3:
        # Has CLS token at position 0
        return tokens[:, 1:, :]   # (B, N_patches, D)
    elif tokens.dim() == 2:
        # Already pooled — need to reconfigure encoder
        raise RuntimeError(
            "Encoder returned pooled (2D) tensor. "
            "Set global_pool='' in timm.create_model() for patch-level tokens."
        )
    return tokens


# ============================================================
# Full pretraining epoch
# ============================================================

# def pretrain_one_epoch(context_encoder: nn.Module,
#                         target_encoder: nn.Module,
#                         predictor: nn.Module,
#                         loader: DataLoader,
#                         masking_fn,
#                         saliency_fn: Optional,
#                         optimizer: torch.optim.Optimizer,
#                         ema_updater: EMAUpdater,
#                         device: torch.device,
#                         epoch: int,
#                         total_epochs: int,
#                         use_amp: bool = True,
#                         accumulate_steps: int = 1,
#                         loss_type: str = "smooth_l1",
#                         wandb_run=None) -> Dict:
#     """One full pretraining epoch."""

#     context_encoder.train()
#     predictor.train()
#     target_encoder.eval()  # Target encoder never in train mode

#     scaler = torch.amp.GradScaler() if (use_amp and device.type == "cuda") else None

#     total_loss = 0.0
#     total_grad_norm = 0.0
#     n_batches = 0
#     t0 = time.time()

#     optimizer.zero_grad()
#     # pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}")
#     # for step_idx, (imgs, _labels) in pbar:
#         # metrics = pretrain_step(
#         #     imgs, context_encoder, target_encoder, predictor,
#         #     masking_fn, saliency_fn, optimizer, scaler, device,
#         #     loss_type=loss_type,
#         #     accumulate_steps=accumulate_steps,
#         #     step_idx=step_idx,
#         # )

#     #     tau = ema_updater.update(context_encoder, target_encoder)

#     #     total_loss     += metrics["loss"]
#     #     total_grad_norm+= metrics["grad_norm"]
#     #     n_batches      += 1
#     for step_idx, (imgs, _labels) in enumerate(tqdm(loader,desc=f"Epoch {epoch:3d}",leave=False, dynamic_ncols=True)):
#         metrics = pretrain_step(
#                 imgs, context_encoder, target_encoder, predictor,
#                 masking_fn, saliency_fn, optimizer, scaler, device,
#                 loss_type=loss_type,
#                 accumulate_steps=accumulate_steps,
#                 step_idx=step_idx,
#             )
#         tau = ema_updater.update(context_encoder, target_encoder)

#         total_loss     += metrics["loss"]
#         total_grad_norm += metrics["grad_norm"]
#         n_batches      += 1

#         # if step_idx % 10 == 0:
#         #     pbar.set_postfix(loss=metrics['loss'])

#         # if step_idx % 50 == 0:
#         #     print(f"    Epoch {epoch} | Step {step_idx}/{len(loader)} | "
#         #           f"Loss: {metrics['loss']:.4f} | τ: {tau:.5f}")

#     epoch_loss     = total_loss / max(n_batches, 1)
#     epoch_time     = time.time() - t0
#     current_tau    = ema_updater.tau
#     current_lr     = optimizer.param_groups[0]["lr"]

#     result = {
#         "epoch":      epoch,
#         "loss":       epoch_loss,
#         "tau":        current_tau,
#         "lr":         current_lr,
#         "grad_norm":  total_grad_norm / max(n_batches, 1),
#         "epoch_time": epoch_time,
#     }

#     if wandb_run:
#         wandb_run.log({
#             "pretrain/loss":      epoch_loss,
#             "pretrain/tau":       current_tau,
#             "pretrain/lr":        current_lr,
#             "pretrain/grad_norm": result["grad_norm"],
#             "pretrain/epoch_time_s": epoch_time,
#             "epoch": epoch,
#         })

#     return result

def pretrain_one_epoch(context_encoder: nn.Module,
                        target_encoder: nn.Module,
                        predictor: nn.Module,
                        loader: DataLoader,
                        masking_fn,
                        saliency_fn: Optional,
                        optimizer: torch.optim.Optimizer,
                        ema_updater: EMAUpdater,
                        device: torch.device,
                        epoch: int,
                        total_epochs: int,
                        use_amp: bool = True,
                        accumulate_steps: int = 1,
                        loss_type: str = "smooth_l1",
                        wandb_run=None) -> Dict:
    """One full pretraining epoch."""

    context_encoder.train()
    predictor.train()
    target_encoder.eval()

    scaler = torch.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    total_loss      = 0.0
    total_grad_norm = 0.0
    n_batches       = 0
    t0              = time.time()

    optimizer.zero_grad()

    pbar = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"Epoch {epoch:3d}/{total_epochs}",
        leave=True,
        dynamic_ncols=True,
    )

    for step_idx, (imgs, _labels) in pbar:
        metrics = pretrain_step(
            imgs, context_encoder, target_encoder, predictor,
            masking_fn, saliency_fn, optimizer, scaler, device,
            loss_type=loss_type,
            accumulate_steps=accumulate_steps,
            step_idx=step_idx,
        )

        tau = ema_updater.update(context_encoder, target_encoder)

        total_loss      += metrics["loss"]
        total_grad_norm += metrics["grad_norm"]
        n_batches       += 1

        # Update tqdm bar every step — single source of truth
        pbar.set_postfix(
            loss=f"{metrics['loss']:.4f}",
            tau=f"{tau:.5f}",
            lr=f"{optimizer.param_groups[0]['lr']:.1e}",
        )

    epoch_loss     = total_loss      / max(n_batches, 1)
    epoch_grad     = total_grad_norm / max(n_batches, 1)
    epoch_time     = time.time() - t0
    current_tau    = ema_updater.tau
    current_lr     = optimizer.param_groups[0]["lr"]

    # Single clean end-of-epoch summary line
    print(
        f"  ✓ Epoch {epoch:3d}/{total_epochs} | "
        f"Loss: {epoch_loss:.4f} | "
        f"τ: {current_tau:.5f} | "
        f"LR: {current_lr:.2e} | "
        f"Time: {epoch_time:.0f}s"
    )

    result = {
        "epoch":      epoch,
        "loss":       epoch_loss,
        "tau":        current_tau,
        "lr":         current_lr,
        "grad_norm":  epoch_grad,
        "epoch_time": epoch_time,
    }

    if wandb_run:
        wandb_run.log({
            "pretrain/loss":         epoch_loss,
            "pretrain/tau":          current_tau,
            "pretrain/lr":           current_lr,
            "pretrain/grad_norm":    epoch_grad,
            "pretrain/epoch_time_s": epoch_time,
            "epoch": epoch,
        })

    return result

# ============================================================
# Linear Probe Monitor (representation quality tracker)
# ============================================================

class LinearProbeMonitor:
    """
    Runs a quick linear probe on frozen encoder every LP_MONITOR_INTERVAL epochs.
    Uses a fraction of training data for speed.
    Records val macro-F1 as primary convergence criterion.
    """

    def __init__(self, splits_dir: Path, norm_mean, norm_std,
                 num_classes: int = 38,
                 monitor_epochs: int = 5,
                 monitor_frac: float = 0.2,
                 batch_size: int = 256,
                 num_workers: int = 4,
                 device: torch.device = None):
        from sklearn.metrics import f1_score
        self.splits_dir     = splits_dir
        self.norm_mean      = norm_mean
        self.norm_std       = norm_std
        self.num_classes    = num_classes
        self.monitor_epochs = monitor_epochs
        self.monitor_frac   = monitor_frac
        self.batch_size     = batch_size
        self.num_workers    = num_workers
        self.device         = device or torch.device("cpu")
        self.history        = []

    def _build_loaders(self):
        from torch.utils.data import DataLoader
        eval_transform = get_eval_transform()
        train_transform = get_pretrain_transform()

        csv_path = self.splits_dir / "plantvillage_splits.csv"
        train_ds = PlantVillagePretrainDataset(csv_path, transform=train_transform)
        val_ds   = PlantVillagePretrainDataset.__new__(PlantVillagePretrainDataset)
        val_ds.__init__(csv_path, transform=eval_transform)  # reuse class but val split
        # Patch to use val split
        df = pd.read_csv(csv_path)
        val_ds.df = df[df["split"] == "val"].reset_index(drop=True)

        # Subset of training data for speed
        n_sub = max(100, int(len(train_ds) * self.monitor_frac))
        indices = random.sample(range(len(train_ds)), n_sub)
        sub_train_ds = Subset(train_ds, indices)

        train_loader = DataLoader(sub_train_ds, batch_size=self.batch_size,
                                   shuffle=True,  num_workers=self.num_workers)
        val_loader   = DataLoader(val_ds,         batch_size=self.batch_size,
                                   shuffle=False, num_workers=self.num_workers)
        return train_loader, val_loader

    @torch.no_grad()
    def _extract_features(self, encoder, loader):
        encoder.eval()
        feats, labels = [], []
        for imgs, lbs in loader:
            imgs = imgs.to(self.device)
            with torch.amp.autocast(device_type = "cuda"):
                tokens = encoder.forward_features(imgs)  # (B, N+1, D) or (B, D)

                # Handle both pooled and unpooled encoder outputs
                if tokens.dim() == 3:
                    # Has CLS + patch tokens → mean-pool patch tokens (drop CLS at index 0)
                    f = tokens[:, 1:, :].mean(dim=1)   # (B, D)
                elif tokens.dim() == 2:
                    # Already pooled
                    f = tokens                           # (B, D)
                else:
                    raise RuntimeError(f"Unexpected encoder output shape: {tokens.shape}")

            feats.append(f.cpu().float())

            
            # with autocast(device_type="cuda"):
            #     f = encoder(imgs)  # (B, D) with global_pool='avg'
            # feats.append(f.cpu())
            labels.extend(lbs.numpy())
        return torch.cat(feats, dim=0).numpy(), np.array(labels)

    def run(self, target_encoder: nn.Module, pretrain_epoch: int,
             wandb_run=None) -> float:
        """
        Trains a quick linear probe on frozen target_encoder features.
        Returns val macro-F1.
        """
        print(f"\n  [LP Monitor] Epoch {pretrain_epoch} — training linear probe...")
        train_loader, val_loader = self._build_loaders()

        # Extract features
        train_feats, train_labels = self._extract_features(target_encoder, train_loader)
        val_feats,   val_labels   = self._extract_features(target_encoder, val_loader)

        # Fast sklearn linear classifier (much faster than full PyTorch training)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import f1_score as sk_f1

        scaler_sk = StandardScaler()
        X_train = scaler_sk.fit_transform(train_feats)
        X_val   = scaler_sk.transform(val_feats)

        clf = LogisticRegression(
            max_iter=300, C=0.316, solver="lbfgs", random_state=42
        )
        clf.fit(X_train, train_labels)
        val_preds = clf.predict(X_val)
        val_f1    = sk_f1(val_labels, val_preds, average="macro", zero_division=0)

        result = {
            "pretrain_epoch": pretrain_epoch,
            "lp_val_macro_f1": val_f1,
            "n_train_samples": len(train_feats),
        }
        self.history.append(result)

        print(f"  [LP Monitor] Val Macro-F1: {val_f1:.4f}  "
              f"(best so far: {max(r['lp_val_macro_f1'] for r in self.history):.4f})")

        if wandb_run:
            wandb_run.log({"lp_monitor/val_macro_f1": val_f1,
                            "epoch": pretrain_epoch})

        return val_f1

    def best_f1(self) -> float:
        if not self.history:
            return 0.0
        return max(r["lp_val_macro_f1"] for r in self.history)

    def best_epoch(self) -> int:
        if not self.history:
            return 0
        return max(self.history, key=lambda r: r["lp_val_macro_f1"])["pretrain_epoch"]


# ============================================================
# Checkpoint I/O
# ============================================================

def save_checkpoint(epoch: int,
                     context_encoder: nn.Module,
                     target_encoder: nn.Module,
                     predictor: nn.Module,
                     optimizer: torch.optim.Optimizer,
                     ema_updater: EMAUpdater,
                     history: List[Dict],
                     lp_history: List[Dict],
                     ckpt_dir: Path,
                     tag: str = ""):
    """Saves full training state for resumability."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fname = ckpt_dir / f"epoch_{epoch:04d}{('_' + tag) if tag else ''}.pth"
    torch.save({
        "epoch":           epoch,
        "context_encoder": context_encoder.state_dict(),
        "target_encoder":  target_encoder.state_dict(),
        "predictor":       predictor.state_dict(),
        "optimizer":       optimizer.state_dict(),
        "ema_step":        ema_updater.current_step,
        "history":         history,
        "lp_history":      lp_history,
    }, fname)
    print(f"  Checkpoint saved → {fname}")
    return fname


def load_checkpoint(ckpt_path: Path,
                     context_encoder: nn.Module,
                     target_encoder: nn.Module,
                     predictor: nn.Module,
                     optimizer: Optional[torch.optim.Optimizer] = None,
                     ema_updater: Optional[EMAUpdater] = None,
                     device: torch.device = None):
    """Loads training state from checkpoint for resuming."""
    device = device or torch.device("cpu")
    ckpt   = torch.load(ckpt_path, map_location=device)

    context_encoder.load_state_dict(ckpt["context_encoder"])
    target_encoder.load_state_dict(ckpt["target_encoder"])
    predictor.load_state_dict(ckpt["predictor"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if ema_updater is not None and "ema_step" in ckpt:
        ema_updater.current_step = ckpt["ema_step"]

    print(f"  Checkpoint loaded from epoch {ckpt['epoch']}: {ckpt_path}")
    return ckpt.get("epoch", 0), ckpt.get("history", []), ckpt.get("lp_history", [])


def export_leaf_jepa_encoder(target_encoder: nn.Module,
                               export_path: Path,
                               epoch: int,
                               lp_val_f1: float,
                               config_info: Dict):
    """
    Exports ONLY the target_encoder state dict as the final Leaf-JEPA encoder.
    This is the checkpoint loaded by B5 and all Stage 5 PEFT experiments.
    """
    export_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "target_encoder": target_encoder.state_dict(),
        "epoch":          epoch,
        "lp_val_macro_f1": lp_val_f1,
        "config":         config_info,
    }, export_path)
    print(f"\n  ✅ Leaf-JEPA encoder exported → {export_path}")
    print(f"     Epoch: {epoch}  |  LP Val Macro-F1: {lp_val_f1:.4f}")
    print(f"\n  Next: Update LEAF_JEPA_CHECKPOINT in config_stage3.py:")
    print(f"     LEAF_JEPA_CHECKPOINT = Path('{export_path}')")


# ============================================================
# Visualisation
# ============================================================

def visualise_masks(img_tensor: torch.Tensor,
                     context_mask: torch.BoolTensor,
                     target_masks: List[torch.BoolTensor],
                     patch_size: int = 14,
                     image_size: int = 224,
                     save_path: Optional[Path] = None,
                     norm_mean: List[float] = None,
                     norm_std: List[float]  = None):
    """
    Visualises the masking strategy on a sample image.
    Context patches: shown normally
    Target patches: highlighted with distinct colours
    Masked (neither): darkened
    """
    import matplotlib.patches as mpatches

    H = W = image_size // patch_size

    # Denormalise image
    if norm_mean and norm_std:
        mean = torch.tensor(norm_mean).view(3, 1, 1)
        std  = torch.tensor(norm_std).view(3, 1, 1)
        img  = (img_tensor * std + mean).clamp(0, 1)
    else:
        img = img_tensor.clamp(0, 1)

    img_np = img.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    # Context mask overlay
    overlay_ctx = img_np.copy()
    for i in range(H * W):
        r, c = i // W, i % W
        r0, c0 = r * patch_size, c * patch_size
        r1, c1 = r0 + patch_size, c0 + patch_size
        if not context_mask[i].item():
            overlay_ctx[r0:r1, c0:c1] *= 0.3  # darken non-context
    axes[1].imshow(overlay_ctx)
    axes[1].set_title(f"Context Mask\n({context_mask.sum().item()} patches)", fontsize=12)
    axes[1].axis("off")

    # Target blocks overlay
    colours = plt.cm.Set1(np.linspace(0, 0.8, len(target_masks)))
    overlay_tgt = img_np.copy() * 0.5  # dim background
    for tm, colour in zip(target_masks, colours):
        for i in range(H * W):
            if tm[i].item():
                r, c = i // W, i % W
                r0, c0 = r * patch_size, c * patch_size
                r1, c1 = r0 + patch_size, c0 + patch_size
                overlay_tgt[r0:r1, c0:c1] = overlay_tgt[r0:r1, c0:c1] * 0.2 + \
                    np.array(colour[:3]) * 0.8

    axes[2].imshow(overlay_tgt)
    n_tgt_patches = sum(tm.sum().item() for tm in target_masks)
    axes[2].set_title(f"Target Blocks ({len(target_masks)} blocks, {n_tgt_patches} patches)", fontsize=12)
    axes[2].axis("off")

    plt.suptitle("I-JEPA Masking Strategy", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Masking visualisation saved → {save_path}")
    plt.close(fig)


def plot_pretrain_curves(history: List[Dict],
                          lp_history: List[Dict],
                          save_path: Path,
                          title: str = "Leaf-JEPA Pretraining"):
    """Plots pretraining loss + EMA tau + linear probe F1 over epochs."""
    epochs     = [h["epoch"]    for h in history]
    losses     = [h["loss"]     for h in history]
    taus       = [h["tau"]      for h in history]
    lp_epochs  = [h["pretrain_epoch"]    for h in lp_history]
    lp_f1s     = [h["lp_val_macro_f1"]  for h in lp_history]

    # Rolling average loss
    window = 10
    rolling_loss = []
    for i in range(len(losses)):
        start = max(0, i - window + 1)
        rolling_loss.append(np.mean(losses[start:i + 1]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, losses,       color="#4fc3f7", alpha=0.5, linewidth=1, label="Loss")
    axes[0].plot(epochs, rolling_loss, color="#0288d1",             linewidth=2, label=f"{window}-ep rolling avg")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Pretraining Loss")
    axes[0].set_title("Pretraining Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # EMA tau
    axes[1].plot(epochs, taus, color="#81c784", linewidth=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("EMA Momentum τ")
    axes[1].set_title("EMA Schedule"); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(min(taus) * 0.9995, 1.0005)

    # LP monitor
    if lp_history:
        axes[2].plot(lp_epochs, lp_f1s, color="#ffb74d", linewidth=2,
                      marker="o", markersize=6, label="LP Val Macro-F1")
        best_ep   = max(lp_history, key=lambda r: r["lp_val_macro_f1"])
        axes[2].axvline(best_ep["pretrain_epoch"], color="#ef5350", linestyle="--",
                         label=f"Best epoch ({best_ep['pretrain_epoch']})")
        axes[2].set_xlabel("Pretrain Epoch"); axes[2].set_ylabel("Macro-F1 (Linear Probe)")
        axes[2].set_title("Representation Quality (LP Monitor)")
        axes[2].legend(); axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "LP monitor not yet run", transform=axes[2].transAxes,
                      ha="center", va="center", fontsize=12, color="gray")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Pretraining curves saved → {save_path}")


def plot_saliency_comparison(img_tensor: torch.Tensor,
                               saliency: np.ndarray,
                               patch_size: int = 14,
                               norm_mean: List[float] = None,
                               norm_std: List[float]  = None,
                               save_path: Optional[Path] = None):
    """Visualises the saliency map alongside the original image."""
    H = W = img_tensor.shape[-1] // patch_size

    if norm_mean and norm_std:
        mean = torch.tensor(norm_mean).view(3, 1, 1)
        std  = torch.tensor(norm_std).view(3, 1, 1)
        img  = (img_tensor * std + mean).clamp(0, 1)
    else:
        img = img_tensor.clamp(0, 1)

    img_np = img.permute(1, 2, 0).numpy()
    sal_map = saliency.reshape(H, W)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image"); axes[0].axis("off")

    im = axes[1].imshow(sal_map, cmap="hot", interpolation="nearest")
    axes[1].set_title("Hue Saliency Map\n(brighter = more disease-like)"); axes[1].axis("off")
    plt.colorbar(im, ax=axes[1])

    overlay = img_np.copy()
    sal_upscaled = np.repeat(np.repeat(sal_map, patch_size, axis=0), patch_size, axis=1)
    sal_norm = (sal_upscaled - sal_upscaled.min()) / (sal_upscaled.max() - sal_upscaled.min() + 1e-8)
    axes[2].imshow(overlay)
    axes[2].imshow(sal_norm, cmap="Reds", alpha=0.4)
    axes[2].set_title("Saliency Overlay"); axes[2].axis("off")

    plt.suptitle("Disease-Region Saliency (Novel Masking Contribution)", fontsize=13)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saliency comparison saved → {save_path}")
    plt.close(fig)
