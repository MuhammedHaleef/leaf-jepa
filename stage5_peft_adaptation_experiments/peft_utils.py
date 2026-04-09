"""
peft_utils.py
=============
All PEFT method implementations and shared training infrastructure for Stage 5.

Contents:
    PEFT Implementations:
        - LoRALinear / inject_lora()
        - AdapterModule / inject_adapters()
        - inject_vpt_shallow() / inject_vpt_deep()
        - inject_bitfit()
    
    Model Building:
        - PEFTClassifier          (frozen encoder + PEFT + linear head)
        - build_peft_model()      (one-line model construction)
        - count_params()
    
    Training:
        - train_one_epoch()
        - evaluate()
        - WarmupCosineScheduler
        - EarlyStopping
        - train_peft()            (unified entry point for any PEFT experiment)
    
    Data:
        - get_transforms()
        - load_split()
        - load_class_weights()
        - build_dataloaders()
    
    Evaluation:
        - extract_features()      (for kNN / t-SNE)
        - knn_evaluate()
        - profile_compute()
    
    Utilities:
        - set_seed()
        - get_device()
        - save_results() / load_results()
        - load_ijepa_encoder()
        - plot_confusion_matrix()
"""

import os
import json
import time
import math
import copy
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report
)

from stage2_dataset_preparation.outputs.augmentation.transforms import (
    get_pretrain_transform, get_eval_transform, get_finetune_transform
)


from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import timm

import torch.multiprocessing as mp

if __name__ == '__main__':
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass

# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        dev = torch.device("cpu")
        print("⚠️ CUDA not available — using CPU (will be very slow)")
    return dev


# ============================================================================
# Transforms Loaded from STAGE 2 - Augmentation strategy OUTPUT transforms.py
# ============================================================================

# def get_transforms(phase: str, norm_mean: list, norm_std: list,
#                    image_crop: int = 224, image_resize: int = 256):
#     """
#     Return transform pipeline for the given phase.
#
#     Phases:
#         'pretrain'  — aggressive augmentation (Stage 4 only)
#         'finetune'  — standard augmentation (Stage 5 training)
#         'eval'      — deterministic (all evaluation)
#     """
#     if phase == "finetune":
#         return transforms.Compose([
#             transforms.Resize(image_resize),
#             transforms.RandomCrop(image_crop),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomVerticalFlip(p=0.5),
#             transforms.ColorJitter(
#                 brightness=0.2, contrast=0.2, saturation=0.2,
#                 hue=0.05  # Capped at 0.05 to preserve disease colour signal
#             ),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=norm_mean, std=norm_std),
#         ])
#     elif phase == "eval":
#         return transforms.Compose([
#             transforms.Resize(image_resize),
#             transforms.CenterCrop(image_crop),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=norm_mean, std=norm_std),
#         ])
#     else:
#         raise ValueError(f"Unknown phase '{phase}'. Use 'finetune' or 'eval'.")
#

# ============================================================================
# Dataset & Data Loading
# ============================================================================

class PlantVillageDataset(Dataset):
    """PlantVillage dataset that loads from a split CSV."""

    def __init__(self, root: Path, split_csv: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.df = pd.read_csv(split_csv)
        # Expect columns: 'path' (relative to root), 'label' (int), 'class_name' (str)
        self.paths  = self.df["path"].tolist()
        self.labels = self.df["label"].tolist()
        self.class_names = sorted(self.df["class_name"].unique())

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.root / self.paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class PlantDocDataset(Dataset):
    """PlantDoc dataset for cross-domain evaluation."""

    def __init__(self, root: Path, transform=None, class_filter: list = None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.class_names = sorted([
            d.name for d in self.root.iterdir()
            if d.is_dir() and (class_filter is None or d.name in class_filter)
        ])
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        
        for cls_name in self.class_names:
            cls_dir = self.root / cls_name
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_split(splits_dir: Path, split_name: str) -> Path:
    """Get path to a split CSV.
    
    Args:
        splits_dir: Stage 2 splits directory
        split_name: e.g., 'train', 'val', 'test', 'train_frac0.10_seed42'
    """
    csv_path = splits_dir / f"{split_name}.csv"
    assert csv_path.exists(), f"Split not found: {csv_path}"
    return csv_path


def load_class_weights(path: Path, device: torch.device) -> torch.Tensor:
    """Load inverse-frequency class weights from Stage 2 JSON."""
    weights = json.loads(Path(path).read_text())
    if isinstance(weights, dict):
        # dict: class_idx -> weight
        w = torch.zeros(len(weights))
        for k, v in weights.items():
            w[int(k)] = v
    else:
        w = torch.tensor(weights, dtype=torch.float32)
    return w.to(device)


def build_dataloaders(
    pv_root: Path, splits_dir: Path, 
    norm_mean: list, norm_std: list,
    fraction: float = 1.0, seed: int = 42,
    batch_size: int = 32, num_workers: int = 4,
    image_crop: int = 224, image_resize: int = 256,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders for a given label fraction."""

    train_tf = get_finetune_transform()
    eval_tf  = get_eval_transform()
    
    # Select appropriate training split
    if fraction < 1.0:
        train_split = f"train_frac{fraction:.2f}_seed{seed}"
    else:
        train_split = "train"
    
    train_ds = PlantVillageDataset(pv_root, load_split(splits_dir, train_split), train_tf)
    val_ds   = PlantVillageDataset(pv_root, load_split(splits_dir, "val"), eval_tf)
    test_ds  = PlantVillageDataset(pv_root, load_split(splits_dir, "test"), eval_tf)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    
    print(f"  Train: {len(train_ds)} images (fraction={fraction}, seed={seed})")
    print(f"  Val:   {len(val_ds)} images")
    print(f"  Test:  {len(test_ds)} images")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# Encoder Loading
# ============================================================================

def load_ijepa_encoder(checkpoint_path: Path, model_name: str, embed_dim: int,
                       device: torch.device, freeze: bool = True) -> nn.Module:
    """Load I-JEPA or Leaf-JEPA encoder from checkpoint.
    
    Works with both Meta I-JEPA checkpoints and Stage 4 Leaf-JEPA exports.
    Always loads the target_encoder (EMA) state dict.
    """
    encoder = timm.create_model(model_name, pretrained=False,
                                num_classes=0, global_pool="avg")
    
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    
    # Handle different checkpoint formats
    if "target_encoder" in ckpt:
        state_dict = ckpt["target_encoder"]
    elif "encoder" in ckpt:
        state_dict = ckpt["encoder"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    
    # Clean key prefixes
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        cleaned[k] = v
    
    msg = encoder.load_state_dict(cleaned, strict=False)
    if msg.missing_keys:
        print(f"  ⚠️  Missing keys: {msg.missing_keys[:5]}...")
    if msg.unexpected_keys:
        print(f"  ⚠️  Unexpected keys: {msg.unexpected_keys[:5]}...")
    
    if freeze:
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()
    
    encoder = encoder.to(device)
    print(f"  Encoder loaded from {Path(checkpoint_path).name} "
          f"({'frozen' if freeze else 'trainable'})")
    return encoder


# ============================================================================
# ──────────────────────────────────────────────────────────────────────────────
# PEFT IMPLEMENTATIONS
# ──────────────────────────────────────────────────────────────────────────────
# ============================================================================


# ────────────────────────── LoRA ──────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for a frozen nn.Linear layer.
    
    W' = W + (B @ A) * (alpha / rank)
    Only B and A are trainable. W remains frozen.
    
    At inference, BA can be merged into W for zero latency overhead.
    """
    
    def __init__(self, original: nn.Linear, rank: int, alpha: float = None,
                 dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank
        
        in_features  = original.in_features
        out_features = original.out_features
        
        # Freeze original weights
        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Kaiming init for A, zero init for B (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen path
        out = self.original(x)
        # LoRA path: x -> dropout -> A -> B -> scale
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return out + lora_out
    
    def merge_weights(self):
        """Merge LoRA weights into original for zero-latency inference."""
        with torch.no_grad():
            self.original.weight.data += (self.lora_B @ self.lora_A * self.scaling)
    
    @property
    def trainable_params(self):
        return self.lora_A.numel() + self.lora_B.numel()


def inject_lora(model: nn.Module, rank: int, alpha: float = None,
                dropout: float = 0.0, target_blocks: list = None) -> int:
    """Inject LoRA into Q and V projections of a ViT encoder.
    
    timm ViT uses a fused 'qkv' linear (in_features -> 3*embed_dim).
    We replace it with a custom module that applies LoRA only to the
    Q and V portions of the fused output.
    
    Args:
        model: timm ViT encoder
        rank: LoRA rank r
        alpha: scaling factor (default: rank)
        dropout: dropout on LoRA path
        target_blocks: list of block indices to inject into (None = all)
    
    Returns:
        Total number of LoRA trainable parameters added.
    """
    total_lora_params = 0
    
    for block_idx, block in enumerate(model.blocks):
        if target_blocks is not None and block_idx not in target_blocks:
            continue
        
        attn = block.attn
        qkv = attn.qkv  # nn.Linear(embed_dim, 3 * embed_dim)
        
        # Replace with LoRA-wrapped version
        lora_qkv = LoRAQKV(qkv, rank=rank, alpha=alpha, dropout=dropout)
        attn.qkv = lora_qkv
        total_lora_params += lora_qkv.trainable_params
    
    return total_lora_params


class LoRAQKV(nn.Module):
    """LoRA wrapper for the fused QKV projection in timm ViT.
    
    The fused linear maps: x (B, N, D) -> (B, N, 3D) where output is [Q, K, V].
    We add LoRA to the Q and V portions only.
    """
    
    def __init__(self, original_qkv: nn.Linear, rank: int,
                 alpha: float = None, dropout: float = 0.0):
        super().__init__()
        self.original_qkv = original_qkv
        
        # Freeze original
        original_qkv.weight.requires_grad = False
        if original_qkv.bias is not None:
            original_qkv.bias.requires_grad = False
        
        in_dim = original_qkv.in_features
        out_dim = original_qkv.out_features
        head_dim = out_dim // 3  # Each of Q, K, V has this size
        
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank
        self.head_dim = head_dim
        
        # LoRA for Q
        self.lora_A_q = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B_q = nn.Parameter(torch.zeros(head_dim, rank))
        nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q)
        
        # LoRA for V
        self.lora_A_v = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B_v = nn.Parameter(torch.zeros(head_dim, rank))
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_v)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original fused QKV
        qkv = self.original_qkv(x)  # (B, N, 3D)
        
        # LoRA additions for Q and V
        x_drop = self.dropout(x)
        lora_q = (x_drop @ self.lora_A_q.T @ self.lora_B_q.T) * self.scaling
        lora_v = (x_drop @ self.lora_A_v.T @ self.lora_B_v.T) * self.scaling
        
        # Add LoRA to Q (first head_dim) and V (last head_dim), skip K (middle)
        qkv[:, :, :self.head_dim] += lora_q
        qkv[:, :, 2 * self.head_dim:] += lora_v
        
        return qkv
    
    @property
    def trainable_params(self):
        return (self.lora_A_q.numel() + self.lora_B_q.numel() +
                self.lora_A_v.numel() + self.lora_B_v.numel())


# ────────────────────── Bottleneck Adapters ───────────────────────────────────

class AdapterModule(nn.Module):
    """Houlsby bottleneck adapter: LayerNorm -> Down -> GeLU -> Up -> Residual.
    
    Inserted after MHA and FFN sublayers in each transformer block.
    """
    
    def __init__(self, embed_dim: int, bottleneck_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.down = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, embed_dim)
        
        # Init near-identity: small down weights, zero up weights
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.layer_norm(x)
        out = self.down(out)
        out = self.act(out)
        out = self.up(out)
        return residual + out
    
    @property
    def trainable_params(self):
        return sum(p.numel() for p in self.parameters())


class AdaptedBlock(nn.Module):
    """Wraps a timm ViT block to insert adapters after MHA and FFN."""
    
    def __init__(self, original_block: nn.Module, embed_dim: int, bottleneck_dim: int):
        super().__init__()
        self.original_block = original_block
        self.adapter_attn = AdapterModule(embed_dim, bottleneck_dim)
        self.adapter_ffn  = AdapterModule(embed_dim, bottleneck_dim)
        
        # Freeze original block
        for p in self.original_block.parameters():
            p.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MHA sublayer
        attn_out = self.original_block.attn(self.original_block.norm1(x))
        x = x + self.original_block.drop_path1(self.original_block.ls1(attn_out))
        x = self.adapter_attn(x)
        
        # FFN sublayer
        ffn_out = self.original_block.mlp(self.original_block.norm2(x))
        x = x + self.original_block.drop_path2(self.original_block.ls2(ffn_out))
        x = self.adapter_ffn(x)
        
        return x


def inject_adapters(model: nn.Module, embed_dim: int, bottleneck_dim: int,
                    target_blocks: list = None) -> int:
    """Inject Houlsby adapters into ViT transformer blocks.
    
    Replaces each targeted block with an AdaptedBlock wrapper.
    Original block parameters remain frozen; only adapter parameters train.
    
    Returns:
        Total number of adapter trainable parameters added.
    """
    total_params = 0
    new_blocks = nn.ModuleList()
    
    for block_idx, block in enumerate(model.blocks):
        if target_blocks is not None and block_idx not in target_blocks:
            new_blocks.append(block)
        else:
            adapted = AdaptedBlock(block, embed_dim, bottleneck_dim)
            new_blocks.append(adapted)
            total_params += adapted.adapter_attn.trainable_params
            total_params += adapted.adapter_ffn.trainable_params
    
    model.blocks = new_blocks
    return total_params


# ────────────────────── Visual Prompt Tuning ──────────────────────────────────

class VPTShallowEncoder(nn.Module):
    """Wraps a timm ViT to prepend learnable prompts at the input layer only.
    
    Prompt tokens are concatenated to the patch embeddings after positional
    encoding, before the first transformer block. They participate in
    self-attention but are stripped from the output before global pooling.
    """
    
    def __init__(self, encoder: nn.Module, num_prompts: int, embed_dim: int):
        super().__init__()
        self.encoder = encoder
        self.num_prompts = num_prompts
        
        # Learnable prompt tokens
        self.prompt_tokens = nn.Parameter(
            torch.zeros(1, num_prompts, embed_dim)
        )
        nn.init.normal_(self.prompt_tokens, std=0.02)
        
        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding + position encoding (timm internals)
        x = self.encoder.patch_embed(x)
        
        # Prepend CLS token if present
        if hasattr(self.encoder, 'cls_token') and self.encoder.cls_token is not None:
            cls_tokens = self.encoder.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        
        # Prepend prompt tokens
        prompts = self.prompt_tokens.expand(B, -1, -1)
        x = torch.cat([prompts, x], dim=1)
        
        # Transformer blocks
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        
        # Remove prompt tokens before pooling
        x = x[:, self.num_prompts:, :]
        
        # Global average pooling (matching timm's global_pool='avg')
        if hasattr(self.encoder, 'cls_token') and self.encoder.cls_token is not None:
            # CLS token is at position 0 after removing prompts
            x = x[:, 1:, :].mean(dim=1)  # Average over patch tokens
        else:
            x = x.mean(dim=1)
        
        return x
    
    @property
    def trainable_params(self):
        return self.prompt_tokens.numel()


class VPTDeepEncoder(nn.Module):
    """Wraps a timm ViT to prepend learnable prompts at every layer.
    
    Each transformer block gets its own set of fresh prompt tokens,
    allowing layer-specific learnable context.
    """
    
    def __init__(self, encoder: nn.Module, num_prompts: int, embed_dim: int,
                 num_layers: int):
        super().__init__()
        self.encoder = encoder
        self.num_prompts = num_prompts
        self.num_layers = num_layers
        
        # Per-layer prompt tokens
        self.prompt_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, num_prompts, embed_dim))
            for _ in range(num_layers)
        ])
        for pt in self.prompt_tokens:
            nn.init.normal_(pt, std=0.02)
        
        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding + position encoding
        x = self.encoder.patch_embed(x)
        
        if hasattr(self.encoder, 'cls_token') and self.encoder.cls_token is not None:
            cls_tokens = self.encoder.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        
        # Process each block with its own prompts
        for layer_idx, block in enumerate(self.encoder.blocks):
            prompts = self.prompt_tokens[layer_idx].expand(B, -1, -1)
            x = torch.cat([prompts, x], dim=1)
            x = block(x)
            # Remove prompts after this layer (next layer gets fresh ones)
            x = x[:, self.num_prompts:, :]
        
        x = self.encoder.norm(x)
        
        # Global average pooling
        if hasattr(self.encoder, 'cls_token') and self.encoder.cls_token is not None:
            x = x[:, 1:, :].mean(dim=1)
        else:
            x = x.mean(dim=1)
        
        return x
    
    @property
    def trainable_params(self):
        return sum(pt.numel() for pt in self.prompt_tokens)


def inject_vpt_shallow(encoder: nn.Module, num_prompts: int,
                       embed_dim: int) -> Tuple[nn.Module, int]:
    """Wrap encoder with VPT-Shallow. Returns (wrapped_encoder, num_params)."""
    wrapped = VPTShallowEncoder(encoder, num_prompts, embed_dim)
    return wrapped, wrapped.trainable_params


def inject_vpt_deep(encoder: nn.Module, num_prompts: int, embed_dim: int,
                    num_layers: int) -> Tuple[nn.Module, int]:
    """Wrap encoder with VPT-Deep. Returns (wrapped_encoder, num_params)."""
    wrapped = VPTDeepEncoder(encoder, num_prompts, embed_dim, num_layers)
    return wrapped, wrapped.trainable_params


# ────────────────────── BitFit ────────────────────────────────────────────────

def inject_bitfit(model: nn.Module) -> int:
    """Unfreeze only bias terms across the entire model.
    
    Returns:
        Number of trainable bias parameters.
    """
    total_bias_params = 0
    for name, param in model.named_parameters():
        if "bias" in name:
            param.requires_grad = True
            total_bias_params += param.numel()
        else:
            param.requires_grad = False
    return total_bias_params


# ============================================================================
# Model Construction
# ============================================================================

class PEFTClassifier(nn.Module):
    """Frozen encoder + PEFT modifications + linear classification head.
    
    This is the standard model for all Stage 5 experiments.
    The encoder may be modified by PEFT (LoRA, Adapters) or wrapped (VPT).
    The classification head is always a single nn.Linear layer.
    """
    
    def __init__(self, encoder: nn.Module, embed_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)


def count_params(model: nn.Module) -> Dict[str, int]:
    """Count parameters in a model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    frozen    = total - trainable
    return {
        "trainable": trainable,
        "total": total,
        "frozen": frozen,
        "pct_trainable": round(100.0 * trainable / total, 4) if total > 0 else 0,
    }


def print_param_summary(model: nn.Module, label: str):
    """Print a formatted parameter count summary."""
    c = count_params(model)
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  Trainable:   {c['trainable']:>12,}")
    print(f"  Frozen:      {c['frozen']:>12,}")
    print(f"  Total:       {c['total']:>12,}")
    print(f"  % Trainable: {c['pct_trainable']:>11.4f}%")
    print(f"{'─'*50}")
    return c


def build_peft_model(
    method: str,
    checkpoint_path: Path,
    model_name: str,
    embed_dim: int,
    num_classes: int,
    device: torch.device,
    # Method-specific kwargs
    rank: int = 8,
    bottleneck_dim: int = 64,
    num_prompts: int = 50,
    num_layers: int = 32,
    lora_alpha: float = None,
    lora_dropout: float = 0.0,
    target_blocks: list = None,
) -> Tuple[nn.Module, Dict[str, int]]:
    """One-line PEFT model construction.
    
    Args:
        method: One of 'lora', 'adapter', 'vpt_shallow', 'vpt_deep',
                'bitfit', 'linear_probe', 'full_ft'
        checkpoint_path: Path to encoder checkpoint (I-JEPA or Leaf-JEPA)
        model_name: timm model name (e.g., 'vit_huge_patch14_224')
        embed_dim: Embedding dimension (e.g., 1280 for ViT-H/14)
        num_classes: Number of output classes
        device: torch device
        **kwargs: Method-specific hyperparameters
    
    Returns:
        (model, param_counts) tuple
    """
    print(f"\n Building PEFT model: method={method}")
    
    # Load frozen encoder
    encoder = load_ijepa_encoder(checkpoint_path, model_name, embed_dim,
                                 device, freeze=True)
    
    peft_params = 0
    
    if method == "lora":
        peft_params = inject_lora(encoder, rank=rank, alpha=lora_alpha,
                                  dropout=lora_dropout, target_blocks=target_blocks)
        model = PEFTClassifier(encoder, embed_dim, num_classes).to(device)
        
    elif method == "adapter":
        peft_params = inject_adapters(encoder, embed_dim, bottleneck_dim,
                                      target_blocks=target_blocks)
        model = PEFTClassifier(encoder, embed_dim, num_classes).to(device)
        
    elif method == "vpt_shallow":
        encoder, peft_params = inject_vpt_shallow(encoder, num_prompts, embed_dim)
        model = PEFTClassifier(encoder, embed_dim, num_classes).to(device)
        
    elif method == "vpt_deep":
        encoder, peft_params = inject_vpt_deep(encoder, num_prompts, embed_dim,
                                               num_layers)
        model = PEFTClassifier(encoder, embed_dim, num_classes).to(device)
        
    elif method == "bitfit":
        model = PEFTClassifier(encoder, embed_dim, num_classes).to(device)
        peft_params = inject_bitfit(model.encoder)
        
    elif method == "linear_probe":
        # Encoder fully frozen, only head trains
        model = PEFTClassifier(encoder, embed_dim, num_classes).to(device)
        
    elif method == "full_ft":
        # Unfreeze everything
        for p in encoder.parameters():
            p.requires_grad = True
        model = PEFTClassifier(encoder, embed_dim, num_classes).to(device)
        
    else:
        raise ValueError(f"Unknown PEFT method: {method}")
    
    # Head is always trainable
    for p in model.head.parameters():
        p.requires_grad = True
    
    params = print_param_summary(model, f"{method.upper()} Classifier")
    return model, params


# ============================================================================
# Training Infrastructure
# ============================================================================

class WarmupCosineScheduler:
    """Linear warmup then cosine decay to 0."""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0
    
    def step(self):
        self._step += 1
        if self._step <= self.warmup_steps:
            # Linear warmup
            scale = self._step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (self._step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale
    
    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


class EarlyStopping:
    """Early stopping based on validation metric."""
    
    def __init__(self, patience: int = 10, mode: str = "max",
                 min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False
    
    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, device: torch.device,
                    scaler: GradScaler = None, use_amp: bool = True,
                    gradient_clip: float = None,
                    scheduler: WarmupCosineScheduler = None) -> Dict:
    """Train for one epoch. Returns dict with loss, accuracy, time."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp and scaler is not None:
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if gradient_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    epoch_time = time.time() - start_time
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "time_s": epoch_time,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: torch.device,
             num_classes: int = 38, use_amp: bool = True) -> Dict:
    """Evaluate model. Returns dict with loss, accuracy, macro_f1, per_class_f1, etc."""
    model.eval()
    total_loss = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if use_amp:
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)
        
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total += labels.size(0)
    
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    
    return {
        "loss": total_loss / total,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
    }


# ============================================================================
# Unified Training Entry Point
# ============================================================================

def train_peft(
    # Method specification
    method: str,
    checkpoint_path: Path,
    # Config values (pass from config_stage5)
    pv_root: Path,
    splits_dir: Path,
    norm_mean: list,
    norm_std: list,
    model_name: str,
    embed_dim: int,
    num_classes: int,
    # Experiment params
    fraction: float = 1.0,
    seed: int = 42,
    lr: float = 3e-4,
    # Training params
    batch_size: int = 32,
    max_epochs: int = 50,
    patience: int = 10,
    weight_decay: float = 0.01,
    warmup_fraction: float = 0.10,
    use_amp: bool = True,
    gradient_clip: float = 1.0,
    num_workers: int = 4,
    # PEFT kwargs
    rank: int = 8,
    bottleneck_dim: int = 64,
    num_prompts: int = 50,
    num_layers: int = 32,
    lora_alpha: float = None,
    lora_dropout: float = 0.0,
    target_blocks: list = None,
    # Class weights
    class_weights_path: Path = None,
    # Outputs
    save_dir: Path = None,
    run_name: str = None,
    # WandB
    wandb_project: str = None,
    wandb_entity: str = None,
    wandb_group: str = None,
) -> Dict:
    """Unified entry point for any PEFT experiment.
    
    Handles the full lifecycle: seed → data → model → train → evaluate → save.
    
    Returns:
        Dict with all results, params, and metadata.
    """
    device = get_device()
    set_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"  PEFT Experiment: {method}")
    print(f"  Fraction: {fraction:.2f} | Seed: {seed} | LR: {lr}")
    print(f"{'='*60}")
    
    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(
        pv_root, splits_dir, norm_mean, norm_std,
        fraction=fraction, seed=seed,
        batch_size=batch_size, num_workers=num_workers,
    )
    
    # ── Model ─────────────────────────────────────────────────────────────
    model, param_counts = build_peft_model(
        method=method,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        embed_dim=embed_dim,
        num_classes=num_classes,
        device=device,
        rank=rank,
        bottleneck_dim=bottleneck_dim,
        num_prompts=num_prompts,
        num_layers=num_layers,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_blocks=target_blocks,
    )
    
    # ── Criterion (with class weights) ────────────────────────────────────
    if class_weights_path and Path(class_weights_path).exists():
        weights = load_class_weights(class_weights_path, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"  Using class weights from {Path(class_weights_path).name}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # ── Optimizer ─────────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    
    # ── Scheduler ─────────────────────────────────────────────────────────
    total_steps = max_epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_fraction)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    
    # ── AMP ───────────────────────────────────────────────────────────────
    scaler = GradScaler() if use_amp and device.type == "cuda" else None
    
    # ── WandB ─────────────────────────────────────────────────────────────
    wandb_run = None
    if wandb_project:
        import wandb
        config_dict = {
            "method": method, "fraction": fraction, "seed": seed, "lr": lr,
            "batch_size": batch_size, "max_epochs": max_epochs,
            "patience": patience, "weight_decay": weight_decay,
            "warmup_fraction": warmup_fraction, "gradient_clip": gradient_clip,
            "rank": rank, "bottleneck_dim": bottleneck_dim,
            "num_prompts": num_prompts, "target_blocks": str(target_blocks),
            "checkpoint": str(checkpoint_path),
            **param_counts,
        }
        wandb_run = wandb.init(
            project=wandb_project, entity=wandb_entity,
            group=wandb_group, name=run_name,
            config=config_dict, reinit=True,
        )
    
    # ── Training loop ─────────────────────────────────────────────────────
    early_stop = EarlyStopping(patience=patience, mode="max")
    best_model_state = None
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_accuracy": [],
               "train_accuracy": [], "lr": [], "epoch_time": []}
    
    # Track peak VRAM
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    for epoch in range(1, max_epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, use_amp=use_amp, gradient_clip=gradient_clip,
            scheduler=scheduler,
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device,
                               num_classes=num_classes, use_amp=use_amp)
        
        # Record history
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["lr"].append(current_lr)
        history["epoch_time"].append(train_metrics["time_s"])
        
        # WandB logging
        if wandb_run:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/accuracy": train_metrics["accuracy"],
                "val/loss": val_metrics["loss"],
                "val/macro_f1": val_metrics["macro_f1"],
                "val/accuracy": val_metrics["accuracy"],
                "lr": current_lr,
                "epoch_time_s": train_metrics["time_s"],
            })
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{max_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val F1: {val_metrics['macro_f1']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {train_metrics['time_s']:.1f}s")
        
        # Early stopping
        if early_stop(val_metrics["macro_f1"], epoch):
            print(f"  ⏹ Early stopping at epoch {epoch} "
                  f"(best val F1: {early_stop.best_score:.4f} at epoch {early_stop.best_epoch})")
            break
        
        # Save best model
        if early_stop.best_epoch == epoch:
            best_model_state = copy.deepcopy(model.state_dict())
    
    # ── Load best model & evaluate on TEST set ────────────────────────────
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    test_metrics = evaluate(model, test_loader, criterion, device,
                            num_classes=num_classes, use_amp=use_amp)
    
    print(f"\n  ── TEST RESULTS ──")
    print(f"  Macro-F1:  {test_metrics['macro_f1']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Best epoch: {early_stop.best_epoch}")
    
    # ── Compute profiling ─────────────────────────────────────────────────
    peak_vram = 0
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / 1e6  # MB
    
    avg_epoch_time = np.mean(history["epoch_time"][1:]) if len(history["epoch_time"]) > 1 \
        else history["epoch_time"][0]
    
    # ── Assemble results dict ─────────────────────────────────────────────
    results = {
        "method": method,
        "hyperparams": {
            "rank": rank, "bottleneck_dim": bottleneck_dim,
            "num_prompts": num_prompts, "target_blocks": str(target_blocks),
            "lora_alpha": lora_alpha, "lora_dropout": lora_dropout,
        },
        "training_config": {
            "lr": lr, "optimizer": "AdamW", "weight_decay": weight_decay,
            "scheduler": "cosine_warmup", "warmup_fraction": warmup_fraction,
            "max_epochs": max_epochs, "early_stop_patience": patience,
            "batch_size": batch_size, "fraction": fraction, "seed": seed,
            "use_amp": use_amp, "gradient_clip": gradient_clip,
        },
        "encoder_checkpoint": str(checkpoint_path),
        "param_count": param_counts,
        "results": {
            "test_macro_f1": test_metrics["macro_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "val_macro_f1": early_stop.best_score,
            "best_epoch": early_stop.best_epoch,
            "total_epochs_run": len(history["train_loss"]),
            "per_class_f1": test_metrics["per_class_f1"],
        },
        "compute": {
            "peak_vram_mb": round(peak_vram, 1),
            "avg_epoch_time_s": round(avg_epoch_time, 2),
            "gpu_model": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        },
        "history": {k: [round(v, 6) for v in vals] for k, vals in history.items()},
    }
    
    # ── Save results ──────────────────────────────────────────────────────
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON results
        fname = run_name or f"{method}_frac{fraction:.2f}_seed{seed}"
        json_path = save_dir / f"{fname}.json"
        
        # Convert numpy to python for JSON serialisation
        results_serialisable = json.loads(
            json.dumps(results, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
        )
        json_path.write_text(json.dumps(results_serialisable, indent=2))
        print(f"  Results saved to {json_path}")
        
        # Confusion matrix figure
        cm_path = save_dir / f"{fname}_confusion.png"
        plot_confusion_matrix(
            test_metrics["confusion_matrix"],
            save_path=cm_path,
            title=f"{method} (frac={fraction}, seed={seed})"
        )
        results["results"]["confusion_matrix_path"] = str(cm_path)
        
        # Save best model checkpoint
        ckpt_path = save_dir / f"{fname}_best.pt"
        torch.save({
            "model_state_dict": best_model_state or model.state_dict(),
            "method": method,
            "param_counts": param_counts,
            "test_macro_f1": test_metrics["macro_f1"],
        }, ckpt_path)
    
    # ── WandB cleanup ─────────────────────────────────────────────────────
    if wandb_run:
        wandb.log({
            "test/macro_f1": test_metrics["macro_f1"],
            "test/accuracy": test_metrics["accuracy"],
            "best_epoch": early_stop.best_epoch,
            "peak_vram_mb": peak_vram,
        })
        wandb.finish()
    
    return results


# ============================================================================
# Feature Extraction & kNN (for Set 3: Cross-Domain)
# ============================================================================

@torch.no_grad()
def extract_features(model_or_encoder: nn.Module, loader: DataLoader,
                     device: torch.device, use_amp: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features from a frozen encoder or PEFTClassifier.
    
    If model is PEFTClassifier, uses model.encoder.
    Otherwise uses model directly.
    
    Returns:
        (features, labels) as numpy arrays.
    """
    if hasattr(model_or_encoder, 'encoder'):
        encoder = model_or_encoder.encoder
    else:
        encoder = model_or_encoder
    
    encoder.eval()
    all_features = []
    all_labels = []
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        if use_amp:
            with autocast(device_type="cuda", dtype=torch.float16):
                features = encoder(images)
        else:
            features = encoder(images)
        all_features.append(features.float().cpu().numpy())
        all_labels.append(labels.numpy())
    
    return np.concatenate(all_features), np.concatenate(all_labels)


def knn_evaluate(train_features: np.ndarray, train_labels: np.ndarray,
                 test_features: np.ndarray, test_labels: np.ndarray,
                 k_values: list = [5, 10, 20],
                 metric: str = "cosine") -> Dict:
    """k-NN evaluation on extracted features.
    
    Returns dict with results for each k value.
    """
    # Normalise features for cosine similarity
    from sklearn.preprocessing import normalize
    train_norm = normalize(train_features)
    test_norm  = normalize(test_features)
    
    results = {}
    for k in k_values:
        knn = KNeighborsClassifier(
            n_neighbors=k, metric="cosine", algorithm="brute"
        )
        knn.fit(train_norm, train_labels)
        preds = knn.predict(test_norm)
        
        macro_f1 = f1_score(test_labels, preds, average="macro", zero_division=0)
        accuracy = accuracy_score(test_labels, preds)
        per_class = f1_score(test_labels, preds, average=None, zero_division=0)
        
        results[f"k={k}"] = {
            "macro_f1": macro_f1,
            "accuracy": accuracy,
            "per_class_f1": per_class.tolist(),
        }
        print(f"  k={k:3d} | Macro-F1: {macro_f1:.4f} | Accuracy: {accuracy:.4f}")
    
    # Best k
    best_k = max(results, key=lambda x: results[x]["macro_f1"])
    results["best_k"] = best_k
    results["best_macro_f1"] = results[best_k]["macro_f1"]
    
    return results


# ============================================================================
# Computational Profiling
# ============================================================================

@torch.no_grad()
def profile_inference(model: nn.Module, device: torch.device,
                      input_shape: tuple = (1, 3, 224, 224),
                      n_warmup: int = 50, n_runs: int = 200,
                      use_amp: bool = True) -> float:
    """Profile inference latency in ms/image.
    
    Returns average inference time in milliseconds.
    """
    model.eval()
    dummy = torch.randn(*input_shape, device=device)
    
    # Warmup
    for _ in range(n_warmup):
        if use_amp:
            with autocast(device_type="cuda", dtype=torch.float16):
                _ = model(dummy)
        else:
            _ = model(dummy)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(n_runs):
        if use_amp:
            with autocast(device_type="cuda", dtype=torch.float16):
                _ = model(dummy)
        else:
            _ = model(dummy)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    total_time = time.time() - start
    ms_per_image = (total_time / n_runs) * 1000
    return round(ms_per_image, 2)


# ============================================================================
# Plotting
# ============================================================================

def plot_confusion_matrix(cm: np.ndarray, save_path: Path = None,
                          title: str = "", class_names: list = None,
                          figsize: tuple = (14, 12)):
    """Plot and optionally save a confusion matrix."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalise for display
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    
    sns.heatmap(cm_norm, annot=False, fmt=".2f", cmap="YlOrRd",
                xticklabels=class_names or range(cm.shape[0]),
                yticklabels=class_names or range(cm.shape[0]),
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion Matrix")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Confusion matrix saved to {save_path}")
    plt.close(fig)


def plot_label_efficiency(results_by_method: Dict, save_path: Path = None,
                          title: str = "Label Efficiency Curves"):
    """Plot label efficiency curves for multiple methods.
    
    Args:
        results_by_method: {method_name: {fraction: {mean, std}}}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colours = plt.cm.Set2(np.linspace(0, 1, len(results_by_method)))
    
    for (method, data), colour in zip(results_by_method.items(), colours):
        fractions = sorted(data.keys())
        means = [data[f]["mean"] for f in fractions]
        stds  = [data[f]["std"] for f in fractions]
        
        ax.plot(fractions, means, "o-", label=method, color=colour, linewidth=2)
        ax.fill_between(fractions,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=colour)
    
    ax.set_xlabel("Label Fraction", fontsize=12)
    ax.set_ylabel("Macro-F1", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.05, 0.10, 0.25, 0.50, 1.0])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"{x:.0%}" if x >= 0.1 else f"{x:.0%}"
    ))
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Label efficiency plot saved to {save_path}")
    plt.close(fig)


def plot_pareto(all_results: list, save_path: Path = None,
                title: str = "Pareto Frontier: Accuracy vs Parameters"):
    """Plot Pareto frontier of accuracy vs trainable parameters.
    
    Args:
        all_results: list of dicts with keys: method, trainable_params, macro_f1, label
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    method_colours = {
        "lora": "#e74c3c", "adapter": "#3498db", "vpt_shallow": "#2ecc71",
        "vpt_deep": "#9b59b6", "bitfit": "#f39c12", "linear_probe": "#95a5a6",
        "full_ft": "#34495e",
    }
    
    for res in all_results:
        colour = method_colours.get(res["method"], "#7f8c8d")
        ax.scatter(res["trainable_params"], res["macro_f1"],
                   c=colour, s=80, alpha=0.7, edgecolors="white", linewidth=0.5,
                   label=res.get("label", ""))
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower right", fontsize=9)
    
    ax.set_xlabel("Trainable Parameters", fontsize=12)
    ax.set_ylabel("Macro-F1", fontsize=12)
    ax.set_xscale("log")
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Result Aggregation
# ============================================================================

def aggregate_seed_results(results_list: list) -> Dict:
    """Aggregate results across multiple seeds.
    
    Args:
        results_list: list of result dicts from train_peft() with same config but different seeds
    
    Returns:
        Dict with mean, std, and per-seed values for key metrics.
    """
    f1_scores = [r["results"]["test_macro_f1"] for r in results_list]
    accuracies = [r["results"]["test_accuracy"] for r in results_list]
    epochs = [r["results"]["best_epoch"] for r in results_list]
    vrams = [r["compute"]["peak_vram_mb"] for r in results_list]
    times = [r["compute"]["avg_epoch_time_s"] for r in results_list]
    
    return {
        "macro_f1": {"mean": np.mean(f1_scores), "std": np.std(f1_scores),
                     "per_seed": f1_scores},
        "accuracy": {"mean": np.mean(accuracies), "std": np.std(accuracies),
                     "per_seed": accuracies},
        "best_epoch": {"mean": np.mean(epochs), "std": np.std(epochs),
                       "per_seed": [int(e) for e in epochs]},
        "peak_vram_mb": {"mean": np.mean(vrams), "std": np.std(vrams)},
        "avg_epoch_time_s": {"mean": np.mean(times), "std": np.std(times)},
        "param_count": results_list[0]["param_count"],
        "method": results_list[0]["method"],
        "hyperparams": results_list[0]["hyperparams"],
        "training_config": {k: v for k, v in results_list[0]["training_config"].items()
                            if k != "seed"},
    }


def save_results(results: dict, path: Path):
    """Save results dict to JSON (handles numpy types)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = json.loads(
        json.dumps(results, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    )
    path.write_text(json.dumps(serialisable, indent=2))
    print(f"  Saved: {path}")


def load_results(path: Path) -> dict:
    """Load results from JSON."""
    return json.loads(Path(path).read_text())
