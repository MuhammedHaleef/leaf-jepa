"""
peft_utils.py
=============
All PEFT method implementations and shared training infrastructure for Stage 5.

Function naming follows baseline_utils.py (Stage 3) conventions for consistency:
  - count_parameters()        (not count_params)
  - print_parameter_summary() (not print_param_summary)
  - profile_vram()            (same as Stage 3)
  - plot_training_curves()    (same as Stage 3)

Contents:
    PEFT Implementations:
        - LoRALinear / LoRAQKV / inject_lora()
        - AdapterModule / AdaptedBlock / inject_adapters()
        - VPTShallowEncoder / inject_vpt_shallow()
        - VPTDeepEncoder / inject_vpt_deep()
        - inject_bitfit()

    Model Building:
        - PEFTClassifier / build_peft_model() / count_parameters()

    Training:
        - train_one_epoch() / evaluate()
        - WarmupCosineScheduler / EarlyStopping
        - train_peft()

    Data:
        - get_transforms() / PlantVillageDataset / PlantDocDataset
        - load_split() / load_class_weights() / build_dataloaders()

    Evaluation & Profiling:
        - extract_features() / knn_evaluate()
        - profile_vram() / profile_inference()

    Plotting:
        - plot_confusion_matrix() / plot_training_curves()
        - plot_label_efficiency() / plot_pareto()

    Utilities:
        - set_seed() / get_device()
        - save_results() / load_results()
        - load_ijepa_encoder() / aggregate_seed_results()
"""

import os
import json
import time
import math
import copy
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report
)
from sklearn.neighbors import KNeighborsClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import timm

# ── PyTorch 2.x compatible AMP imports ────────────────────────────────────────
try:
    from torch.amp import GradScaler, autocast       # PyTorch >= 2.0
    _AMP_DTYPE_ARG = {"device_type": "cuda"}
except ImportError:
    from torch.cuda.amp import GradScaler, autocast   # PyTorch < 2.0
    _AMP_DTYPE_ARG = {"device_type": "cuda", "dtype": torch.float16}

_AMP_DEVICE = "cuda"


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
        print("Warning: CUDA not available, using CPU (will be very slow)")
    return dev


# ============================================================================
# Transforms (3 strictly separated pipelines from Stage 2)
# ============================================================================

def get_transforms(phase: str, norm_mean: list, norm_std: list,
                   image_crop: int = 224, image_resize: int = 256):
    """Return transform pipeline for the given phase.

    Phases:
        'finetune' -- standard augmentation (Stage 5 training)
        'eval'     -- deterministic (all evaluation)
    """
    if phase == "finetune":
        return transforms.Compose([
            transforms.Resize(image_resize),
            transforms.RandomCrop(image_crop),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
    elif phase == "eval":
        return transforms.Compose([
            transforms.Resize(image_resize),
            transforms.CenterCrop(image_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
    else:
        raise ValueError(f"Unknown phase '{phase}'. Use 'finetune' or 'eval'.")


# ============================================================================
# Dataset & Data Loading
# ============================================================================

class PlantVillageDataset(Dataset):
    """PlantVillage dataset that loads from a split CSV."""

    def __init__(self, root: Path, split_csv: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.df = pd.read_csv(split_csv)
        self.paths = self.df["filepath"].tolist()
        self.labels = self.df["label_idx"].tolist()
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
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_split(splits_dir: Path, split_name: str,
               transform=None, pv_root: Path = None) -> Union[Path, "PlantVillageDataset"]:
    """Load a split CSV, optionally returning a ready Dataset.

    Two calling conventions (Stage 3 compatible):
        load_split(splits_dir, "train")                    -> returns Path
        load_split(splits_dir, "train", transform, pv_root) -> returns Dataset
    """
    csv_path = Path(splits_dir) / f"{split_name}.csv"
    assert csv_path.exists(), f"Split not found: {csv_path}"
    if transform is not None and pv_root is not None:
        return PlantVillageDataset(pv_root, csv_path, transform)
    return csv_path


def load_class_weights(path: Path, device: torch.device) -> torch.Tensor:
    """Load inverse-frequency class weights from Stage 2 JSON."""
    weights_dict = json.loads(Path(path).read_text())
    
    if isinstance(weights_dict, dict):
        # Extract the float values in the order they appear in the JSON
        # Ensure the JSON order matches your dataset class indices!
        weight_list = [float(v) for v in weights_dict.values()]
        w = torch.tensor(weight_list, dtype=torch.float32)
    else:
        w = torch.tensor(weights_dict, dtype=torch.float32)
        
    return w.to(device)


def build_dataloaders(
    pv_root: Path, splits_dir: Path,
    norm_mean: list, norm_std: list,
    fraction: float = 1.0, seed: int = 42,
    batch_size: int = 32, num_workers: int = 4,
    image_crop: int = 224, image_resize: int = 256,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders for a given label fraction."""
    train_tf = get_transforms("finetune", norm_mean, norm_std, image_crop, image_resize)
    eval_tf = get_transforms("eval", norm_mean, norm_std, image_crop, image_resize)

    train_split = f"fractions/frac_{fraction:.2f}_seed{seed}" if fraction < 1.0 else "plantvillage_train"

    train_ds = PlantVillageDataset(pv_root, load_split(splits_dir, train_split), train_tf)
    val_ds = PlantVillageDataset(pv_root, load_split(splits_dir, "plantvillage_val"), eval_tf)
    test_ds = PlantVillageDataset(pv_root, load_split(splits_dir, "plantvillage_test"), eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
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
    """Load I-JEPA or Leaf-JEPA encoder from checkpoint."""
    encoder = timm.create_model(model_name, pretrained=False,
                                num_classes=0, global_pool="", no_embed_class=True)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    if "target_encoder" in ckpt:
        state_dict = ckpt["target_encoder"]
    elif "encoder" in ckpt:
        state_dict = ckpt["encoder"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "").replace("backbone.", "")] = v

    msg = encoder.load_state_dict(cleaned, strict=False)
    if msg.missing_keys:
        print(f"  Warning: Missing keys: {msg.missing_keys[:5]}...")
    if msg.unexpected_keys:
        print(f"  Warning: Unexpected keys: {msg.unexpected_keys[:5]}...")

    if freeze:
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()

    encoder = encoder.to(device)
    print(f"  Encoder loaded from {Path(checkpoint_path).name} "
          f"({'frozen' if freeze else 'trainable'})")
    return encoder


# ============================================================================
# PEFT: LoRA
# ============================================================================

class LoRALinear(nn.Module):
    """Low-Rank Adaptation for a frozen nn.Linear: W' = W + BA * (alpha/rank)."""

    def __init__(self, original: nn.Linear, rank: int,
                 alpha: float = None, dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank

        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(rank, original.in_features))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.original(x)
        return out + self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling

    def merge_weights(self):
        with torch.no_grad():
            self.original.weight.data += (self.lora_B @ self.lora_A * self.scaling)

    @property
    def trainable_params(self):
        return self.lora_A.numel() + self.lora_B.numel()


class LoRAQKV(nn.Module):
    """LoRA for timm's fused QKV projection. Adds LoRA to Q and V only."""

    def __init__(self, original_qkv: nn.Linear, rank: int,
                 alpha: float = None, dropout: float = 0.0):
        super().__init__()
        self.original_qkv = original_qkv
        original_qkv.weight.requires_grad = False
        if original_qkv.bias is not None:
            original_qkv.bias.requires_grad = False

        in_dim = original_qkv.in_features
        head_dim = original_qkv.out_features // 3

        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank
        self.head_dim = head_dim

        self.lora_A_q = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B_q = nn.Parameter(torch.zeros(head_dim, rank))
        self.lora_A_v = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B_v = nn.Parameter(torch.zeros(head_dim, rank))
        nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q)
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_v)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        qkv = self.original_qkv(x)
        x_drop = self.dropout(x)
        lora_q = (x_drop @ self.lora_A_q.T @ self.lora_B_q.T) * self.scaling
        lora_v = (x_drop @ self.lora_A_v.T @ self.lora_B_v.T) * self.scaling
        # Clone to avoid in-place modification on autograd graph
        qkv = qkv.clone()
        qkv[:, :, :self.head_dim] = qkv[:, :, :self.head_dim] + lora_q
        qkv[:, :, 2 * self.head_dim:] = qkv[:, :, 2 * self.head_dim:] + lora_v
        return qkv

    @property
    def trainable_params(self):
        return (self.lora_A_q.numel() + self.lora_B_q.numel() +
                self.lora_A_v.numel() + self.lora_B_v.numel())


def inject_lora(model: nn.Module, rank: int, alpha: float = None,
                dropout: float = 0.0, target_blocks: list = None) -> int:
    """Inject LoRA into Q,V of a ViT encoder. Returns total LoRA params."""
    total = 0
    for idx, block in enumerate(model.blocks):
        if target_blocks is not None and idx not in target_blocks:
            continue
        lora_qkv = LoRAQKV(block.attn.qkv, rank=rank, alpha=alpha, dropout=dropout)
        block.attn.qkv = lora_qkv
        total += lora_qkv.trainable_params
    return total


# ============================================================================
# PEFT: Bottleneck Adapters (Houlsby)
# ============================================================================

class AdapterModule(nn.Module):
    """Bottleneck adapter: LayerNorm -> Down -> GeLU -> Up -> Residual."""

    def __init__(self, embed_dim: int, bottleneck_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.down = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, embed_dim)
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(self.layer_norm(x))))

    @property
    def trainable_params(self):
        return sum(p.numel() for p in self.parameters())


class AdaptedBlock(nn.Module):
    """Wraps a timm ViT block with adapters after MHA and FFN.

    Detects timm version internals (drop_path1/ls1 vs older API)
    for compatibility across timm >= 0.6 through >= 1.0.
    """

    def __init__(self, original_block: nn.Module, embed_dim: int, bottleneck_dim: int):
        super().__init__()
        self.original_block = original_block
        self.adapter_attn = AdapterModule(embed_dim, bottleneck_dim)
        self.adapter_ffn = AdapterModule(embed_dim, bottleneck_dim)
        for p in self.original_block.parameters():
            p.requires_grad = False

        # Detect timm block internals
        self._has_ls = hasattr(original_block, "ls1")
        self._has_drop_path = hasattr(original_block, "drop_path1")
        self._has_stochastic_depth = hasattr(original_block, "drop_path")

    def _apply_residual(self, x, sublayer_out, drop_path_fn, ls_fn):
        """Apply residual with optional drop_path and layer_scale."""
        if ls_fn is not None:
            sublayer_out = ls_fn(sublayer_out)
        if drop_path_fn is not None:
            sublayer_out = drop_path_fn(sublayer_out)
        return x + sublayer_out

    def forward(self, x):
        # ── MHA sublayer ──
        attn_out = self.original_block.attn(self.original_block.norm1(x))

        dp1 = getattr(self.original_block, "drop_path1", None)
        ls1 = getattr(self.original_block, "ls1", None)
        x = self._apply_residual(x, attn_out, dp1, ls1)
        x = self.adapter_attn(x)

        # ── FFN sublayer ──
        ffn_out = self.original_block.mlp(self.original_block.norm2(x))

        dp2 = getattr(self.original_block, "drop_path2", None)
        ls2 = getattr(self.original_block, "ls2", None)
        x = self._apply_residual(x, ffn_out, dp2, ls2)
        x = self.adapter_ffn(x)

        return x


def inject_adapters(model: nn.Module, embed_dim: int, bottleneck_dim: int,
                    target_blocks: list = None) -> int:
    """Inject Houlsby adapters into ViT blocks. Returns total adapter params."""
    total = 0
    # 1. Create a standard list to hold the blocks
    new_blocks_list = [] 
    
    for idx, block in enumerate(model.blocks):
        if target_blocks is not None and idx not in target_blocks:
            new_blocks_list.append(block)
        else:
            adapted = AdaptedBlock(block, embed_dim, bottleneck_dim)
            new_blocks_list.append(adapted)
            total += adapted.adapter_attn.trainable_params + adapted.adapter_ffn.trainable_params
    
    # 2. Wrap the list in nn.Sequential so it has a .forward() method
    model.blocks = nn.Sequential(*new_blocks_list) 
    
    return total


# ============================================================================
# PEFT: Visual Prompt Tuning
# ============================================================================

class VPTShallowEncoder(nn.Module):
    """Wraps timm ViT to prepend learnable prompts at input layer only."""

    def __init__(self, encoder: nn.Module, num_prompts: int, embed_dim: int):
        super().__init__()
        self.encoder = encoder
        self.num_prompts = num_prompts
        self.prompt_tokens = nn.Parameter(torch.zeros(1, num_prompts, embed_dim))
        nn.init.normal_(self.prompt_tokens, std=0.02)
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        x = self.encoder.patch_embed(x)
        if hasattr(self.encoder, "cls_token") and self.encoder.cls_token is not None:
            x = torch.cat([self.encoder.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        x = torch.cat([self.prompt_tokens.expand(B, -1, -1), x], dim=1)
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        x = x[:, self.num_prompts:, :]  # Remove prompts
        if hasattr(self.encoder, "cls_token") and self.encoder.cls_token is not None:
            x = x[:, 1:, :].mean(dim=1)
        else:
            x = x.mean(dim=1)
        return x

    @property
    def trainable_params(self):
        return self.prompt_tokens.numel()


class VPTDeepEncoder(nn.Module):
    """Wraps timm ViT to prepend learnable prompts at every layer."""

    def __init__(self, encoder: nn.Module, num_prompts: int,
                 embed_dim: int, num_layers: int):
        super().__init__()
        self.encoder = encoder
        self.num_prompts = num_prompts
        self.prompt_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, num_prompts, embed_dim))
            for _ in range(num_layers)
        ])
        for pt in self.prompt_tokens:
            nn.init.normal_(pt, std=0.02)
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        x = self.encoder.patch_embed(x)
        if hasattr(self.encoder, "cls_token") and self.encoder.cls_token is not None:
            x = torch.cat([self.encoder.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        for layer_idx, block in enumerate(self.encoder.blocks):
            x = torch.cat([self.prompt_tokens[layer_idx].expand(B, -1, -1), x], dim=1)
            x = block(x)
            x = x[:, self.num_prompts:, :]
        x = self.encoder.norm(x)
        if hasattr(self.encoder, "cls_token") and self.encoder.cls_token is not None:
            x = x[:, 1:, :].mean(dim=1)
        else:
            x = x.mean(dim=1)
        return x

    @property
    def trainable_params(self):
        return sum(pt.numel() for pt in self.prompt_tokens)


def inject_vpt_shallow(encoder, num_prompts, embed_dim):
    """Returns (wrapped_encoder, param_count)."""
    w = VPTShallowEncoder(encoder, num_prompts, embed_dim)
    return w, w.trainable_params

def inject_vpt_deep(encoder, num_prompts, embed_dim, num_layers):
    """Returns (wrapped_encoder, param_count)."""
    w = VPTDeepEncoder(encoder, num_prompts, embed_dim, num_layers)
    return w, w.trainable_params


# ============================================================================
# PEFT: BitFit
# ============================================================================

def inject_bitfit(model: nn.Module) -> int:
    """Unfreeze only bias terms. Returns count of bias params."""
    total = 0
    for name, param in model.named_parameters():
        if "bias" in name:
            param.requires_grad = True
            total += param.numel()
        else:
            param.requires_grad = False
    return total


# ============================================================================
# Model Construction
# ============================================================================

class PEFTClassifier(nn.Module):
    """Frozen encoder + PEFT + linear head."""
    def __init__(self, encoder, embed_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x))


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters. Name matches baseline_utils.py (Stage 3)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable
    return {
        "trainable": trainable, "total": total, "frozen": frozen,
        "pct_trainable": round(100.0 * trainable / total, 4) if total > 0 else 0,
    }

# Alias for any code using the shorter name
count_params = count_parameters


def print_parameter_summary(model: nn.Module, label: str) -> Dict[str, int]:
    """Print formatted parameter summary. Name matches baseline_utils.py."""
    c = count_parameters(model)
    print(f"\n{'~'*50}")
    print(f"  {label}")
    print(f"{'~'*50}")
    print(f"  Trainable:   {c['trainable']:>12,}")
    print(f"  Frozen:      {c['frozen']:>12,}")
    print(f"  Total:       {c['total']:>12,}")
    print(f"  %% Trainable: {c['pct_trainable']:>11.4f}%%")
    print(f"{'~'*50}")
    return c

# Alias
print_param_summary = print_parameter_summary


def build_peft_model(
    method: str, checkpoint_path: Path, model_name: str, embed_dim: int,
    num_classes: int, device: torch.device,
    rank: int = 8, bottleneck_dim: int = 64, num_prompts: int = 50,
    num_layers: int = 32, lora_alpha: float = None, lora_dropout: float = 0.0,
    target_blocks: list = None,
) -> Tuple[nn.Module, Dict[str, int]]:
    """One-line PEFT model construction. Returns (model, param_counts)."""
    print(f"\n Building PEFT model: method={method}")
    encoder = load_ijepa_encoder(checkpoint_path, model_name, embed_dim, device, freeze=True)

    if method == "lora":
        inject_lora(encoder, rank=rank, alpha=lora_alpha,
                     dropout=lora_dropout, target_blocks=target_blocks)
        model = PEFTClassifier(encoder, embed_dim, num_classes).to(device)
    elif method == "adapter":
        inject_adapters(encoder, embed_dim, bottleneck_dim, target_blocks=target_blocks)
        model = PEFTClassifier(encoder, embed_dim, num_classes).to(device)
    elif method == "vpt_shallow":
        encoder, _ = inject_vpt_shallow(encoder, num_prompts, embed_dim)
        model = PEFTClassifier(encoder, embed_dim, num_classes).to(device)
    elif method == "vpt_deep":
        encoder, _ = inject_vpt_deep(encoder, num_prompts, embed_dim, num_layers)
        model = PEFTClassifier(encoder, embed_dim, num_classes).to(device)
    elif method == "bitfit":
        model = PEFTClassifier(encoder, embed_dim, num_classes).to(device)
        inject_bitfit(model.encoder)
    elif method == "linear_probe":
        model = PEFTClassifier(encoder, embed_dim, num_classes).to(device)
    elif method == "full_ft":
        for p in encoder.parameters():
            p.requires_grad = True
        model = PEFTClassifier(encoder, embed_dim, num_classes).to(device)
    else:
        raise ValueError(f"Unknown PEFT method: {method}")

    for p in model.head.parameters():
        p.requires_grad = True

    params = print_parameter_summary(model, f"{method.upper()} Classifier")
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
            scale = self._step / max(1, self.warmup_steps)
        else:
            progress = (self._step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


class EarlyStopping:
    """Early stopping on a validation metric."""
    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        improved = (score > self.best_score + self.min_delta) if self.mode == "max" \
            else (score < self.best_score - self.min_delta)
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(model, loader, optimizer, criterion, device,
                    scaler=None, use_amp=True, gradient_clip=None, scheduler=None):
    """Train one epoch. Returns dict with loss, accuracy, time_s."""
    model.train()
    total_loss = correct = total = 0
    t0 = time.time()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        labels = labels.long() # CrossEntropy requires Long type for indices
        if labels.dim() > 1:
            labels = labels.squeeze()
        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            with autocast(_AMP_DEVICE):
                logits = model(images)
                if logits.dim() == 3:
                    logits = logits[:, 0, :]
                # print(f"Logits shape: {logits.shape}") # Should be [32, 38]
                # print(f"Labels shape: {labels.shape}") # Should be [32]
                # print(f"Weight shape: {criterion.weight.shape}") # Should be [38]
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if gradient_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            if logits.dim() == 3:
                logits = logits[:, 0, :]
            loss = criterion(logits, labels)
            loss.backward()
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return {"loss": total_loss / total, "accuracy": correct / total, "time_s": time.time() - t0}


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=38, use_amp=True):
    """Evaluate model. Returns dict with loss, accuracy, macro_f1, per_class_f1, confusion_matrix."""
    model.eval()
    total_loss = total = 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if use_amp:
            with autocast(_AMP_DEVICE):
                logits = model(images)
                if logits.dim() == 3:
                    logits = logits[:, 0, :]
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            if logits.dim() == 3:
                logits = logits[:, 0, :]
            loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total += labels.size(0)

    preds = np.array(all_preds)
    labels_arr = np.array(all_labels)
    return {
        "loss": total_loss / total,
        "accuracy": accuracy_score(labels_arr, preds),
        "macro_f1": f1_score(labels_arr, preds, average="macro", zero_division=0),
        "per_class_f1": f1_score(labels_arr, preds, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(labels_arr, preds, labels=list(range(num_classes))),
        "predictions": preds,
        "labels": labels_arr,
    }


# ============================================================================
# Unified Training Entry Point
# ============================================================================

def train_peft(
    method, checkpoint_path, pv_root, splits_dir, norm_mean, norm_std,
    model_name, embed_dim, num_classes,
    fraction=1.0, seed=42, lr=3e-4,
    batch_size=32, max_epochs=50, patience=10, weight_decay=0.01,
    warmup_fraction=0.10, use_amp=True, gradient_clip=1.0, num_workers=4,
    rank=8, bottleneck_dim=64, num_prompts=50, num_layers=32,
    lora_alpha=None, lora_dropout=0.0, target_blocks=None,
    class_weights_path=None, save_dir=None, run_name=None,
    wandb_project=None, wandb_entity=None, wandb_group=None,
):
    """Unified PEFT experiment: seed -> data -> model -> train -> eval -> save."""
    device = get_device()
    set_seed(seed)
    print(f"\n{'='*60}\n  PEFT: {method} | frac={fraction} | seed={seed} | lr={lr}\n{'='*60}")

    train_loader, val_loader, test_loader = build_dataloaders(
        pv_root, splits_dir, norm_mean, norm_std,
        fraction=fraction, seed=seed, batch_size=batch_size, num_workers=num_workers)

    model, param_counts = build_peft_model(
        method=method, checkpoint_path=checkpoint_path, model_name=model_name,
        embed_dim=embed_dim, num_classes=num_classes, device=device,
        rank=rank, bottleneck_dim=bottleneck_dim, num_prompts=num_prompts,
        num_layers=num_layers, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_blocks=target_blocks)

    if class_weights_path and Path(class_weights_path).exists():
        criterion = nn.CrossEntropyLoss(weight=load_class_weights(class_weights_path, device))
    else:
        criterion = nn.CrossEntropyLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    total_steps = max_epochs * len(train_loader)
    scheduler = WarmupCosineScheduler(optimizer, int(total_steps * warmup_fraction), total_steps)
    scaler = GradScaler() if use_amp and device.type == "cuda" else None

    wandb_run = None
    if wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=wandb_project, entity=wandb_entity, group=wandb_group, name=run_name,
            config={"method": method, "fraction": fraction, "seed": seed, "lr": lr,
                    "batch_size": batch_size, "max_epochs": max_epochs, "patience": patience,
                    "weight_decay": weight_decay, "rank": rank, "bottleneck_dim": bottleneck_dim,
                    "num_prompts": num_prompts, "target_blocks": str(target_blocks),
                    "checkpoint": str(checkpoint_path), **param_counts},
            reinit=True)

    early_stop = EarlyStopping(patience=patience, mode="max")
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [],
               "val_accuracy": [], "train_accuracy": [], "lr": [], "epoch_time": []}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, max_epochs + 1):
        tm = train_one_epoch(model, train_loader, optimizer, criterion, device,
                             scaler=scaler, use_amp=use_amp, gradient_clip=gradient_clip,
                             scheduler=scheduler)
        vm = evaluate(model, val_loader, criterion, device, num_classes=num_classes, use_amp=use_amp)

        cur_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(tm["loss"])
        history["train_accuracy"].append(tm["accuracy"])
        history["val_loss"].append(vm["loss"])
        history["val_macro_f1"].append(vm["macro_f1"])
        history["val_accuracy"].append(vm["accuracy"])
        history["lr"].append(cur_lr)
        history["epoch_time"].append(tm["time_s"])

        if wandb_run:
            import wandb
            wandb.log({"epoch": epoch, "train/loss": tm["loss"], "train/accuracy": tm["accuracy"],
                        "val/loss": vm["loss"], "val/macro_f1": vm["macro_f1"],
                        "val/accuracy": vm["accuracy"], "lr": cur_lr, "epoch_time_s": tm["time_s"]})

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{max_epochs} | TrLoss: {tm['loss']:.4f} | "
                  f"ValF1: {vm['macro_f1']:.4f} | Acc: {vm['accuracy']:.4f} | "
                  f"LR: {cur_lr:.2e} | {tm['time_s']:.1f}s")

        if early_stop(vm["macro_f1"], epoch):
            print(f"  Early stop epoch {epoch} (best F1={early_stop.best_score:.4f} @{early_stop.best_epoch})")
            break
        if early_stop.best_epoch == epoch:
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    test_m = evaluate(model, test_loader, criterion, device, num_classes=num_classes, use_amp=use_amp)
    print(f"\n  TEST: Macro-F1={test_m['macro_f1']:.4f}  Acc={test_m['accuracy']:.4f}  BestEpoch={early_stop.best_epoch}")

    peak_vram = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0
    avg_time = np.mean(history["epoch_time"][1:]) if len(history["epoch_time"]) > 1 else history["epoch_time"][0]

    results = {
        "method": method,
        "hyperparams": {"rank": rank, "bottleneck_dim": bottleneck_dim, "num_prompts": num_prompts,
                        "target_blocks": str(target_blocks), "lora_alpha": lora_alpha, "lora_dropout": lora_dropout},
        "training_config": {"lr": lr, "optimizer": "AdamW", "weight_decay": weight_decay,
                            "scheduler": "cosine_warmup", "warmup_fraction": warmup_fraction,
                            "max_epochs": max_epochs, "early_stop_patience": patience,
                            "batch_size": batch_size, "fraction": fraction, "seed": seed,
                            "use_amp": use_amp, "gradient_clip": gradient_clip},
        "encoder_checkpoint": str(checkpoint_path),
        "param_count": param_counts,
        "results": {"test_macro_f1": test_m["macro_f1"], "test_accuracy": test_m["accuracy"],
                     "val_macro_f1": early_stop.best_score, "best_epoch": early_stop.best_epoch,
                     "total_epochs_run": len(history["train_loss"]),
                     "per_class_f1": test_m["per_class_f1"]},
        "compute": {"peak_vram_mb": round(peak_vram, 1), "avg_epoch_time_s": round(avg_time, 2),
                     "gpu_model": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"},
        "history": {k: [round(v, 6) for v in vs] for k, vs in history.items()},
    }

    if save_dir:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        fname = run_name or f"{method}_frac{fraction:.2f}_seed{seed}"
        save_results(results, save_dir / f"{fname}.json")
        plot_confusion_matrix(test_m["confusion_matrix"], save_path=save_dir / f"{fname}_confusion.png",
                              title=f"{method} (frac={fraction}, seed={seed})")
        results["results"]["confusion_matrix_path"] = str(save_dir / f"{fname}_confusion.png")
        torch.save({"model_state_dict": best_state or model.state_dict(), "method": method,
                     "param_counts": param_counts, "test_macro_f1": test_m["macro_f1"]},
                    save_dir / f"{fname}_best.pt")

    if wandb_run:
        import wandb
        wandb.log({"test/macro_f1": test_m["macro_f1"], "test/accuracy": test_m["accuracy"],
                    "best_epoch": early_stop.best_epoch, "peak_vram_mb": peak_vram})
        wandb.finish()

    return results


# ============================================================================
# Feature Extraction & kNN
# ============================================================================

@torch.no_grad()
def extract_features(model_or_encoder, loader, device, use_amp=True):
    """Extract features. Returns (features, labels) as numpy arrays."""
    encoder = model_or_encoder.encoder if hasattr(model_or_encoder, "encoder") else model_or_encoder
    encoder.eval()
    feats, labs = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        if use_amp:
            with autocast(_AMP_DEVICE):
                f = encoder(images)
        else:
            f = encoder(images)
        feats.append(f.float().cpu().numpy())
        labs.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(labs)


def knn_evaluate(train_features, train_labels, test_features, test_labels,
                 k_values=None, metric="cosine"):
    """k-NN evaluation. Returns dict with per-k results and best_macro_f1."""
    if k_values is None:
        k_values = [5, 10, 20]
    from sklearn.preprocessing import normalize
    tr = normalize(train_features); te = normalize(test_features)
    results = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute")
        knn.fit(tr, train_labels)
        preds = knn.predict(te)
        mf1 = f1_score(test_labels, preds, average="macro", zero_division=0)
        acc = accuracy_score(test_labels, preds)
        results[f"k={k}"] = {"macro_f1": mf1, "accuracy": acc,
                              "per_class_f1": f1_score(test_labels, preds, average=None, zero_division=0).tolist()}
        print(f"  k={k:3d} | Macro-F1: {mf1:.4f} | Accuracy: {acc:.4f}")
    best_k = max(results, key=lambda x: results[x]["macro_f1"])
    results["best_k"] = best_k
    results["best_macro_f1"] = results[best_k]["macro_f1"]
    return results


# ============================================================================
# Profiling
# ============================================================================

def profile_vram(model, input_shape, device):
    """Profile peak VRAM (GB) during fwd+bwd. Matches baseline_utils.py."""
    if device.type != "cuda":
        return 0.0
    model.train()
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    dummy = torch.randn(*input_shape, device=device)
    target = torch.zeros(input_shape[0], dtype=torch.long, device=device)
    try:
        with autocast(_AMP_DEVICE):
            out = model(dummy)
            loss = F.cross_entropy(out, target)
        loss.backward()
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache(); return -1.0
        raise
    peak = torch.cuda.max_memory_allocated()
    model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
    return peak / 1e9


@torch.no_grad()
def profile_inference(model, device, input_shape=(1, 3, 224, 224),
                      n_warmup=50, n_runs=200, use_amp=True):
    """Profile inference latency (ms/image)."""
    model.eval()
    dummy = torch.randn(*input_shape, device=device)
    for _ in range(n_warmup):
        with autocast(_AMP_DEVICE) if use_amp else nullcontext():
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        with autocast(_AMP_DEVICE) if use_amp else nullcontext():
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return round((time.time() - t0) / n_runs * 1000, 2)


# contextlib fallback for profile_inference
from contextlib import nullcontext


# ============================================================================
# Plotting
# ============================================================================

def plot_confusion_matrix(cm, save_path=None, title="", class_names=None, figsize=(14, 12)):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=figsize)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(cm_norm, annot=False, cmap="YlOrRd",
                xticklabels=class_names or range(cm.shape[0]),
                yticklabels=class_names or range(cm.shape[0]), ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title or "Confusion Matrix")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history, save_path=None, title="Training Curves"):
    """Plot loss + F1 curves. Matches baseline_utils.py."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train", color="#2c3e50")
    axes[0].plot(epochs, history["val_loss"], label="Val", color="#e74c3c")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["val_macro_f1"], label="Val Macro-F1", color="#27ae60")
    if "train_accuracy" in history:
        axes[1].plot(epochs, history["train_accuracy"], label="Train Acc", color="#2c3e50", ls="--", alpha=0.7)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.suptitle(title); plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_label_efficiency(results_by_method, save_path=None, title="Label Efficiency Curves"):
    """Plot label efficiency curves for multiple methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colours = plt.cm.Set2(np.linspace(0, 1, len(results_by_method)))
    for (method, data), c in zip(results_by_method.items(), colours):
        fracs = sorted(data.keys())
        means = [data[f]["mean"] for f in fracs]
        stds = [data[f].get("std", 0) for f in fracs]
        ax.plot(fracs, means, "o-", label=method, color=c, lw=2)
        ax.fill_between(fracs, [m-s for m,s in zip(means,stds)], [m+s for m,s in zip(means,stds)], alpha=0.15, color=c)
    ax.set_xlabel("Label Fraction"); ax.set_ylabel("Macro-F1"); ax.set_title(title)
    ax.set_xscale("log"); ax.set_xticks([0.01,0.05,0.10,0.25,0.50,1.0])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0%}"))
    ax.legend(loc="lower right"); ax.grid(alpha=0.3); plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(all_results, save_path=None, title="Pareto: F1 vs Params"):
    """Plot Pareto frontier."""
    fig, ax = plt.subplots(figsize=(10, 6))
    mc = {"lora":"#e74c3c","adapter":"#3498db","vpt_shallow":"#2ecc71",
          "vpt_deep":"#9b59b6","bitfit":"#f39c12","linear_probe":"#95a5a6","full_ft":"#34495e"}
    for r in all_results:
        ax.scatter(r["trainable_params"], r["macro_f1"], c=mc.get(r["method"],"#7f8c8d"),
                   s=80, alpha=0.7, edgecolors="white", lw=0.5, label=r.get("label",""))
    h, l = ax.get_legend_handles_labels()
    ax.legend(dict(zip(l,h)).values(), dict(zip(l,h)).keys(), loc="lower right", fontsize=9)
    ax.set_xlabel("Trainable Parameters"); ax.set_ylabel("Macro-F1")
    ax.set_xscale("log"); ax.set_title(title); ax.grid(alpha=0.3); plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Result Aggregation & I/O
# ============================================================================

def aggregate_seed_results(results_list):
    """Aggregate train_peft results across seeds."""
    f1s = [r["results"]["test_macro_f1"] for r in results_list]
    accs = [r["results"]["test_accuracy"] for r in results_list]
    eps = [r["results"]["best_epoch"] for r in results_list]
    vrams = [r["compute"]["peak_vram_mb"] for r in results_list]
    times = [r["compute"]["avg_epoch_time_s"] for r in results_list]
    return {
        "macro_f1": {"mean": float(np.mean(f1s)), "std": float(np.std(f1s)), "per_seed": f1s},
        "accuracy": {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "per_seed": accs},
        "best_epoch": {"mean": float(np.mean(eps)), "std": float(np.std(eps)), "per_seed": [int(e) for e in eps]},
        "peak_vram_mb": {"mean": float(np.mean(vrams)), "std": float(np.std(vrams))},
        "avg_epoch_time_s": {"mean": float(np.mean(times)), "std": float(np.std(times))},
        "param_count": results_list[0]["param_count"],
        "method": results_list[0]["method"],
        "hyperparams": results_list[0]["hyperparams"],
        "training_config": {k: v for k, v in results_list[0]["training_config"].items() if k != "seed"},
    }


def save_results(results, path):
    """Save to JSON (handles numpy)."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    s = json.loads(json.dumps(results, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    path.write_text(json.dumps(s, indent=2))
    print(f"  Saved: {path}")


def load_results(path):
    """Load from JSON."""
    return json.loads(Path(path).read_text())