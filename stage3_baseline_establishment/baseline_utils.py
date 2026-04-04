"""
baseline_utils.py
=================
Shared utilities for all Stage 3 baseline experiments.

Contains:
  - seed_everything        : full reproducibility setup
  - PlantVillageDataset    : torchvision-compatible dataset (loads from Stage 2 splits)
  - get_transforms         : train / val / test transform factories
  - train_one_epoch        : single epoch with AMP mixed precision
  - evaluate               : macro-F1, per-class F1, accuracy, confusion matrix
  - EarlyStopping          : patience-based on val macro-F1
  - save_results           : save metrics dict to JSON
  - plot_confusion_matrix  : save confusion matrix figure
  - plot_tsne              : t-SNE visualisation of features
  - label_efficiency_sweep : run across all fractions x seeds
  - load_ijepa_encoder     : load Meta I-JEPA or Leaf-JEPA checkpoint

Author : Leaf-JEPA IRP
Stage  : 3 — Baseline Establishment
"""

import json
import time
import sys, os
import random
import copy
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler

# Version-compatible autocast
try:
    from torch.amp import autocast as _autocast
    def autocast_ctx(device="cuda", enabled=True):
        return _autocast(device_type=device, enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast as _autocast
    def autocast_ctx(device="cuda", enabled=True):
        return _autocast(enabled=enabled)
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class PlantVillageDataset(Dataset):
    """
    Loads images from a list of (path, label) pairs.
    Designed to work with Stage 2 split JSON files.

    Split JSON format:
        {"paths": ["/abs/path/img1.jpg", ...], "labels": [0, 1, ...],
         "class_names": ["Apple___healthy", ...]}
    """
    VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(self, paths, labels, transform=None):
        self.paths = [Path(p) for p in paths]
        self.labels = labels
        self.transform = transform
        assert len(self.paths) == len(self.labels), \
            f"Mismatch: {len(self.paths)} paths vs {len(self.labels)} labels"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_split(split_json: Path):
    """Load a split JSON and return (paths, labels, class_names)."""
    with open(split_json) as f:
        data = json.load(f)
    return data["paths"], data["labels"], data.get("class_names", None)


def load_fraction_split(splits_dir: Path, fraction: float, seed: int):
    """
    Load a label-efficiency subset from Stage 2.
    Expected file: splits_dir / fractions / f"train_frac{fraction:.2f}_seed{seed}.json"
    """
    fname = f"fraction_{fraction:.2f}_seed{seed}.json"
    fpath = splits_dir / "fractions" / fname
    if not fpath.exists():
        raise FileNotFoundError(
            f"Fraction split not found: {fpath}. "
            "Ensure Stage 2 label splits have been created."
        )
    return load_split(fpath)


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════════

def get_transforms(phase: str, norm_mean: list, norm_std: list,
                   image_resize: int = 256, image_crop: int = 224):
    """
    Return transforms for a given phase.

    Phases:
      'train'  — random resize crop + horizontal flip + colour jitter + normalise
      'val'    — deterministic: resize 256 -> centre crop 224 -> normalise
      'test'   — identical to 'val'
    """
    normalise = transforms.Normalize(mean=norm_mean, std=norm_std)

    if phase == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_crop, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            normalise,
        ])
    else:  # val or test — deterministic
        return transforms.Compose([
            transforms.Resize(image_resize),
            transforms.CenterCrop(image_crop),
            transforms.ToTensor(),
            normalise,
        ])


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimiser, scaler, device,
                    grad_accum_steps=1):
    """
    Train for one epoch with AMP mixed precision.
    Returns average loss.
    """
    model.train()
    running_loss = 0.0
    n_batches = 0
    optimiser.zero_grad()

    from tqdm import tqdm
    pbar = tqdm(loader, desc="  Training", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")

    for i, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast_ctx(enabled=scaler.is_enabled()):
            outputs = model(images)
            loss = criterion(outputs, labels) / grad_accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad()

        running_loss += loss.item() * grad_accum_steps
        n_batches += 1
        pbar.set_postfix(loss=f"{running_loss/n_batches:.4f}")

    pbar.close()
    return running_loss / max(n_batches, 1)


def evaluate(model, loader, device, num_classes=38):
    """
    Evaluate model on a dataloader.
    Returns dict with macro_f1, accuracy, per_class_f1, all_preds, all_labels.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        from tqdm import tqdm
        for images, labels in tqdm(loader, desc="  Evaluating", leave=False,
                                   bar_format="{l_bar}{bar:30}{r_bar}"):
            images = images.to(device, non_blocking=True)
            with autocast_ctx(enabled=True):
                outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
        "per_class_f1": [float(f) for f in per_class_f1],
        "confusion_matrix": cm.tolist(),
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


def get_oof_probabilities(model, loader, device):
    """Get softmax probabilities for OOF prediction (CleanLab)."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            with autocast_ctx(enabled=True):
                outputs = model(images)
            probs = torch.softmax(outputs.float(), dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# EARLY STOPPING
# ═══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """Early stopping based on val macro-F1 (higher is better)."""

    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -1.0
        self.counter = 0
        self.best_state = None
        self.stopped = False

    def step(self, score, model):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True

    def load_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ═══════════════════════════════════════════════════════════════════════════════
# I/O UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(results: dict, path: Path):
    """Save results dict to JSON (handles numpy types)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean = json.loads(json.dumps(results, default=_convert))
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"  Saved: {path}")


def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    """Save a confusion matrix heatmap as PNG."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_tsne(features, labels, class_names, save_path, title="t-SNE", perplexity=30):
    """Generate and save a t-SNE visualisation."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Running t-SNE on {features.shape[0]} samples...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                n_iter=1000, init="pca", learning_rate="auto")
    embedded = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(14, 12))
    n_classes = len(set(labels))
    cmap = plt.cm.get_cmap("tab20", n_classes) if n_classes <= 20 else plt.cm.get_cmap("nipy_spectral", n_classes)

    for c in sorted(set(labels)):
        mask = labels == c
        name = class_names[c] if class_names and c < len(class_names) else str(c)
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   c=[cmap(c)], label=name, s=8, alpha=0.6)

    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=5, ncol=3, loc="center left", bbox_to_anchor=(1, 0.5),
              markerscale=2)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# I-JEPA ENCODER LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_ijepa_encoder(checkpoint_path, model_name="vit_huge_patch14_224",
                       device=torch.device("cpu")):
    """
    Load Meta I-JEPA or Leaf-JEPA checkpoint into a timm ViT model.

    Args:
        checkpoint_path: path to .pth.tar or .pth file
        model_name: timm model string
        device: target device

    Returns:
        model with loaded weights, on device
    """
    import timm

    model = timm.create_model(model_name, pretrained=False,
                              num_classes=0, global_pool="avg")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Meta checkpoint stores target_encoder (EMA) — prefer this
    if isinstance(ckpt, dict):
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
    else:
        state_dict = ckpt

    # Clean up key prefixes
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"  Loaded checkpoint: {Path(checkpoint_path).name}")
    print(f"  Missing keys:  {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")

    return model.to(device)


# ═══════════════════════════════════════════════════════════════════════════════
# LABEL EFFICIENCY SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def label_efficiency_sweep(
        model_factory,
        splits_dir,
        val_paths, val_labels,
        test_paths, test_labels,
        class_names,
        fractions,
        seeds,
        norm_mean, norm_std,
        batch_size, lr, head_lr, weight_decay, epochs, patience,
        device,
        baseline_id,
        baselines_dir,
        figures_dir,
        wandb_project=None,
        wandb_entity=None,
        grad_accum=1,
        num_workers=4,
):
    """
    Run a model across all label fractions and seeds.

    Args:
        model_factory: callable() -> (model, param_groups) where param_groups
                       is a list of dicts for the optimiser.
        splits_dir: path to Stage 2 splits directory
        val_paths, val_labels: validation data
        test_paths, test_labels: test data
        class_names: list of class name strings
        fractions: list of floats
        seeds: list of ints
        ...training hyperparams...

    Returns:
        dict of results keyed by f"frac{frac}_seed{seed}"
    """
    import wandb

    train_tf = get_transforms("train", norm_mean, norm_std)
    eval_tf  = get_transforms("val", norm_mean, norm_std)

    val_ds   = PlantVillageDataset(val_paths, val_labels, transform=eval_tf)
    test_ds  = PlantVillageDataset(test_paths, test_labels, transform=eval_tf)
    val_loader  = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    all_results = {}

    for frac in fractions:
        for seed in seeds:
            run_key = f"frac{frac:.2f}_seed{seed}"
            run_name = f"{baseline_id}-frac{frac:.2f}-seed{seed}"
            print(f"\n{'='*60}")
            print(f"  {run_name}")
            print(f"{'='*60}")

            seed_everything(seed)

            # Load fraction subset
            try:
                train_paths, train_labels, _ = load_fraction_split(splits_dir, frac, seed)
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")
                continue

            train_ds = PlantVillageDataset(train_paths, train_labels, transform=train_tf)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=True,
                                      drop_last=len(train_ds) > batch_size)

            # Build model
            model, param_groups = model_factory()
            model = model.to(device)
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Trainable params: {n_trainable:,}")
            print(f"  Training samples: {len(train_ds):,}")

            criterion = nn.CrossEntropyLoss()
            optimiser = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
            scaler = GradScaler(enabled=torch.cuda.is_available())
            es = EarlyStopping(patience=patience)

            # WandB
            if wandb_project:
                os.environ["WANDB__SERVICE_WAIT"] = "10"
                os.environ["WANDB_DISABLED"] = "true"
                try:
                    wandb.init(project=wandb_project, entity=wandb_entity,
                               name=run_name, reinit=True,
                               config={"fraction": frac, "seed": seed,
                                       "train_samples": len(train_ds),
                                       "trainable_params": n_trainable})
                except Exception:
                    print("WandB init failed — training without logging")
                    wandb_project = False

            # Train
            for epoch in range(epochs):
                train_loss = train_one_epoch(model, train_loader, criterion,
                                             optimiser, scaler, device, grad_accum)
                val_result = evaluate(model, val_loader, device)
                scheduler.step()

                if wandb_project:
                    wandb.log({"train_loss": train_loss,
                               "val_macro_f1": val_result["macro_f1"],
                               "val_accuracy": val_result["accuracy"],
                               "epoch": epoch})

                es.step(val_result["macro_f1"], model)
                if es.stopped:
                    print(f"  Early stop at epoch {epoch+1}, best val F1: {es.best_score:.4f}")
                    break

            # Load best and evaluate on test
            es.load_best(model)
            test_result = evaluate(model, test_loader, device)

            print(f"  Test macro-F1: {test_result['macro_f1']:.4f}")
            print(f"  Test accuracy: {test_result['accuracy']:.4f}")

            # Save confusion matrix
            cm_path = figures_dir / f"{baseline_id}_confusion_matrix_{run_key}.png"
            plot_confusion_matrix(
                np.array(test_result["confusion_matrix"]),
                class_names, cm_path,
                title=f"{baseline_id} | frac={frac} seed={seed} | F1={test_result['macro_f1']:.3f}"
            )

            if wandb_project:
                wandb.log({"test_macro_f1": test_result["macro_f1"],
                           "test_accuracy": test_result["accuracy"]})
                wandb.finish()

            all_results[run_key] = {
                "fraction": frac,
                "seed": seed,
                "train_samples": len(train_ds),
                "macro_f1": test_result["macro_f1"],
                "accuracy": test_result["accuracy"],
                "per_class_f1": test_result["per_class_f1"],
                "best_val_f1": float(es.best_score),
            }

            # Free GPU memory
            del model, optimiser, scheduler, scaler
            torch.cuda.empty_cache()

    # Save all results
    save_results(all_results, baselines_dir / f"{baseline_id}_label_efficiency.json")
    return all_results