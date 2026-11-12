"""Fine-tuning of the pretrained DenseNet-121 on RSNA (experiment E1, standard ERM)."""
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from src.dataset import CachedCXRDataset
from src.models import OpacityNet, load_xrv_densenet, predict_logits


@dataclass
class TrainConfig:
    experiment: str = "E1"
    seed: int = 0
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: float = 1.0
    patience: int = 3
    num_workers: int = 0          # 0 on macOS; 4 on Kaggle
    max_train_samples: int | None = None   # small value for a quick smoke test
    amp: bool = True              # mixed precision, only used on CUDA


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RandomIntensity(nn.Module):
    """Small random contrast/brightness change in the [-1024, 1024] range."""

    def __init__(self, contrast: float = 0.1, brightness: float = 50.0):
        super().__init__()
        self.contrast, self.brightness = contrast, brightness

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast
        b = (torch.rand(1).item() * 2 - 1) * self.brightness
        return (x * a + b).clamp(-1024.0, 1024.0)


def train_transform() -> nn.Module:
    # No horizontal flip: chest anatomy is not left-right symmetric.
    return v2.Compose([
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=-1024.0),
        RandomIntensity(),
    ])


def make_loaders(cfg: TrainConfig, cache_path: Path, splits: pd.DataFrame):
    train = splits[splits.split == "train"]
    if cfg.max_train_samples:
        train = train.sample(cfg.max_train_samples, random_state=cfg.seed)
    val = splits[splits.split == "val"]
    train_ds = CachedCXRDataset(cache_path, train.patientId, train.target, train_transform())
    val_ds = CachedCXRDataset(cache_path, val.patientId, val.target)
    g = torch.Generator().manual_seed(cfg.seed)
    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                  persistent_workers=cfg.num_workers > 0)
    return (DataLoader(train_ds, shuffle=True, drop_last=True, generator=g, **common),
            DataLoader(val_ds, shuffle=False, **common),
            val.target.to_numpy())


def make_optimizer(model: OpacityNet, cfg: TrainConfig, steps_per_epoch: int):
    opt = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": cfg.lr_backbone},
        {"params": model.classifier.parameters(), "lr": cfg.lr_head},
    ], weight_decay=cfg.weight_decay)
    total = cfg.epochs * steps_per_epoch
    warmup = max(1, int(cfg.warmup_epochs * steps_per_epoch))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def evaluate_auc(model: nn.Module, loader, y_true: np.ndarray, device: torch.device) -> float:
    logits = predict_logits(model, loader, device)[:, 0]
    return roc_auc_score(y_true, logits)


def train_model(cfg: TrainConfig, cache_path: Path, splits: pd.DataFrame,
                out_dir: Path, device: torch.device) -> pd.DataFrame:
    """Train one model, keep the checkpoint with the best validation AUC, return the history."""
    set_seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{cfg.experiment.lower()}_seed{cfg.seed}"
    ckpt_path = out_dir / f"{run_name}.pt"

    train_loader, val_loader, y_val = make_loaders(cfg, cache_path, splits)
    model = OpacityNet(load_xrv_densenet()).to(device)
    opt, sched = make_optimizer(model, cfg, len(train_loader))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Sanity check: before training, the model must reproduce E0 on the validation set.
    best_auc = evaluate_auc(model, val_loader, y_val, device)
    history = [{"epoch": 0, "train_loss": float("nan"), "val_auc": best_auc, "seconds": 0.0}]
    print(f"[{run_name}] epoch 0 (pretrained, no training): val AUC = {best_auc:.4f}")
    torch.save(model.state_dict(), ckpt_path)
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = F.binary_cross_entropy_with_logits(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())

        val_auc = evaluate_auc(model, val_loader, y_val, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_auc": val_auc, "seconds": time.time() - t0})
        print(f"[{run_name}] epoch {epoch}: loss = {np.mean(losses):.4f}, "
              f"val AUC = {val_auc:.4f} ({time.time() - t0:.0f}s)")

        if val_auc > best_auc:
            best_auc, epochs_without_improvement = val_auc, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                print(f"[{run_name}] early stopping at epoch {epoch}")
                break

    history = pd.DataFrame(history)
    history.to_csv(out_dir / f"{run_name}_history.csv", index=False)
    (out_dir / f"{run_name}_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(f"[{run_name}] best val AUC = {best_auc:.4f} -> {ckpt_path}")
    return history


def load_trained(ckpt_path: Path, device: torch.device) -> OpacityNet:
    """Rebuild an OpacityNet and load fine-tuned weights."""
    model = OpacityNet(load_xrv_densenet())
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.to(device).eval()
