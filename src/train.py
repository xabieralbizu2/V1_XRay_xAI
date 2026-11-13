"""Fine-tuning of the pretrained DenseNet-121 on RSNA.

E1: standard training (default config).
E2: lung-only input (`input_mask="lung"` with the lung-hull masks), trained and evaluated that way.
E4: adversarial deconfounding of the score from the view (`adv_weight > 0`), see src/adversarial.py.
Optional: view-balanced reweighting (`reweight=True`), so that the view does not predict the label.
"""
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
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from src.adversarial import ViewAdversary
from src.dataset import CachedCXRDataset
from src.masks import MaskedCXRDataset
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
    input_mask: str | None = None  # "lung" for E2: lung-only input in training and evaluation
    mask_path: str | None = None   # lung-hull cache used when input_mask is set
    adv_weight: float = 0.0        # > 0 for E4: weight of the reversed adversary gradient
    reweight: bool = False         # view-balanced sample weights
    select_on: str = "auc"         # checkpoint selection: "auc" (E1) or "within_view_auc"


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


def make_dataset(cfg, cache_path: Path, patient_ids, labels=None, transform=None) -> Dataset:
    """Dataset matching the model's input: full image, or lung-only for E2.

    `cfg` can be a TrainConfig or the dict saved in `<run>_config.json`.
    """
    cfg = cfg if isinstance(cfg, dict) else asdict(cfg)
    if cfg.get("input_mask"):
        return MaskedCXRDataset(cache_path, cfg["mask_path"], patient_ids, mode=cfg["input_mask"],
                                fill="black", dilate_px=0, labels=labels, transform=transform)
    return CachedCXRDataset(cache_path, patient_ids, labels, transform)


class _WithView(Dataset):
    """Adds the view (1 = AP) and a sample weight to each (image, label) item."""

    def __init__(self, base: Dataset, is_ap: np.ndarray, weights: np.ndarray):
        self.base, self.is_ap, self.weights = base, is_ap.astype(np.float32), weights.astype(np.float32)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, y = self.base[i]
        return x, y, torch.tensor([self.is_ap[i]]), torch.tensor([self.weights[i]])


def view_balanced_weights(df: pd.DataFrame) -> np.ndarray:
    """w = P(y) / P(y | view): after weighting, the label prevalence is the same in AP and PA."""
    p_y = df.target.map(df.target.value_counts(normalize=True))
    p_y_view = df.groupby("view").target.transform(lambda t: t.map(t.value_counts(normalize=True)))
    w = (p_y / p_y_view).to_numpy()
    return w / w.mean()


def make_loaders(cfg: TrainConfig, cache_path: Path, splits: pd.DataFrame):
    train = splits[splits.split == "train"]
    if cfg.max_train_samples:
        train = train.sample(cfg.max_train_samples, random_state=cfg.seed)
    val = splits[splits.split == "val"]
    weights = view_balanced_weights(train) if cfg.reweight else np.ones(len(train))
    train_ds = _WithView(make_dataset(cfg, cache_path, train.patientId, train.target, train_transform()),
                         (train.view == "AP").to_numpy(), weights)
    val_ds = make_dataset(cfg, cache_path, val.patientId, val.target)
    g = torch.Generator().manual_seed(cfg.seed)
    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                  persistent_workers=cfg.num_workers > 0)
    return (DataLoader(train_ds, shuffle=True, drop_last=True, generator=g, **common),
            DataLoader(val_ds, shuffle=False, **common),
            val.reset_index(drop=True))


def make_optimizer(model: OpacityNet, cfg: TrainConfig, steps_per_epoch: int,
                   adversary: nn.Module | None = None):
    groups = [
        {"params": model.features.parameters(), "lr": cfg.lr_backbone},
        {"params": model.classifier.parameters(), "lr": cfg.lr_head},
    ]
    if adversary is not None:
        groups.append({"params": adversary.parameters(), "lr": cfg.lr_head})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    total = cfg.epochs * steps_per_epoch
    warmup = max(1, int(cfg.warmup_epochs * steps_per_epoch))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def validation_metrics(model: nn.Module, loader, val: pd.DataFrame, device: torch.device) -> dict:
    """Overall AUC, AUC within AP and within PA, their mean, and view leakage among Normal studies."""
    s = predict_logits(model, loader, device)[:, 0]
    y, ap = val.target.to_numpy(), (val.view == "AP").to_numpy()
    out = {"val_auc": roc_auc_score(y, s),
           "val_auc_ap": roc_auc_score(y[ap], s[ap]),
           "val_auc_pa": roc_auc_score(y[~ap], s[~ap])}
    out["val_within_view_auc"] = (out["val_auc_ap"] + out["val_auc_pa"]) / 2
    if "class" in val.columns:
        normal = (val["class"] == "Normal").to_numpy()
        out["val_leakage"] = roc_auc_score(ap[normal], s[normal])
    return out


def train_model(cfg: TrainConfig, cache_path: Path, splits: pd.DataFrame,
                out_dir: Path, device: torch.device) -> pd.DataFrame:
    """Train one model, keep the best checkpoint on validation (`cfg.select_on`), return the history."""
    set_seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{cfg.experiment.lower()}_seed{cfg.seed}"
    ckpt_path = out_dir / f"{run_name}.pt"
    key = "val_auc" if cfg.select_on == "auc" else "val_within_view_auc"

    train_loader, val_loader, val = make_loaders(cfg, cache_path, splits)
    model = OpacityNet(load_xrv_densenet()).to(device)
    adversary = ViewAdversary().to(device) if cfg.adv_weight > 0 else None
    opt, sched = make_optimizer(model, cfg, len(train_loader), adversary)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Epoch 0: the pretrained model (E0) on this run's validation input.
    metrics = validation_metrics(model, val_loader, val, device)
    best = metrics[key]
    history = [{"epoch": 0, "train_loss": float("nan"), "adv_loss": float("nan"), **metrics, "seconds": 0.0}]
    print(f"[{run_name}] epoch 0 (pretrained): " + ", ".join(f"{k} = {v:.4f}" for k, v in metrics.items()))
    torch.save(model.state_dict(), ckpt_path)
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        if adversary is not None:
            adversary.train()
        t0, losses, adv_losses = time.time(), [], []
        for x, y, v, w in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            x, y, v, w = x.to(device), y.to(device), v.to(device), w.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logit = model(x)
                loss = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).mean()
                total = loss
                if adversary is not None:
                    adv_logit = adversary(logit.float(), y, cfg.adv_weight)
                    adv_loss = F.binary_cross_entropy_with_logits(adv_logit, v)
                    total = loss + adv_loss   # the reversal layer turns this into a min-max game
                    adv_losses.append(adv_loss.item())
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())

        metrics = validation_metrics(model, val_loader, val, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "adv_loss": float(np.mean(adv_losses)) if adv_losses else float("nan"),
                        **metrics, "seconds": time.time() - t0})
        print(f"[{run_name}] epoch {epoch}: loss = {np.mean(losses):.4f}, "
              + ", ".join(f"{k} = {v:.4f}" for k, v in metrics.items())
              + f" ({time.time() - t0:.0f}s)")

        if metrics[key] > best:
            best, epochs_without_improvement = metrics[key], 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                print(f"[{run_name}] early stopping at epoch {epoch}")
                break

    history = pd.DataFrame(history)
    history.to_csv(out_dir / f"{run_name}_history.csv", index=False)
    (out_dir / f"{run_name}_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(f"[{run_name}] best {key} = {best:.4f} -> {ckpt_path}")
    return history


def load_trained(ckpt_path: Path, device: torch.device) -> OpacityNet:
    """Rebuild an OpacityNet and load fine-tuned weights."""
    model = OpacityNet(load_xrv_densenet())
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.to(device).eval()
