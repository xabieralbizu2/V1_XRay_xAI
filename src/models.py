# src/models.py
"""Pretrained TorchXRayVision models."""
import torch
import torchxrayvision as xrv
from tqdm import tqdm


def load_xrv_densenet(weights: str = "densenet121-res224-chex") -> torch.nn.Module:
    """DenseNet-121 trained only on CheXpert (no NIH/RSNA images: no test leakage)."""
    model = xrv.models.DenseNet(weights=weights)
    model.op_threshs = None  # return raw logits, not recalibrated outputs (verify in your version)
    return model.eval()


@torch.no_grad()
def predict_logits(model: torch.nn.Module, loader, device: torch.device):
    """Logits for every pathology, shape (N, n_pathologies)."""
    model.to(device).eval()
    return torch.cat([model(x.to(device)).cpu() for x in tqdm(loader, desc="Predict")]).numpy()