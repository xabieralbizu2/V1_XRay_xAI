"""Pretrained TorchXRayVision models."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
from tqdm import tqdm


def load_xrv_densenet(weights: str = "densenet121-res224-chex") -> nn.Module:
    """DenseNet-121 trained only on CheXpert (no NIH/RSNA images: no test leakage)."""
    model = xrv.models.DenseNet(weights=weights)
    model.op_threshs = None  # return raw logits, not recalibrated outputs (verify in your version)
    return model.eval()


class OpacityNet(nn.Module):
    """Single-logit classifier on top of the pretrained DenseNet-121 features.

    The head is initialised with the weights of one pretrained output (default
    `Lung Opacity`), so before any training it reproduces E0 exactly.
    """

    def __init__(self, base: nn.Module, init_from: str = "Lung Opacity"):
        super().__init__()
        self.features = base.features
        in_features = base.classifier.in_features
        self.classifier = nn.Linear(in_features, 1)
        j = base.pathologies.index(init_from)
        with torch.no_grad():
            self.classifier.weight.copy_(base.classifier.weight[j : j + 1])
            self.classifier.bias.copy_(base.classifier.bias[j : j + 1])

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """1024-d embedding after global average pooling (used for linear probes)."""
        f = F.relu(self.features(x))
        return F.adaptive_avg_pool2d(f, 1).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x))


@torch.no_grad()
def predict_logits(model: nn.Module, loader, device: torch.device):
    """Logits for every output, shape (N, n_outputs). Accepts loaders with or without labels."""
    model.to(device).eval()
    out = []
    for batch in tqdm(loader, desc="Predict", leave=False):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        out.append(model(x.to(device)).float().cpu())
    return torch.cat(out).numpy()
