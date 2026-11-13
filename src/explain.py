"""Attribution methods for OpacityNet: Grad-CAM, HiResCAM and Integrated Gradients.

Every function takes a batch `x` of shape (N, 1, H, W) in [-1024, 1024] and returns numpy
maps of shape (N, H, W) at the input resolution. CAMs are non-negative; IG is signed.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ActivationHook:
    """Keeps the output tensor of a module during the forward pass."""

    def __init__(self, module: nn.Module):
        self.activation = None
        self._handle = module.register_forward_hook(self._forward)

    def _forward(self, module, inputs, output):
        self.activation = output

    def remove(self):
        self._handle.remove()


def _upsample(maps: torch.Tensor, size) -> np.ndarray:
    maps = F.interpolate(maps[:, None], size=size, mode="bilinear", align_corners=False)
    return maps[:, 0].detach().cpu().numpy()


def cam(model: nn.Module, x: torch.Tensor, layer: nn.Module | None = None,
        method: str = "gradcam") -> np.ndarray:
    """Class activation maps of the logit at `layer` (default: the last convolutional block).

    - gradcam:  channel weights = spatial mean of the gradient (Selvaraju et al., 2017)
    - hirescam: element-wise gradient x activation, summed over channels (Draelos & Carin, 2020)

    For a model that ends with global average pooling and a linear head, both coincide at the
    last block up to a constant factor; they differ at earlier layers.
    """
    layer = model.features if layer is None else layer
    hook = _ActivationHook(layer)
    with torch.enable_grad():
        logit = model(x)[:, 0]
        a = hook.activation
        g, = torch.autograd.grad(logit.sum(), a)
    hook.remove()
    if method == "gradcam":
        m = (g.mean(dim=(2, 3), keepdim=True) * a).sum(1)
    elif method == "hirescam":
        m = (g * a).sum(1)
    else:
        raise ValueError(f"unknown method {method!r}")
    return _upsample(torch.relu(m), x.shape[-2:])


def integrated_gradients(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor | None = None,
                         steps: int = 32) -> np.ndarray:
    """Integrated Gradients of the logit (Sundararajan et al., 2017), Riemann sum with `steps`.

    The default baseline is a black image (-1024 = air in the TorchXRayVision range).
    """
    baseline = torch.full_like(x, -1024.0) if baseline is None else baseline
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=x.device)
    total = torch.zeros_like(x)
    with torch.enable_grad():
        for alpha in alphas:
            xi = (baseline + alpha * (x - baseline)).requires_grad_(True)
            logit = model(xi)[:, 0].sum()
            total += torch.autograd.grad(logit, xi)[0]
    ig = (x - baseline) * total / steps
    return ig[:, 0].detach().cpu().numpy()
