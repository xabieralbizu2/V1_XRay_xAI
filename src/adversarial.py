"""Adversarial deconfounding of the opacity score from the acquisition view (experiment E4).

An adversary tries to predict the view (AP = 1) from the classifier's logit **and the true
label**. A gradient-reversal layer makes the classifier maximise the adversary's loss, so the
score becomes independent of the view *given the label*: a healthy AP film and a healthy PA
film should get the same score. Conditioning on the label (Zhang et al., 2018) keeps the real
prevalence difference between views, unlike the unconditional version (Janizek et al., 2020).
"""
import torch
import torch.nn as nn


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.weight * grad, None


class ViewAdversary(nn.Module):
    """Small MLP: (logit, label) -> logit of P(view = AP)."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, logit: torch.Tensor, label: torch.Tensor, weight: float) -> torch.Tensor:
        z = torch.cat([_GradReverse.apply(logit, weight), label], dim=1)
        return self.net(z)
