"""Linear probe for the acquisition view on the 1024-d embedding, and its alignment with the head."""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


@torch.no_grad()
def embeddings(model, loader, device) -> np.ndarray:
    """Embeddings (N, 1024) from `model.embed` (ReLU + global average pooling)."""
    model.to(device).eval()
    out = []
    for batch in tqdm(loader, desc="Embed", leave=False):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        out.append(model.embed(x.to(device)).float().cpu())
    return torch.cat(out).numpy()


def view_probe(train_emb, train_is_ap, eval_emb, eval_is_ap, C: float = 1.0):
    """Standardised logistic regression predicting AP vs PA. Returns (pipeline, eval AUC)."""
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    probe.fit(train_emb, train_is_ap)
    return probe, roc_auc_score(eval_is_ap, probe.decision_function(eval_emb))


def view_direction(probe) -> np.ndarray:
    """Probe direction expressed in raw embedding space (undoing the standardisation)."""
    scaler, logreg = probe[0], probe[-1]
    return logreg.coef_[0] / scaler.scale_


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
