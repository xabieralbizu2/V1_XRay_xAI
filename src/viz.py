"""Small plotting helpers shared by the notebooks."""
import matplotlib.pyplot as plt
import numpy as np


def normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    m = m - m.min()
    return m / (m.max() + 1e-8)


def overlay(ax, img, mask=None, heat=None, title=None, alpha=0.45):
    """Grey image with an optional heat map and/or lung-mask contour."""
    ax.imshow(normalise(img), cmap="gray", vmin=0, vmax=1)
    if heat is not None:
        ax.imshow(normalise(heat), cmap="jet", alpha=alpha, vmin=0, vmax=1)
    if mask is not None:
        ax.contour(np.asarray(mask, dtype=float), levels=[0.5], colors="lime", linewidths=0.8)
    if title:
        ax.set_title(title, fontsize=8)
    ax.axis("off")


def show_grid(images, masks=None, heats=None, titles=None, ncols=8, size=2.0):
    """Grid of overlays; returns the figure."""
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        if i >= n:
            ax.axis("off")
            continue
        overlay(ax, images[i],
                None if masks is None else masks[i],
                None if heats is None else heats[i],
                None if titles is None else titles[i])
    return fig
