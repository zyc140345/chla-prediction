"""Single-channel log10 Chl-a pseudo-labels used as forecast targets and model inputs."""

from __future__ import annotations

import numpy as np

from chla_prediction.imagery.archive import Scene, load_pseudo_label
from chla_prediction.retrieval.chla import log10_chla
from chla_prediction.retrieval.empirical import ndci_chla, three_band_chla

__all__ = ["pseudo_label_frame"]


def pseudo_label_frame(
    scene: Scene,
    reflectance: np.ndarray,
    valid: np.ndarray,
    bands: list[str],
    name: str,
    window: tuple[slice, slice] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace a reflectance frame ``[C, H, W]`` by its ``[1, H, W]`` log10 Chl-a pseudo-label.

    ``mdn_chla`` is read from the precomputed raster of the scene (pass the
    ``window`` the reflectance was read with); ``ndci_chla`` and
    ``three_band_chla`` are computed from the reflectance. Pixels where the
    pseudo-label is undefined leave the validity mask.
    """
    if name == "mdn_chla":
        frame = load_pseudo_label(scene, name, window)
        finite = np.isfinite(frame[0])
        return np.where(finite, frame, 0.0).astype(np.float32), valid & finite
    if name == "ndci_chla":
        return log10_chla(ndci_chla(reflectance, bands))[None].astype(np.float32), valid
    if name == "three_band_chla":
        concentration = three_band_chla(reflectance, bands)
        label_valid = valid & np.isfinite(concentration)
        return np.where(label_valid, log10_chla(concentration), 0.0)[None].astype(np.float32), label_valid
    raise ValueError(f"Unknown pseudo-label {name!r}")
