"""Chl-a maps from reflectance through the MDN or an empirical retrieval, keyed by pseudo-label name."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from chla_prediction.recipe import CHLA_RANGE
from chla_prediction.retrieval.empirical import ndci_chla, three_band_chla
from chla_prediction.retrieval.mdn import MDN, MDN_BANDS

__all__ = ["RETRIEVAL_BANDS", "ChlaRetrieval", "log10_chla"]

# Reflectance bands each retrieval reads.
RETRIEVAL_BANDS = {
    "mdn_chla": MDN_BANDS,
    "ndci_chla": ["B4", "B5"],
    "three_band_chla": ["B4", "B5", "B6"],
}
_EMPIRICAL = {"ndci_chla": ndci_chla, "three_band_chla": three_band_chla}


def log10_chla(chla: np.ndarray, chla_range: tuple[float, float] = CHLA_RANGE) -> np.ndarray:
    """log10 of Chl-a (mg/m^3) clipped to ``chla_range``."""
    return np.log10(np.clip(chla, *chla_range))


class ChlaRetrieval:
    """Chl-a (mg/m^3) of a reflectance stack by any retrieval; the MDN needs its weights."""

    def __init__(self, mdn_weights: Path | None = None, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.mdn = MDN(mdn_weights).to(self.device) if mdn_weights is not None else None

    def __call__(self, name: str, reflectance: np.ndarray, bands: list[str], valid: np.ndarray) -> np.ndarray:
        """``[H, W]`` Chl-a of ``reflectance`` ``[C, H, W]`` at ``valid`` pixels, NaN elsewhere."""
        if name == "mdn_chla":
            if self.mdn is None:
                raise ValueError("The MDN retrieval needs mdn_weights")
            stack = torch.from_numpy(reflectance[[bands.index(band) for band in MDN_BANDS]]).to(self.device)
            return self.mdn.chla_map(stack, torch.from_numpy(valid).to(self.device)).cpu().numpy()
        return np.where(valid, _EMPIRICAL[name](reflectance, bands), np.float32(np.nan))
