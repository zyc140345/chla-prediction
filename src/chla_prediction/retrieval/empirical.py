"""Empirical Chl-a retrievals, the alternative pseudo-labels of the cross-retrieval study.

NDCI Chl-a (Mishra and Mishra 2012) and the MSI three-band model
(Ogashawara et al. 2021).
"""

from __future__ import annotations

import numpy as np

from chla_prediction.recipe import CHLA_RANGE

__all__ = ["ndci", "ndci_chla", "three_band_chla"]


def ndci(reflectance: np.ndarray, bands: list[str], eps: float = 1.0e-4) -> np.ndarray:
    """Normalized difference chlorophyll index (B5 - B4) / (B5 + B4)."""
    red_edge = reflectance[bands.index("B5")]
    red = reflectance[bands.index("B4")]
    return (red_edge - red) / (red_edge + red + eps)


def ndci_chla(reflectance: np.ndarray, bands: list[str]) -> np.ndarray:
    """Chl-a (mg/m^3) ``14.039 + 86.115 NDCI + 194.325 NDCI^2``, clipped to ``CHLA_RANGE``.

    Mishra and Mishra (2012), doi:10.1016/j.rse.2011.10.016.
    """
    index = ndci(reflectance, bands)
    return np.clip(14.039 + 86.115 * index + 194.325 * index**2, *CHLA_RANGE)


def three_band_chla(reflectance: np.ndarray, bands: list[str]) -> np.ndarray:
    """Chl-a (mg/m^3) of the MSI three-band model, NaN where undefined or nonpositive.

    ``232.329 (1/B4 - 1/B5) B6 + 23.174`` with the published coefficients of
    Ogashawara et al. (2021), doi:10.3390/rs13081542, Tables 2 and 3.
    """
    red, edge, nir = reflectance[[bands.index(b) for b in ("B4", "B5", "B6")]]
    valid = np.isfinite(red) & np.isfinite(edge) & np.isfinite(nir) & (red > 0) & (edge > 0) & (nir > 0)
    result = np.full(red.shape, np.nan, dtype=np.float32)
    result[valid] = 232.329 * (1.0 / red[valid] - 1.0 / edge[valid]) * nir[valid] + 23.174
    result[~np.isfinite(result) | (result <= 0)] = np.nan
    return result
