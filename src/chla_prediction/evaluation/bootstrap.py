"""Percentile bootstrap of a mean over forecast samples.

Samples are resampled, not waters: five waters are too few to resample.
"""

from __future__ import annotations

import numpy as np

__all__ = ["mean_ci"]


def mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_resamples: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile interval of the mean, ``(nan, nan)`` below two samples."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), float("nan")
    draws = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[draws].mean(axis=1)
    tail = 100.0 * (1.0 - confidence) / 2.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return float(low), float(high)
