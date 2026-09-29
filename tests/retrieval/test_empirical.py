import numpy as np
import pytest

from chla_prediction.retrieval.empirical import three_band_chla


def test_three_band_published_calibration_scale_and_invalid_pixels() -> None:
    bands = ["B4", "B5", "B6"]
    reflectance = np.array([[[0.01, 0, 0.02]], [[0.02, 0.02, 0.01]], [[0.01, 0.01, 0.01]]], dtype=np.float32)
    concentration = three_band_chla(reflectance, bands)
    assert concentration[0, 0] == pytest.approx(139.3385)
    assert np.isnan(concentration[0, 1:]).all()
    # The band-ratio form is invariant to the reflectance scale (surface reflectance or Rrs).
    assert np.allclose(concentration, three_band_chla(reflectance / np.pi, bands), equal_nan=True)
