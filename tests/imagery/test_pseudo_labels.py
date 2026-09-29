import numpy as np
import pytest

from chla_prediction.imagery.pseudo_labels import pseudo_label_frame
from chla_prediction.retrieval.empirical import ndci_chla


def test_ndci_pseudo_label_frame_is_the_log10_retrieval() -> None:
    reflectance = np.array([[[0.02, 0.05]], [[0.03, 0.02]]], np.float32)  # B4, B5
    valid = np.array([[True, False]])
    frame, frame_valid = pseudo_label_frame(None, reflectance, valid, ["B4", "B5"], "ndci_chla")
    assert frame.shape == (1, 1, 2)
    assert frame.dtype == np.float32
    assert np.allclose(10.0 ** frame[0], ndci_chla(reflectance, ["B4", "B5"]))
    assert np.array_equal(frame_valid, valid)


def test_three_band_pseudo_label_frame_drops_undefined_retrievals() -> None:
    bands = ["B4", "B5", "B6"]
    reflectance = np.array([[[0.01, 0, 0.02]], [[0.02, 0.02, 0.01]], [[0.01, 0.01, 0.01]]], dtype=np.float32)
    frame, frame_valid = pseudo_label_frame(None, reflectance, np.ones((1, 3), bool), bands, "three_band_chla")
    assert frame_valid.tolist() == [[True, False, False]]
    assert np.isfinite(frame).all()
    assert frame[0, 0, 0] == pytest.approx(np.log10(139.3385))
