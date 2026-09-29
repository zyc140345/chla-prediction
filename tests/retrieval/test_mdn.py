import numpy as np
import pytest
import torch

from chla_prediction.retrieval.mdn import MDN, MDN_BANDS


@pytest.fixture(scope="module")
def mdn(mdn_weights) -> MDN:
    return MDN(mdn_weights)


def test_export_takes_one_wavelength_per_mdn_band(mdn_weights) -> None:
    assert len(np.load(mdn_weights)["wavelengths"]) == len(MDN_BANDS)


def test_port_matches_reference_implementation(mdn: MDN, repo_root) -> None:
    table = np.loadtxt(
        repo_root / "tests" / "data" / "mdn_msi_reference_subset.csv", delimiter=",", skiprows=1, dtype=np.float32
    )
    rrs, reference = torch.from_numpy(table[:, :7]), table[:, 7]
    assert np.allclose(mdn(rrs).numpy(), reference, rtol=1e-4)


def test_chla_map_masks_invalid_pixels(mdn: MDN) -> None:
    reflectance = torch.rand(7, 4, 5) * 0.05 + 0.005
    valid = torch.ones(4, 5)
    valid[0, 0] = 0
    reflectance[2, 1, 1] = 0.0
    chla = mdn.chla_map(reflectance, valid)
    assert chla.shape == (4, 5)
    assert torch.isnan(chla[0, 0])
    assert torch.isnan(chla[1, 1])
    assert torch.isfinite(chla[2:]).all()
    assert (chla[2:] > 0).all()
