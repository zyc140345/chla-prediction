"""Shared fixtures: synthetic rasters and scene archives, small configs and model inputs, experiment runs.

The helper fixtures hand out plain functions and are session-scoped, so
module-scoped fixtures can use them as well.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
import yaml
from rasterio.transform import from_origin

from chla_prediction.config import ALL_BANDS, DataConfig, ModelConfig
from chla_prediction.io import write_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
WATER_ID = "hushan"
SCENE_SIZE = 40
SCENE_CRS = "EPSG:32649"


def _write_raster(path: Path, values: np.ndarray, descriptions: Sequence[str] | None = None) -> Path:
    """Write a ``[bands, H, W]`` array as a GeoTIFF of its dtype on a 10 m UTM grid.

    ``descriptions`` name the bands; ``imagery.archive.read_bands`` locates bands by them.
    """
    count, height, width = values.shape
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": values.dtype,
        "crs": SCENE_CRS,
        "transform": from_origin(466000, 2209000, 10, 10),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values)
        if descriptions is not None:
            dst.descriptions = tuple(descriptions)
    return path


@dataclass(frozen=True)
class SyntheticArchive:
    """Files of a one-water synthetic scene archive."""

    manifest: Path
    water_mask_dir: Path

    @property
    def archives(self) -> list[dict]:
        """The ``data.archives`` config entry naming this archive."""
        return [{"water_id": WATER_ID, "manifest_path": str(self.manifest)}]


def _write_scene_archive(directory: Path, n_scenes: int, seed: int) -> SyntheticArchive:
    """Write ``n_scenes`` 40 x 40 scenes ten days apart from 2021-01-01, their STAC manifest and a water mask.

    Reflectance is random. The SCL flags the top four rows cloudy and the rest
    water; the water mask covers all but a two-pixel border.
    """
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    scl = np.full((1, SCENE_SIZE, SCENE_SIZE), 6, dtype=np.uint8)
    scl[:, :4] = 8
    rows = []
    for index in range(n_scenes):
        scene_date = date(2021, 1, 1) + timedelta(days=10 * index)
        stem = f"{WATER_ID}_s2_{scene_date:%Y%m%d}_synthetic"
        reflectance = rng.integers(200, 3000, (len(ALL_BANDS), SCENE_SIZE, SCENE_SIZE), dtype=np.uint16)
        rows.append(
            {
                "scene_date": f"{scene_date:%Y%m%d}",
                "local_path": str(_write_raster(directory / f"{stem}.tif", reflectance, ALL_BANDS)),
                "scl_path": str(_write_raster(directory / f"{stem}_SCL.tif", scl)),
                "roi_cloud_shadow_fraction": 0.1,
                "bands": ALL_BANDS,
                "crs": SCENE_CRS,
            }
        )
    manifest = directory / "stac_download_manifest.jsonl"
    write_jsonl(manifest, rows)

    water_mask_dir = directory / "water_mask"
    water_mask_dir.mkdir()
    water = np.zeros((2, SCENE_SIZE, SCENE_SIZE), dtype=np.uint8)
    water[:, 2:-2, 2:-2] = 100
    # The two bands a config can select as water_mask_band.
    _write_raster(water_mask_dir / f"{WATER_ID}_gsw.tif", water, ["recurrence", "s1_median_water"])
    return SyntheticArchive(manifest, water_mask_dir)


def _data_config(sequence: dict | None = None, **fields) -> DataConfig:
    """A one-water DataConfig whose archive is never read; ``sequence`` updates the sequence options."""
    return DataConfig(
        archives=[{"water_id": WATER_ID, "manifest_path": "manifest.jsonl"}],
        sequence={"train_end": "2023-12-31", "val_end": "2024-06-30"} | (sequence or {}),
        **fields,
    )


def _model_config(**fields) -> ModelConfig:
    """A ModelConfig with small widths for fast CPU tests."""
    return ModelConfig(**{"base_channels": 8, "latent_channels": 16, "hidden_channels": 16} | fields)


def _forecast_inputs(
    batch: int = 2,
    steps: int = 4,
    channels: int = len(ALL_BANDS),
    size: int = 32,
    input_targets: bool = False,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Random keyword inputs of ``Forecaster.forecast`` with every frame and pixel valid.

    ``input_targets`` adds one-channel log10 Chl-a input frames in [-0.5, 1.5].
    """
    generator = torch.Generator().manual_seed(seed)
    inputs = {
        "images": torch.rand(batch, steps, channels, size, size, generator=generator),
        "masks": torch.ones(batch, steps, size, size),
        "frame_valid": torch.ones(batch, steps),
        "interval_days": torch.rand(batch, steps, generator=generator) * 20,
        "lead_days": torch.rand(batch, generator=generator) * 20,
    }
    if input_targets:
        inputs["input_targets"] = torch.rand(batch, steps, 1, size, size, generator=generator) * 2.0 - 0.5
        inputs["input_target_masks"] = torch.ones(batch, steps, size, size)
    return inputs


def _write_yaml(path: Path, config: dict) -> Path:
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _run_experiment(main: Callable[[list[str]], None], path: Path, config: dict) -> Path:
    """Write ``config`` to ``path``, run the experiment entry point ``main`` on it and return the path."""
    main(["--config", str(_write_yaml(path, config))])
    return path


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def mdn_weights() -> Path:
    """The exported MDN weights; tests using them skip where they are absent."""
    path = REPO_ROOT / "data" / "models" / "mdn" / "mdn_msi_chl.npz"
    if not path.exists():
        pytest.skip(f"MDN weights not exported at {path} (see tools/data/export_mdn_weights.py)")
    return path


@pytest.fixture(scope="session")
def write_raster() -> Callable[..., Path]:
    return _write_raster


@pytest.fixture(scope="session")
def write_scene_archive() -> Callable[..., SyntheticArchive]:
    return _write_scene_archive


@pytest.fixture(scope="session")
def make_data_config() -> Callable[..., DataConfig]:
    return _data_config


@pytest.fixture(scope="session")
def make_model_config() -> Callable[..., ModelConfig]:
    return _model_config


@pytest.fixture(scope="session")
def make_forecast_inputs() -> Callable[..., dict[str, torch.Tensor]]:
    return _forecast_inputs


@pytest.fixture(scope="session")
def write_yaml() -> Callable[[Path, dict], Path]:
    return _write_yaml


@pytest.fixture(scope="session")
def run_experiment() -> Callable[..., Path]:
    return _run_experiment
