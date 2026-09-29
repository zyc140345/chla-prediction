from datetime import date
from pathlib import Path

import numpy as np
import pytest

from chla_prediction.imagery.archive import (
    Scene,
    SceneArchive,
    load_pseudo_label,
    load_reflectance,
    read_scl,
    scene_shape,
)

BANDS = ["B2", "B3", "B4"]
HEIGHT, WIDTH = 20, 24


def test_windowed_loads_match_cropped_full_loads(tmp_path, write_raster) -> None:
    rng = np.random.default_rng(3)
    reflectance = rng.integers(0, 3000, (len(BANDS), HEIGHT, WIDTH), dtype=np.uint16)
    scl = rng.integers(0, 12, (1, HEIGHT, WIDTH), dtype=np.uint8)
    pseudo_label = rng.normal(size=(1, HEIGHT, WIDTH)).astype(np.float32)
    scene = Scene(
        water_id="water",
        scene_date=date(2021, 1, 1),
        path=write_raster(tmp_path / "scene.tif", reflectance, BANDS),
        scl_path=write_raster(tmp_path / "scene_SCL.tif", scl),
        cloud_fraction=0.0,
        crs="EPSG:32649",
        pseudo_labels={"mdn_chla": write_raster(tmp_path / "scene_mdn.tif", pseudo_label)},
    )
    rows, cols = slice(3, 15), slice(5, 21)
    window = (rows, cols)

    assert scene_shape(scene) == (HEIGHT, WIDTH)

    full, full_valid = load_reflectance(scene, BANDS)
    part, part_valid = load_reflectance(scene, BANDS, window)
    np.testing.assert_array_equal(part, full[:, rows, cols])
    np.testing.assert_array_equal(part_valid, full_valid[rows, cols])

    np.testing.assert_array_equal(read_scl(scene.scl_path, window), read_scl(scene.scl_path)[rows, cols])
    np.testing.assert_array_equal(
        load_pseudo_label(scene, "mdn_chla", window), load_pseudo_label(scene, "mdn_chla")[:, rows, cols]
    )


def test_majority_crs_keeps_the_scenes_of_one_grid() -> None:
    def scene(day: int, crs: str) -> Scene:
        return Scene("water", date(2021, 1, day), Path("x.tif"), Path("x_scl.tif"), 0.0, crs)

    archive = SceneArchive("water", (scene(1, "EPSG:32650"), scene(2, "EPSG:32651"), scene(3, "EPSG:32650")))
    with pytest.warns(UserWarning, match="dropping 1 scenes"):
        kept = archive.majority_crs()
    assert [s.crs for s in kept.scenes] == ["EPSG:32650", "EPSG:32650"]
