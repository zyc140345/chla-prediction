from datetime import date
from pathlib import Path

import numpy as np

from chla_prediction.imagery import climatology
from chla_prediction.imagery.archive import Scene, SceneArchive
from chla_prediction.imagery.climatology import SEASONAL_BINS, seasonal_bin_index


def _climatology(monkeypatch, records: list[tuple[date, float]], size: int) -> climatology.Climatology:
    """Climatology through 2023 of scenes whose pseudo-label is a constant field and whose SCL is all water."""
    scenes = tuple(
        Scene("water", day, Path(f"{value}.tif"), Path("scl.tif"), 0.0, "EPSG:32650", {"p": Path("x")})
        for day, value in records
    )
    monkeypatch.setattr(
        climatology,
        "load_pseudo_label",
        lambda scene, name: np.full((1, size, size), float(scene.path.stem), np.float32),
    )
    monkeypatch.setattr(climatology, "read_scl", lambda path: np.full((size, size), 6, np.uint8))
    return climatology.compute_climatology(SceneArchive("water", scenes), "p", date(2023, 12, 31))


def test_seasonal_bin_index_covers_the_year() -> None:
    assert seasonal_bin_index(date(2021, 1, 1)) == 0
    assert seasonal_bin_index(date(2021, 1, 8)) == 0
    assert seasonal_bin_index(date(2021, 1, 9)) == 1
    assert seasonal_bin_index(date(2021, 12, 26)) == 44
    assert seasonal_bin_index(date(2021, 12, 27)) == 45
    assert seasonal_bin_index(date(2021, 12, 31)) == SEASONAL_BINS - 1
    assert seasonal_bin_index(date(2020, 12, 31)) == SEASONAL_BINS - 1


def test_climatology_averages_each_position_over_years(monkeypatch) -> None:
    # The same eight-day bin (Jan 1-8) in three years, and one 2021 scene in bin 25.
    records = [(date(2019, 1, 3), 1.0), (date(2020, 1, 5), 2.0), (date(2021, 1, 7), 3.0), (date(2021, 7, 25), 9.0)]
    result = _climatology(monkeypatch, records, size=2)
    assert result.full[0, 0, 0] == 2.0
    assert result.full[25, 0, 0] == 9.0
    assert np.isnan(result.full[1, 0, 0])
    assert result.base(date(2024, 1, 8))[0, 0] == 2.0


def test_climatology_weights_years_equally_and_excludes_held_out_scenes(monkeypatch) -> None:
    records = [(date(2019, 1, 2), 1.0), (date(2019, 1, 3), 3.0), (date(2020, 1, 2), 8.0), (date(2021, 1, 2), 20.0)]
    records.append((date(2024, 1, 2), 1000.0))
    result = _climatology(monkeypatch, records, size=1)
    # Annual means 2, 8, 20 give 10; neither pooled mean 8 nor median 5.5.
    assert result.full[0, 0, 0] == 10.0
    assert np.isnan(result.full[1, 0, 0])
