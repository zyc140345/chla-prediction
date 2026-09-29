"""Scene archives of the configured waters and their chronological sample splits."""

from __future__ import annotations

import warnings

from chla_prediction.config import DataConfig, SequenceConfig
from chla_prediction.imagery.archive import SceneArchive
from chla_prediction.imagery.sequences import SequenceSample, build_samples, split_samples

__all__ = ["load_waters", "split_water"]

Splits = dict[str, list[SequenceSample]]


def split_water(archive: SceneArchive, sequence: SequenceConfig) -> Splits:
    """Samples of one water, split chronologically into ``train``, ``val`` and ``test``."""
    dates = [scene.scene_date for scene in archive.scenes]
    window, min_observations = sequence.input_window, sequence.min_input_observations
    samples = build_samples(archive.water_id, dates, window, min_observations)
    splits = split_samples(samples, sequence.train_end, sequence.val_end, sequence.train_start)
    if sequence.train_max_lead_days is not None:
        extended = build_samples(archive.water_id, dates, window, min_observations, sequence.train_max_lead_days)
        splits["train"] = split_samples(extended, sequence.train_end, sequence.val_end, sequence.train_start)["train"]
    return splits


def load_waters(data: DataConfig) -> tuple[list[SceneArchive], dict[str, Splits]]:
    """Load every configured archive and its sample splits, keyed by water.

    Only scenes under the cloud limit on the majority CRS are kept; a water
    left without scenes is skipped with a warning.
    """
    archives = []
    for entry in data.archive_entries():
        archive = SceneArchive.from_stac_manifest(entry.manifest_path, entry.water_id)
        archive = archive.filtered(data.max_scene_cloud_fraction)
        if not archive.scenes:
            warnings.warn(
                f"Skipping {entry.water_id}: no scene with cloud <= {data.max_scene_cloud_fraction}", stacklevel=2
            )
            continue
        archives.append(archive.majority_crs())
    if not archives:
        raise ValueError("No archives with usable scenes")
    return archives, {archive.water_id: split_water(archive, data.sequence) for archive in archives}
