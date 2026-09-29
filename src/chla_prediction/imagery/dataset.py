"""Torch datasets over scene archives for training on MDN pseudo-labels."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from chla_prediction.config import DataConfig
from chla_prediction.imagery.archive import Scene, SceneArchive, load_reflectance, scene_shape
from chla_prediction.imagery.pseudo_labels import pseudo_label_frame
from chla_prediction.imagery.sequences import SequenceSample

__all__ = ["SequenceDataset", "SingleFrameDataset", "load_input_frames", "load_model_frames"]

Window = tuple[slice, slice]


def load_model_frames(
    scene: Scene, data: DataConfig, window: Window | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Input frame, input validity, target frame and target validity of a scene.

    The target frame is the scene in the model's output space: reflectance
    bands, or a log10 Chl-a pseudo-label with ``input_product`` or
    ``target_product``. For an input scene it is the persistence base.
    ``window`` limits the read to one crop.
    """
    reflectance, valid = load_reflectance(scene, data.input_bands, window)
    if data.input_product:
        frame, valid = pseudo_label_frame(scene, reflectance, valid, data.input_bands, data.input_product, window)
        return frame, valid, frame, valid
    if data.target_product:
        target, target_valid = pseudo_label_frame(
            scene, reflectance, valid, data.input_bands, data.target_product, window
        )
        return reflectance, valid, target, target_valid
    indices = [data.input_bands.index(band) for band in data.output_bands]
    return reflectance, valid, reflectance[indices], valid


def load_input_frames(
    archive: SceneArchive, sample: SequenceSample, data: DataConfig, window: Window | None = None
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """``load_model_frames`` of every input scene of a sample, as four lists."""
    frames = [load_model_frames(archive.scenes[index], data, window) for index in sample.input_indices]
    images, valids, bases, base_valids = (list(parts) for parts in zip(*frames, strict=True))
    return images, valids, bases, base_valids


class _WaterShapes:
    """Raster ``(height, width)`` per water, read once; a water's scenes share one grid after ``majority_crs``."""

    def __init__(self) -> None:
        self._shapes: dict[str, tuple[int, int]] = {}

    def get(self, scene: Scene) -> tuple[int, int]:
        if scene.water_id not in self._shapes:
            self._shapes[scene.water_id] = scene_shape(scene)
        return self._shapes[scene.water_id]


def _left_pad(array: np.ndarray, pad: int) -> np.ndarray:
    return np.concatenate([np.zeros((pad, *array.shape[1:]), array.dtype), array])


def _crop_slices(height: int, width: int, crop: int, rng: np.random.Generator | None) -> tuple[slice, slice]:
    if height < crop or width < crop:
        raise ValueError(f"Scene ({height}x{width}) is smaller than crop_size {crop}")
    if rng is None:
        top, left = (height - crop) // 2, (width - crop) // 2
    else:
        top = int(rng.integers(0, height - crop + 1))
        left = int(rng.integers(0, width - crop + 1))
    return slice(top, top + crop), slice(left, left + crop)


class SingleFrameDataset(Dataset):
    """Every scene as one randomly cropped frame for the reconstruction loss."""

    def __init__(self, archives: list[SceneArchive], data: DataConfig, seed: int):
        self.scenes: list[Scene] = [scene for archive in archives for scene in archive.scenes]
        self.data = data
        self.seed = seed
        self.shapes = _WaterShapes()

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng((self.seed, index))
        scene = self.scenes[index]
        height, width = self.shapes.get(scene)
        window = _crop_slices(height, width, self.data.crop_size, rng)
        image, valid, target, target_valid = load_model_frames(scene, self.data, window)
        return {
            "image": torch.from_numpy(image),
            "valid": torch.from_numpy(valid).float(),
            "target": torch.from_numpy(target),
            "target_valid": torch.from_numpy(target_valid).float(),
        }


class SequenceDataset(Dataset):
    """Past-window sequences predicting a later scene.

    Sequences shorter than the input window are left-padded with empty frames
    flagged invalid through ``frame_valid``. Training crops are random,
    validation crops centered; ``spatial_sampling: resize`` resizes whole
    scenes to ``crop_size`` instead.
    """

    def __init__(
        self,
        archives: list[SceneArchive],
        samples_per_water: dict[str, list[SequenceSample]],
        data: DataConfig,
        random_crop: bool,
        seed: int,
    ):
        self.archives = {archive.water_id: archive for archive in archives}
        self.entries = [sample for water_id in sorted(samples_per_water) for sample in samples_per_water[water_id]]
        self.data = data
        self.crop_size = data.crop_size
        self.random_crop = random_crop
        self.seed = seed
        self.shapes = _WaterShapes()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.entries[index]
        archive = self.archives[sample.water_id]
        rng = np.random.default_rng((self.seed, index)) if self.random_crop else None

        target_scene = archive.scenes[sample.target_index]
        height, width = self.shapes.get(target_scene)
        resize_scene = self.data.spatial_sampling == "resize"
        if resize_scene:
            window = (slice(0, height), slice(0, width))
        else:
            window = _crop_slices(height, width, self.crop_size, rng)
        frames, valids, bases, base_valids = load_input_frames(archive, sample, self.data, window)
        _, _, target, target_valid = load_model_frames(target_scene, self.data, window)
        if resize_scene:
            # nearest-exact matches the TF2 nearest resize of the pix2pix reference code.
            def resize(array: np.ndarray) -> np.ndarray:
                shape = array.shape
                tensor = torch.from_numpy(np.ascontiguousarray(array)).float().reshape(1, -1, *shape[-2:])
                result = torch.nn.functional.interpolate(tensor, (self.crop_size, self.crop_size), mode="nearest-exact")
                return result.numpy().reshape(*shape[:-2], self.crop_size, self.crop_size).astype(array.dtype)

            frames = [resize(x) for x in frames]
            valids = [resize(x) for x in valids]
            bases = [resize(x) for x in bases]
            base_valids = [resize(x) for x in base_valids]
            target, target_valid = resize(target), resize(target_valid)
        # Left-pad short sequences to the input window with invalid frames.
        pad = self.data.sequence.input_window - len(frames)
        frame_valid = _left_pad(np.ones(len(frames), np.float32), pad)
        images = _left_pad(np.stack(frames), pad)
        masks = _left_pad(np.stack(valids), pad)
        # Input frames in the output space (persistence base) with their own validity.
        input_targets = _left_pad(np.stack(bases), pad)
        input_target_masks = _left_pad(np.stack(base_valids), pad)
        intervals = _left_pad(np.asarray(sample.interval_days, dtype=np.float32), pad)

        return {
            "images": torch.from_numpy(images),
            "masks": torch.from_numpy(masks).float(),
            "frame_valid": torch.from_numpy(frame_valid),
            "interval_days": torch.from_numpy(intervals),
            "input_targets": torch.from_numpy(input_targets),
            "input_target_masks": torch.from_numpy(input_target_masks).float(),
            "target": torch.from_numpy(target),
            "target_valid": torch.from_numpy(target_valid).float(),
            "lead_days": torch.tensor(float(sample.lead_days)),
        }
