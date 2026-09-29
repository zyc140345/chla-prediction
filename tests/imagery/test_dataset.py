import torch

from chla_prediction.config import DataConfig
from chla_prediction.imagery.dataset import SequenceDataset, load_model_frames
from chla_prediction.imagery.waters import load_waters


def test_resize_sampling_keeps_the_whole_scene_and_its_validity(tmp_path, write_scene_archive) -> None:
    archive = write_scene_archive(tmp_path, n_scenes=10, seed=7)
    data = DataConfig(
        archives=archive.archives,
        sequence={"input_window": 3, "min_input_observations": 2, "train_end": "2021-02-20", "val_end": "2021-03-12"},
        crop_size=32,
        spatial_sampling="resize",
        target_product="ndci_chla",
    )
    archives, splits = load_waters(data)
    samples = {water_id: water_splits["val"] for water_id, water_splits in splits.items()}
    dataset = SequenceDataset(archives, samples, data, random_crop=False, seed=42)
    item = dataset[0]
    assert item["target"].shape[-2:] == (32, 32)
    # The synthetic top four rows are cloudy: resizing preserves that border;
    # the center crop would remove all four rows and falsely look fully valid.
    assert not item["target_valid"][:3].any()
    assert item["target_valid"][4:].all()

    sample = dataset.entries[0]
    scene = dataset.archives[sample.water_id].scenes[sample.target_index]
    _, _, native, _ = load_model_frames(scene, data)
    expected = torch.nn.functional.interpolate(torch.from_numpy(native)[None], (32, 32), mode="nearest-exact")[0]
    torch.testing.assert_close(item["target"], expected)
