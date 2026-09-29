from datetime import date

import pytest
from pydantic import ValidationError

from chla_prediction.config import PhaseConfig, TabularBaselineConfig, TrainForecasterConfig

TRAIN_CONFIG = """
run_name: unit
data:
  archives:
    - water_id: hushan
      manifest_path: manifest.jsonl
  sequence:
    train_end: 2023-12-31
    val_end: 2024-12-31
  target_product: mdn_chla
phases:
  - name: warmup
    objective: reconstruction
    epochs: 1
"""


def _load(tmp_path, text: str) -> TrainForecasterConfig:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return TrainForecasterConfig.from_yaml(path)


def test_train_config_loads_from_yaml(tmp_path) -> None:
    config = _load(tmp_path, TRAIN_CONFIG)
    assert config.data.archives[0].water_id == "hushan"
    assert config.data.sequence.train_end == date(2023, 12, 31)
    assert [phase.name for phase in config.phases] == ["warmup"]


def test_unknown_key_rejected(tmp_path) -> None:
    with pytest.raises(ValidationError):
        _load(tmp_path, TRAIN_CONFIG + "\nunknown_key: 1\n")


def test_output_bands_must_be_input_subset(tmp_path) -> None:
    text = TRAIN_CONFIG.replace("  sequence:", "  input_bands: [B2, B3]\n  output_bands: [B4, B5]\n  sequence:")
    with pytest.raises(ValidationError, match="subset"):
        _load(tmp_path, text)


def test_training_needs_a_pseudo_label_target(tmp_path) -> None:
    with pytest.raises(ValidationError, match="pseudo-label target"):
        _load(tmp_path, TRAIN_CONFIG.replace("  target_product: mdn_chla\n", ""))


def test_checkpoint_epoch_must_be_within_the_training_budget() -> None:
    with pytest.raises(ValueError, match="epoch budget"):
        PhaseConfig(name="forecast", objective="forecast", epochs=3, checkpoint_epochs=[4])


def test_tabular_baseline_needs_the_product_as_input(make_data_config) -> None:
    TabularBaselineConfig(run_name="rf", data=make_data_config(input_product="mdn_chla"))
    with pytest.raises(ValidationError, match="set data.input_product"):
        TabularBaselineConfig(run_name="rf", data=make_data_config(target_product="mdn_chla"))
