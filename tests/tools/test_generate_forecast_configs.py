import itertools
from pathlib import Path

import generate_forecast_configs as generator
import pytest
import yaml

from chla_prediction.config import EvaluateForecastConfig, TrainForecasterConfig
from chla_prediction.io import read_jsonl

RUNS = generator.OUTPUT_DIR
PRETRAINED = f"{RUNS}/pretrain_ours/checkpoint_best.pt"
REGIMES = ("scratch", "adapted")
# Baogu has no validation set to select a checkpoint on.
CROSS_WATERS = sorted(set(generator.EVALUATION_WATERS) - {"baogu"})
CROSS_PLANS = [f"diagnostics_cross_{axis}" for axis, _ in generator.CROSS_RETRIEVALS]


@pytest.fixture(scope="module")
def matrix(tmp_path_factory) -> Path:
    """Directory of every generated config, written once for the module."""
    directory = tmp_path_factory.mktemp("configs")
    generator.generate(directory)
    return directory


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _train_config(matrix: Path, task: dict) -> TrainForecasterConfig:
    return TrainForecasterConfig.from_yaml(matrix / Path(task["train_config"]).name)


def _matched_run(task: dict) -> tuple:
    return task["regime"], task["data_amount"], task["seed"], task["water_id"], task["learning_rate"]


def test_only_reproduced_baseline_uses_plateau_scheduler(matrix) -> None:
    pretrain = _yaml(matrix / "pretrain_ours.yaml")
    baseline = _yaml(matrix / "train_scratch_convlstm_hushan.yaml")
    assert "lr_plateau_patience" not in pretrain["optim"]
    assert baseline["optim"]["lr_plateau_patience"] > 0


def test_plain_convlstm_infers_from_the_final_epoch_checkpoint(matrix) -> None:
    for water in generator.EVALUATION_WATERS:
        config = _yaml(matrix / f"infer_scratch_convlstm_{water}.yaml")
        assert Path(config["checkpoint"]).name == "checkpoint_forecast.pt"


def test_climatology_is_precomputed_for_the_evaluation_waters_only(matrix) -> None:
    precompute = _yaml(matrix / "precompute_climatology.yaml")
    assert {archive["water_id"] for archive in precompute["data"]["archives"]} == set(generator.EVALUATION_WATERS)
    assert "archive_root" not in precompute["data"]


def test_learning_curve_configs_log_every_training_step_for_each_seed(matrix) -> None:
    manifests = {}
    for seed, amount, regime in itertools.product(
        generator.LEARNING_CURVE_SEEDS, generator.LEARNING_CURVE_AMOUNTS, REGIMES
    ):
        family = f"curve_ours_{regime}_{amount}_s{seed}"
        config = _yaml(matrix / f"train_{family}_hushan.yaml")
        manifests[family] = f"{RUNS}/merged_{family}.jsonl"
        assert config["run_name"] == f"train_{family}_hushan"
        assert config["seed"] == seed
        assert config["optim"]["step_log_interval"] == 1
        if regime == "adapted":
            assert config["init_checkpoint"] == PRETRAINED
        else:
            assert "init_checkpoint" not in config

    evaluation = _yaml(matrix / "evaluate_learning_curve.yaml")
    for family, manifest in manifests.items():
        assert evaluation["prediction_manifests"][family] == manifest
    assert evaluation["prediction_manifests"]["ours_zero"] == f"{RUNS}/infer_ours_zero/prediction_manifest.jsonl"


def test_curve_diagnostic_plan_holds_each_data_amount_run_once(matrix) -> None:
    curve = read_jsonl(matrix / "diagnostics_curve.jsonl")
    curve_runs = [(t["regime"], t["data_amount"], t["seed"], t["water_id"]) for t in curve]
    assert sorted(curve_runs) == sorted(
        itertools.product(
            REGIMES, generator.LEARNING_CURVE_AMOUNTS, generator.LEARNING_CURVE_SEEDS, generator.EVALUATION_WATERS
        )
    )


@pytest.mark.parametrize(("axis", "pseudo_label"), generator.CROSS_RETRIEVALS)
def test_cross_retrieval_runs_train_one_year_on_another_target_with_checkpoints(matrix, axis, pseudo_label) -> None:
    tasks = read_jsonl(matrix / f"diagnostics_cross_{axis}.jsonl")
    assert sorted(map(_matched_run, tasks)) == sorted(
        itertools.product(
            REGIMES,
            [generator.CROSS_AMOUNT],
            generator.LEARNING_CURVE_SEEDS,
            CROSS_WATERS,
            generator.CROSS_LEARNING_RATES.values(),
        )
    )
    for task in tasks:
        config = _train_config(matrix, task)
        assert config.data.target_product == pseudo_label
        assert str(config.data.sequence.train_start) == generator.LEARNING_CURVE_AMOUNTS[generator.CROSS_AMOUNT]
        assert config.phases[0].epochs == 40
        assert config.phases[0].patience is None
        assert config.phases[0].checkpoint_epochs == [0, 1, 3, 5, 10, 20, 30, 40]
        assert config.optim.lr_plateau_patience is None


@pytest.mark.parametrize("plan", CROSS_PLANS)
def test_scratch_and_adapted_runs_differ_only_in_initialization(matrix, plan) -> None:
    scratch_tasks = [task for task in read_jsonl(matrix / f"{plan}.jsonl") if task["regime"] == "scratch"]
    assert scratch_tasks
    for task in scratch_tasks:
        scratch = _train_config(matrix, task)
        adapted = TrainForecasterConfig.from_yaml(
            matrix / Path(task["train_config"]).name.replace("scratch", "adapted")
        )
        assert scratch.init_checkpoint is None
        assert str(adapted.init_checkpoint) == PRETRAINED
        excluded = {"run_name", "init_checkpoint"}
        assert scratch.model_dump(exclude=excluded) == adapted.model_dump(exclude=excluded)


@pytest.mark.parametrize("axis", [axis for axis, _ in generator.CROSS_RETRIEVALS])
def test_cross_retrieval_evaluation_scores_on_its_own_axis(matrix, axis) -> None:
    evaluation = EvaluateForecastConfig.from_yaml(matrix / f"evaluate_cross_{axis}.yaml")
    assert evaluation.chla_axis == axis
    assert evaluation.mdn_weights is None
