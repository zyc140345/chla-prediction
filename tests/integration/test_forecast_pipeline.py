"""Training, inference, evaluation, pseudo-label precomputation and checkpoint diagnostics on synthetic scenes."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch

from chla_prediction.config import EvaluateForecastConfig, TrainForecasterConfig
from chla_prediction.evaluation.diagnostics import CheckpointEvaluator
from chla_prediction.experiments.evaluate_forecast import main as evaluate_main
from chla_prediction.experiments.precompute_climatology import main as climatology_main
from chla_prediction.experiments.precompute_pseudo_labels import main as pseudo_labels_main
from chla_prediction.experiments.run_forecast_inference import main as inference_main
from chla_prediction.experiments.train_forecaster import main as train_main
from chla_prediction.io import read_jsonl
from chla_prediction.training import loop

SEQUENCE = {"input_window": 3, "min_input_observations": 2, "train_end": "2021-02-20", "val_end": "2021-03-12"}
SMALL_MODEL = {"base_channels": 8, "latent_channels": 16, "hidden_channels": 16}
CPU_OPTIM = {"batch_size": 2, "num_workers": 0, "device": "cpu"}


def _train_config(run_name: str, output_dir: Path, data: dict, forecast_phase: dict, **fields) -> dict:
    """A one-epoch warm-up, then forecasting, on 32-pixel crops on the CPU; ``fields`` override the top level."""
    return {
        "run_name": run_name,
        "output_dir": str(output_dir),
        "data": data | {"crop_size": 32},
        "model": SMALL_MODEL,
        "phases": [
            {"name": "warmup", "objective": "reconstruction", "epochs": 1},
            {"name": "forecast", "objective": "forecast"} | forecast_phase,
        ],
        "optim": CPU_OPTIM,
    } | fields


def _infer_config(run_name: str, output_dir: Path, data: dict, model: dict, checkpoint: Path | None) -> dict:
    return {
        "run_name": run_name,
        "output_dir": str(output_dir),
        "data": data,
        "model": model,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "split": "test",
        "device": "cpu",
    }


@pytest.fixture(scope="module")
def archive(tmp_path_factory, write_scene_archive):
    """A synthetic ten-scene archive that the tests using it only read."""
    return write_scene_archive(tmp_path_factory.mktemp("archive"), n_scenes=10, seed=7)


@dataclass(frozen=True)
class Pipeline:
    train_config: Path
    run_dir: Path
    persistence_manifest: Path
    evaluate_config: Path
    evaluation_dir: Path


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory, archive, run_experiment) -> Pipeline:
    """Train the model on NDCI pseudo-labels, forecast the test split with it and with persistence, and evaluate both.

    The NDCI pseudo-label is computed from reflectance, so no MDN weights are needed.
    """
    root = tmp_path_factory.mktemp("pipeline")
    outputs = root / "outputs"
    data = {"archives": archive.archives, "sequence": SEQUENCE}
    trained = data | {"target_product": "ndci_chla"}
    train = _train_config("train", outputs, trained, {"epochs": 1, "checkpoint_epochs": [0, 1]}, seed=7)
    train_config = run_experiment(train_main, root / "train.yaml", train)
    run_dir = outputs / "train"

    manifests = {}
    for name, model_data, model, checkpoint in [
        ("ours", trained, SMALL_MODEL, run_dir / "checkpoint_forecast.pt"),
        ("persistence", data, {"name": "persistence_forecaster"}, None),
    ]:
        infer = _infer_config(f"infer_{name}", outputs, model_data, model, checkpoint)
        run_experiment(inference_main, root / f"infer_{name}.yaml", infer)
        manifests[name] = outputs / f"infer_{name}" / "prediction_manifest.jsonl"

    evaluate = {
        "run_name": "evaluation",
        "output_dir": str(outputs),
        "data": data,
        "prediction_manifests": {name: str(path) for name, path in manifests.items()},
        "water_mask_dir": str(archive.water_mask_dir),
        "min_valid_fraction": 0.01,
        "chla_axis": "ndci",
    }
    evaluate_config = run_experiment(evaluate_main, root / "evaluate.yaml", evaluate)
    return Pipeline(train_config, run_dir, manifests["persistence"], evaluate_config, outputs / "evaluation")


def test_training_keeps_the_final_epoch_checkpoint_and_logs_finite_losses(pipeline) -> None:
    run_dir = pipeline.run_dir
    manifest = json.loads((run_dir / "pretrain_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sample_counts"]["train_sequences"] > 0
    assert manifest["sample_counts"]["val_sequences"] > 0
    assert (run_dir / "checkpoint_warmup.pt").exists()
    initial = torch.load(run_dir / "checkpoint_forecast_epoch000.pt", weights_only=True)
    final = torch.load(run_dir / "checkpoint_forecast_epoch001.pt", weights_only=True)
    accepted = torch.load(run_dir / "checkpoint_forecast.pt", weights_only=True)
    assert any(not torch.equal(initial[key], final[key]) for key in initial)
    assert all(torch.equal(final[key], accepted[key]) for key in final)
    records = read_jsonl(run_dir / "training_log.jsonl")
    assert all(np.isfinite(record["train_loss"]) for record in records)
    assert any("val_loss" in record for record in records)


def test_persistence_forecast_rasters_mark_nodata_as_nan(pipeline) -> None:
    first_prediction = read_jsonl(pipeline.persistence_manifest)[0]
    with rasterio.open(first_prediction["prediction_path"]) as stream:
        assert np.isnan(stream.nodata)


def test_evaluation_scores_every_model_with_skill_against_persistence(pipeline) -> None:
    summary = pd.read_csv(pipeline.evaluation_dir / "summary_metrics.csv")
    assert set(summary.model) == {"ours", "persistence"}
    assert summary.rmse.notna().all()
    persistence_skill = summary[summary.model == "persistence"].skill_vs_reference.iloc[0]
    assert abs(persistence_skill) < 1e-9


def test_checkpoint_evaluator_reproduces_the_evaluation_scores(pipeline, tmp_path) -> None:
    diagnostics = CheckpointEvaluator(
        TrainForecasterConfig.from_yaml(pipeline.train_config),
        EvaluateForecastConfig.from_yaml(pipeline.evaluate_config),
    )
    scored = diagnostics.score(pipeline.run_dir / "checkpoint_forecast.pt", "test", tmp_path / "diagnostic_test.csv")
    original = pd.read_csv(pipeline.evaluation_dir / "per_sample_metrics.csv")
    original = original[original.model == "ours"].sort_values("sample_id")
    scored = scored.sort_values("sample_id")
    assert scored.sample_id.tolist() == original.sample_id.tolist()
    np.testing.assert_allclose(scored.rmse, original.rmse, rtol=1e-6)


class _Kill(Exception):
    """Stands in for a wall-clock kill between two training epochs."""


def test_interrupted_training_resumes_bit_identical(tmp_path, archive, monkeypatch, run_experiment, write_yaml) -> None:
    """A resumed run's checkpoints and logs equal an uninterrupted run's exactly.

    That covers weights, optimizer, plateau-scheduler learning rate and the shuffle RNG.
    """
    data = {"archives": archive.archives, "sequence": SEQUENCE, "target_product": "ndci_chla"}
    outputs = tmp_path / "outputs"

    def config(run_name: str) -> dict:
        return _train_config(
            run_name,
            outputs,
            data,
            {"epochs": 3, "patience": 2},
            seed=7,
            optim=CPU_OPTIM | {"lr_plateau_patience": 1},
        )

    run_experiment(train_main, tmp_path / "straight.yaml", config("straight"))

    resumed_config = write_yaml(tmp_path / "resumed.yaml", config("resumed"))
    real_run_epoch = loop._run_epoch
    calls = 0

    def interrupting(*args, **kwargs):
        nonlocal calls
        calls += 1
        # Warm-up train (1), forecasting epoch 0 train (2) and validation (3):
        # die in forecasting epoch 1, after epoch 0's resume state reached disk.
        if calls == 4:
            raise _Kill
        return real_run_epoch(*args, **kwargs)

    monkeypatch.setattr(loop, "_run_epoch", interrupting)
    with pytest.raises(_Kill):
        train_main(["--config", str(resumed_config)])
    monkeypatch.setattr(loop, "_run_epoch", real_run_epoch)

    resumed_dir = outputs / "resumed"
    assert (resumed_dir / "checkpoint_resume.pt").exists()
    train_main(["--config", str(resumed_config)])
    assert not (resumed_dir / "checkpoint_resume.pt").exists()

    straight_dir = outputs / "straight"
    for name in ("checkpoint_warmup.pt", "checkpoint_forecast.pt", "checkpoint_best.pt"):
        straight = torch.load(straight_dir / name, weights_only=True)
        resumed = torch.load(resumed_dir / name, weights_only=True)
        assert straight.keys() == resumed.keys()
        assert all(torch.equal(straight[key], resumed[key]) for key in straight)
    for log in ("training_log.jsonl", "training_steps.jsonl"):
        straight_log = (straight_dir / log).read_text(encoding="utf-8")
        assert straight_log.strip()
        assert straight_log == (resumed_dir / log).read_text(encoding="utf-8")


def test_mdn_pseudo_labels_train_and_score_on_both_chla_axes(
    tmp_path, mdn_weights, write_scene_archive, run_experiment
) -> None:
    """Precompute MDN pseudo-labels and their climatology, then forecast and score as the paper's methods do.

    Persistence forecasts reflectance and a trained model forecasts the MDN
    pseudo-label; both are scored against MDN and against NDCI Chl-a. (The
    synthetic archive is too short for a climatology of the test dates.)
    """
    # Precomputation writes into the archive, so this test has its own.
    archive = write_scene_archive(tmp_path / "archive", n_scenes=10, seed=7)
    outputs = tmp_path / "outputs"
    weights = str(mdn_weights)
    data = {"archives": archive.archives, "sequence": SEQUENCE}

    precompute = {"data": data, "mdn_weights": weights, "device": "cpu"}
    run_experiment(pseudo_labels_main, tmp_path / "precompute.yaml", precompute)
    rows = read_jsonl(archive.manifest)
    assert all("mdn_chla" in row["products"] for row in rows)
    with rasterio.open(archive.manifest.parent / rows[0]["products"]["mdn_chla"]) as src:
        product = src.read(1)
    assert np.isfinite(product).all()
    # log10 of the clip range; float32 log10 may round 1000 to 3.0000002 depending on the platform.
    assert product.min() >= -1.0 - 1e-6
    assert product.max() <= 3.0 + 1e-6
    # A second run is a no-op.
    run_experiment(pseudo_labels_main, tmp_path / "precompute.yaml", precompute)

    # Eight-day climatology of the MDN pseudo-labels.
    run_experiment(climatology_main, tmp_path / "climatology.yaml", {"data": data, "product": "mdn_chla"})
    with rasterio.open(archive.manifest.parent / "climatology_mdn_chla_8day_mean.tif") as src:
        assert src.count == 46

    # Reflectance in, MDN pseudo-label (log10 Chl-a) out, spectral stage
    # initialized from the MDN.
    trained_data = data | {"crop_size": 32, "target_product": "mdn_chla"}
    trained_model = SMALL_MODEL | {"encoder": "mdn_spectral", "mdn_weights": weights}
    train = _train_config("train_product_target", outputs, trained_data, {"epochs": 1}, model=trained_model)
    run_experiment(train_main, tmp_path / "train_product_target.yaml", train)
    variants = [
        ("persistence", data, {"name": "persistence_forecaster"}, None),
        ("product_target", trained_data, trained_model, outputs / "train_product_target" / "checkpoint_forecast.pt"),
    ]
    manifests = {}
    for variant, variant_data, model, checkpoint in variants:
        infer = _infer_config(f"infer_{variant}", outputs, variant_data, model, checkpoint)
        run_experiment(inference_main, tmp_path / f"infer_{variant}.yaml", infer)
        manifests[variant] = str(outputs / f"infer_{variant}" / "prediction_manifest.jsonl")

    evaluate = {
        "run_name": "evaluation",
        "output_dir": str(outputs),
        "data": data,
        "prediction_manifests": manifests,
        "skill_reference": "persistence",
        "water_mask_dir": str(archive.water_mask_dir),
        "min_valid_fraction": 0.01,
        "mdn_weights": weights,
    }
    run_experiment(evaluate_main, tmp_path / "evaluate.yaml", evaluate)
    summary = pd.read_csv(outputs / "evaluation" / "summary_metrics.csv").set_index("model")
    assert summary.loc[list(manifests), ["chla_rmse", "rmse_log"]].notna().all().all()
    assert np.isfinite(summary.loc["persistence", "ndci_mae"])
    assert np.isnan(summary.loc["product_target", "ndci_mae"])
    overall = pd.read_csv(outputs / "evaluation" / "overall_summary_metrics.csv")
    assert set(overall.model) == set(manifests)

    # The same forecasts scored against NDCI Chl-a. The reflectance forecast
    # passes through the empirical retrieval; the MDN log10 Chl-a forecast is
    # compared across retrievals, and its own error in MDN space still needs
    # the MDN.
    evaluate_ndci = evaluate | {"run_name": "evaluation_ndci", "chla_axis": "ndci"}
    run_experiment(evaluate_main, tmp_path / "evaluate_ndci.yaml", evaluate_ndci)
    summary = pd.read_csv(outputs / "evaluation_ndci" / "summary_metrics.csv").set_index("model")
    assert summary.loc[list(manifests), ["chla_rmse", "rmse_log"]].notna().all().all()
