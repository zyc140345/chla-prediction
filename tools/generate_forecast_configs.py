"""Generate every run config into ``configs/generated/`` and those of the main comparison into ``configs/paper/``.

Every config derives from ``chla_prediction.recipe`` and the variant tables
below, and every run writes into ``outputs/runs/``. A variant is the
proposed model (``ours``), a baseline or an ablation; each ablation changes
one thing. ``configs/generated/`` is not tracked; ``configs/paper/`` is, and
holds the configs of the paper's main comparison (``PAPER_CONFIGS``).
Config names:

    pretrain_<v>, adapt_<v>_<water>       multi-water pretraining and few-shot transfer of variant v
    infer_<v>_zero, infer_<v>_<water>     test forecasts, zero-shot and after transfer
    train_scratch_<s>_<water>, infer_...  models trained from scratch on the target water
    baseline_rf, baseline_xgboost         per-pixel regressors
    pretrain_pix2pix, infer_pix2pix_zero  the zero-shot pix2pix baseline
    infer_persistence, infer_climatology  reference forecasts
    train_curve_*, infer_curve_*          data-amount study, its checkpoints scored by diagnostics_curve.jsonl
    train_cross_*                         cross-retrieval study, planned in diagnostics_cross_*.jsonl
    evaluate*                             evaluations
    precompute_*                          pseudo-labels and the climatology

    uv run python tools/generate_forecast_configs.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from chla_prediction.io import write_jsonl
from chla_prediction.recipe import (
    CROP_SIZE,
    EVALUATION_WATERS,
    FORECAST_OBJECTIVE,
    MAX_SCENE_CLOUD_FRACTION,
    MDN_WEIGHTS,
    PRETRAINING_OPTIM,
    PRETRAINING_PHASES,
    PROPOSED_MODEL,
    SEQUENCE,
    SHORELINE_EROSION_PX,
    TRANSFER_OPTIM,
    TRANSFER_PHASES,
    WATER_MASK_BAND,
    WATER_MASK_THRESHOLD,
)

SEED = 42
OUTPUT_DIR = "outputs/runs"
CONFIG_DIR = "configs/generated"
PAPER_DIR = "configs/paper"
CORPUS_DIR = "data/corpus"
EVALUATION_DIR = "data/evaluation"
CLAY_CHECKPOINT = "models/clay/clay-v1.5.ckpt"

# The proposed model with a ConvGRU in place of the temporal attention.
MDN_ENCODER_MODEL = PROPOSED_MODEL | {"name": "conv_gru_forecaster"}
# The per-pixel regressors also train on every later scene within this many days.
MULTI_LEAD_DAYS = 45
# The stacked ConvLSTM and pix2pix train on the forecast loss alone.
FORECAST_ONLY_OBJECTIVE = {"reconstruction_weight": 0.0}
# Epoch caps; with patience the phase stops at its validation optimum.
PIX2PIX_PHASES = [{"name": "forecast", "objective": "forecast", "epochs": 12, "patience": 2}]
SCRATCH_PHASES = [
    {"name": "warmup", "objective": "reconstruction", "epochs": 5},
    {"name": "forecast", "objective": "forecast", "epochs": 40, "patience": 5},
]
SCRATCH_OPTIM = {"learning_rate": 2.0e-4, "batch_size": 4}
# Data-amount study: adaptation-set start per amount.
LEARNING_CURVE_AMOUNTS = {"all": None, "2y": "2022-01-01", "1y": "2023-01-01"}
LEARNING_CURVE_SEEDS = [42, 43, 44]
# Cross-retrieval study: (chla_axis, pseudo-label) on the one-year adaptation sets, each run
# at two learning rates with the checkpoint then selected on validation.
CROSS_RETRIEVALS = [("ndci", "ndci_chla"), ("three_band", "three_band_chla")]
CROSS_AMOUNT = "1y"
CROSS_LEARNING_RATES = {"lr5e5": 5e-5, "lr2e4": 2e-4}


def variant(model: dict, *, batch_size: int | None = None) -> dict:
    """A variant pretrained on the corpus and transferred to each evaluation water."""
    return {
        "model": model,
        "objective": FORECAST_OBJECTIVE,
        "pretrain_batch_size": batch_size or PRETRAINING_OPTIM["batch_size"],
        "data_extra": {},
    }


def with_data(spec: dict, **data: object) -> dict:
    return spec | {"data_extra": data}


VARIANTS = {
    "ours": variant(PROPOSED_MODEL),
    # Encoders; ``rand`` keeps the MDN-shaped encoder with random initialization.
    "rand": variant(PROPOSED_MODEL | {"mdn_weights": None}),
    "resnet18": variant(PROPOSED_MODEL | {"encoder": "resnet"}),
    "resnet50": variant(PROPOSED_MODEL | {"encoder": "resnet50"}, batch_size=4),
    "convnext": variant(PROPOSED_MODEL | {"encoder": "convnext"}),
    "convnext_l": variant(
        PROPOSED_MODEL | {"encoder": "convnext", "encoder_width": 256, "encoder_depth": 9}, batch_size=4
    ),
    # Temporal modules.
    "convgru": variant(MDN_ENCODER_MODEL),
    "convlstm": variant(PROPOSED_MODEL | {"name": "conv_lstm_forecaster"}),
    "simvp": variant(MDN_ENCODER_MODEL | {"name": "simvp_forecaster"}),
    "stt": variant(MDN_ENCODER_MODEL | {"name": "space_time_transformer_forecaster"}),
    # Without time conditioning.
    "notime": variant(PROPOSED_MODEL | {"time_conditioning": False}),
    # Clay v1.5 with rank-8 LoRA; full fine-tuning does not fit in 24 GB of GPU memory.
    "clay_lora": variant(
        PROPOSED_MODEL
        | {"encoder": "clay_frozen", "clay_checkpoint": CLAY_CHECKPOINT, "clay_lora_rank": 8, "mdn_weights": None},
        batch_size=4,
    ),
    # Pretraining-corpus size: seeded corpus subsets.
    **{f"corpus{size}": with_data(variant(PROPOSED_MODEL), max_archives=size) for size in (25, 50, 100, 200)},
}

# Trained from scratch on one target water.
SCRATCH_VARIANTS = {
    # Yao et al. (2023): Adam 1e-4, ReduceLROnPlateau, 150 epochs, batch 2, MSE.
    "convlstm": {
        "model": {"name": "stacked_conv_lstm_forecaster", "hidden_channels": 64, "temporal_depth": 4},
        "objective": FORECAST_ONLY_OBJECTIVE | {"prediction_loss": "mse"},
        "phases": [{"name": "forecast", "objective": "forecast", "epochs": 150}],
        "optim": {"learning_rate": 1.0e-4, "weight_decay": 0.0, "batch_size": 2, "lr_plateau_patience": 10},
    },
    # The proposed model without pretraining.
    "ours": {"model": PROPOSED_MODEL, "objective": FORECAST_OBJECTIVE, "phases": SCRATCH_PHASES},
}


def archive(water: str) -> dict:
    return {"water_id": water, "manifest_path": f"{EVALUATION_DIR}/{water}/stac_download_manifest.jsonl"}


def data_block(archives: dict, input_product: str | None = None, target_product: str | None = None, **extra) -> dict:
    block = archives | {
        "sequence": SEQUENCE,
        "max_scene_cloud_fraction": MAX_SCENE_CLOUD_FRACTION,
        "crop_size": CROP_SIZE,
    }
    if input_product:
        block["input_product"] = input_product
    if target_product:
        block["target_product"] = target_product
    return block | extra


CORPUS_ARCHIVES = {"archive_root": CORPUS_DIR}
TARGET_ARCHIVES = {"archives": [archive(water) for water in EVALUATION_WATERS]}


def variant_data(spec: dict, archives: dict) -> dict:
    """Reflectance input and the MDN pseudo-label as the target."""
    return data_block(archives, target_product="mdn_chla", **spec.get("data_extra", {}))


def water_data(spec: dict, water: str) -> dict:
    return variant_data(spec, {"archives": [archive(water)]})


class Matrix:
    """Configs by name, plus the JSONL task plans."""

    def __init__(self):
        self.output_dir = OUTPUT_DIR
        self.config_dir = CONFIG_DIR
        self.configs: dict[str, dict] = {}
        self.plans: dict[str, list[dict]] = {}

    def run(self, name: str) -> str:
        return f"{self.output_dir}/{name}"

    def train(
        self,
        name: str,
        data: dict,
        spec: dict,
        phases: list[dict],
        optim: dict,
        seed: int = SEED,
        init_checkpoint: str | None = None,
    ) -> None:
        config = {
            "run_name": name,
            "seed": seed,
            "output_dir": self.output_dir,
            "data": data,
            "model": spec["model"],
            "objective": spec["objective"],
            "phases": phases,
            "optim": optim,
        }
        if init_checkpoint:
            config["init_checkpoint"] = init_checkpoint
        self.configs[name] = config

    def infer(
        self, name: str, data: dict, model: dict, checkpoint: str | None, split: str = "test", seed: int | None = None
    ) -> str:
        """Add an inference config; return its prediction manifest."""
        config = {"run_name": name} | ({"seed": seed} if seed is not None else {})
        config |= {"output_dir": self.output_dir, "data": data, "model": model}
        if checkpoint:
            config["checkpoint"] = checkpoint
        self.configs[name] = config | {"split": split, "device": "cuda"}
        return self.run(f"{name}/prediction_manifest.jsonl")

    def evaluate(self, name: str, run_name: str, manifests: dict | None, **extra) -> None:
        config = {"run_name": run_name, "output_dir": self.output_dir, "data": data_block(TARGET_ARCHIVES)}
        if manifests is not None:
            config["prediction_manifests"] = manifests
        self.configs[name] = (
            config
            | {
                "water_mask_band": WATER_MASK_BAND,
                "water_mask_threshold": WATER_MASK_THRESHOLD,
                "shoreline_erosion_px": SHORELINE_EROSION_PX,
            }
            | extra
        )


def pretrained_variants(matrix: Matrix) -> dict[str, str]:
    """Pretraining, zero-shot and adapted inference of every variant; the manifests of the main evaluation."""
    manifests = {}
    for name, spec in VARIANTS.items():
        pretrained = matrix.run(f"pretrain_{name}/checkpoint_best.pt")
        optim = PRETRAINING_OPTIM | {"batch_size": spec["pretrain_batch_size"], "num_workers": 12, "device": "cuda"}
        matrix.train(f"pretrain_{name}", variant_data(spec, CORPUS_ARCHIVES), spec, PRETRAINING_PHASES, optim)
        manifests[f"{name}_zero"] = matrix.infer(
            f"infer_{name}_zero", variant_data(spec, TARGET_ARCHIVES), spec["model"], pretrained
        )
        manifests[f"{name}_adapted"] = matrix.run(f"merged_{name}_adapted.jsonl")
        for water in EVALUATION_WATERS:
            data = water_data(spec, water)
            optim = TRANSFER_OPTIM | {"num_workers": 4, "device": "cuda"}
            matrix.train(f"adapt_{name}_{water}", data, spec, TRANSFER_PHASES, optim, init_checkpoint=pretrained)
            adapted = matrix.run(f"adapt_{name}_{water}/checkpoint_forecast.pt")
            matrix.infer(f"infer_{name}_{water}", data, spec["model"], adapted)
    return manifests


def scratch_variants(matrix: Matrix) -> dict[str, str]:
    manifests = {}
    for name, spec in SCRATCH_VARIANTS.items():
        for water in EVALUATION_WATERS:
            data = water_data(spec, water)
            optim = SCRATCH_OPTIM | {"num_workers": 8, "device": "cuda"} | spec.get("optim", {})
            matrix.train(f"train_scratch_{name}_{water}", data, spec, spec["phases"], optim)
            # The weights the last phase hands on: the best validation epoch, or
            # the final epoch for the stacked ConvLSTM, which has no early stopping.
            trained = matrix.run(f"train_scratch_{name}_{water}/checkpoint_forecast.pt")
            matrix.infer(f"infer_scratch_{name}_{water}", data, spec["model"], trained)
        manifests[f"scratch_{name}"] = matrix.run(f"merged_scratch_{name}.jsonl")
    return manifests


def learned_baselines(matrix: Matrix) -> dict[str, str]:
    """The per-pixel regressors and pix2pix."""
    pixel_data = data_block(TARGET_ARCHIVES, "mdn_chla", sequence=SEQUENCE | {"train_max_lead_days": MULTI_LEAD_DAYS})
    matrix.configs["baseline_xgboost"] = {
        "run_name": "baseline_xgboost",
        "estimator": "xgboost",
        "device": "cuda",
        "n_jobs": 6,
        "output_dir": matrix.output_dir,
        "data": pixel_data,
        "split": "test",
        "seed": SEED,
    }
    matrix.configs["baseline_rf"] = {
        "run_name": "baseline_rf",
        "output_dir": matrix.output_dir,
        "data": pixel_data,
        "split": "test",
        "seed": SEED,
        "estimator": "random_forest",
        "max_iter": 200,
    }
    # pix2pix is pretrained on whole scenes resized to 256 pixels and applied zero-shot.
    pix2pix = {"model": {"name": "pix2pix_forecaster"}, "objective": FORECAST_ONLY_OBJECTIVE}
    optim = PRETRAINING_OPTIM | {"num_workers": 12, "device": "cuda"}
    resized = {"target_product": "mdn_chla", "spatial_sampling": "resize"}
    matrix.train("pretrain_pix2pix", data_block(CORPUS_ARCHIVES, **resized), pix2pix, PIX2PIX_PHASES, optim)
    checkpoint = matrix.run("pretrain_pix2pix/checkpoint_best.pt")
    return {
        "xgboost_pixel": matrix.run("baseline_xgboost/prediction_manifest.jsonl"),
        "rf_pixel": matrix.run("baseline_rf/prediction_manifest.jsonl"),
        "pix2pix_zero": matrix.infer(
            "infer_pix2pix_zero", data_block(TARGET_ARCHIVES, **resized), pix2pix["model"], checkpoint, seed=SEED
        ),
    }


def reference_forecasts(matrix: Matrix) -> dict[str, str]:
    return {
        "persistence": matrix.infer(
            "infer_persistence", data_block(TARGET_ARCHIVES), {"name": "persistence_forecaster"}, None
        ),
        "climatology": matrix.infer(
            "infer_climatology",
            data_block(TARGET_ARCHIVES, "mdn_chla", climatology_base=True),
            {"name": "climatology_forecaster"},
            None,
        ),
    }


def learning_curve(matrix: Matrix) -> None:
    """Data amounts x (from scratch, transferred) x seeds for the proposed model, every step logged."""
    spec = VARIANTS["ours"]
    manifests = {}
    for seed in LEARNING_CURVE_SEEDS:
        for amount, train_start in LEARNING_CURVE_AMOUNTS.items():
            for mode in ("scratch", "adapted"):
                family = f"curve_ours_{mode}_{amount}_s{seed}"
                for water in EVALUATION_WATERS:
                    data = water_data(spec, water)
                    if train_start:
                        data |= {"sequence": SEQUENCE | {"train_start": train_start}}
                    logged = {"num_workers": 4, "device": "cuda", "step_log_interval": 1}
                    if mode == "scratch":
                        matrix.train(
                            f"train_{family}_{water}", data, spec, SCRATCH_PHASES, SCRATCH_OPTIM | logged, seed
                        )
                    else:
                        pretrained = matrix.run("pretrain_ours/checkpoint_best.pt")
                        optim = TRANSFER_OPTIM | logged
                        matrix.train(f"train_{family}_{water}", data, spec, TRANSFER_PHASES, optim, seed, pretrained)
                    trained = matrix.run(f"train_{family}_{water}/checkpoint_forecast.pt")
                    matrix.infer(f"infer_{family}_{water}", data, spec["model"], trained)
                manifests[family] = matrix.run(f"merged_{family}.jsonl")
    manifests["ours_zero"] = matrix.run("infer_ours_zero/prediction_manifest.jsonl")
    manifests["persistence"] = matrix.run("infer_persistence/prediction_manifest.jsonl")
    matrix.evaluate(
        "evaluate_learning_curve",
        "evaluation_learning_curve",
        manifests,
        skill_reference="persistence",
        mdn_weights=MDN_WEIGHTS,
    )


def checkpoint_diagnostics(matrix: Matrix) -> None:
    """Train, validation and test scores of the data-amount runs' checkpoints."""
    destination = matrix.run("overfit_diagnostics")
    evaluation = f"{matrix.config_dir}/evaluate.yaml"
    curve = []
    for amount in ("1y", "all", "2y"):
        for water in sorted(EVALUATION_WATERS, key=lambda name: name != "hushan"):
            for seed in LEARNING_CURVE_SEEDS:
                for mode in ("scratch", "adapted"):
                    name = f"curve_ours_{mode}_{amount}_s{seed}_{water}"
                    curve.append(
                        {
                            "kind": "curve",
                            "regime": mode,
                            "data_amount": amount,
                            "seed": seed,
                            "water_id": water,
                            "source_manifest": matrix.run(f"train_{name}/pretrain_manifest.json"),
                            "evaluation_config": evaluation,
                            "destination": f"{destination}/curve/{name}",
                        }
                    )
    matrix.plans["diagnostics_curve"] = curve


def cross_retrieval(matrix: Matrix) -> None:
    """The proposed model from scratch and transferred with the NDCI or three-band pseudo-label as target.

    Each condition runs at every rate of ``CROSS_LEARNING_RATES`` and saves
    checkpoints along the way; the selection on validation picks one.
    """
    spec = VARIANTS["ours"]
    phases = [
        {"name": "forecast", "objective": "forecast", "epochs": 40, "checkpoint_epochs": [0, 1, 3, 5, 10, 20, 30, 40]}
    ]
    for axis, pseudo_label in CROSS_RETRIEVALS:
        root = matrix.run(f"cross_retrieval/{axis}")
        evaluation_name = f"evaluate_cross_{axis}"
        matrix.evaluate(
            evaluation_name, f"evaluation_cross_{axis}", None, skill_reference="persistence", chla_axis=axis
        )
        tasks = []
        # Baogu has no validation set to select a checkpoint on.
        for water in sorted(EVALUATION_WATERS, key=lambda name: name != "hushan"):
            if water == "baogu":
                continue
            for seed in LEARNING_CURVE_SEEDS:
                for mode in ("scratch", "adapted"):
                    for rate_name, rate in CROSS_LEARNING_RATES.items():
                        name = f"train_cross_{axis}_{mode}_{CROSS_AMOUNT}_s{seed}_{rate_name}_{water}"
                        data = water_data(spec, water)
                        data["sequence"] = SEQUENCE | {"train_start": LEARNING_CURVE_AMOUNTS[CROSS_AMOUNT]}
                        data["target_product"] = pseudo_label
                        optim = TRANSFER_OPTIM | {
                            "learning_rate": rate,
                            "num_workers": 4,
                            "device": "cuda",
                            "step_log_interval": 1,
                        }
                        pretrained = matrix.run("pretrain_ours/checkpoint_best.pt") if mode == "adapted" else None
                        matrix.train(name, data, spec, phases, optim, seed, pretrained)
                        matrix.configs[name]["output_dir"] = f"{root}/training"
                        tasks.append(
                            {
                                "kind": "matched",
                                "regime": mode,
                                "data_amount": CROSS_AMOUNT,
                                "seed": seed,
                                "water_id": water,
                                "learning_rate": rate,
                                "train_config": f"{matrix.config_dir}/{name}.yaml",
                                "evaluation_config": f"{matrix.config_dir}/{evaluation_name}.yaml",
                                "destination": f"{root}/matched/{name}",
                                "retrieval": axis,
                            }
                        )
        matrix.plans[f"diagnostics_cross_{axis}"] = tasks


def build_matrix() -> Matrix:
    matrix = Matrix()
    matrix.configs["precompute_pseudo_labels"] = {
        # Every pretraining-corpus water and every evaluation water.
        "data": {"archive_root": CORPUS_DIR, **TARGET_ARCHIVES, "sequence": SEQUENCE},
        "mdn_weights": MDN_WEIGHTS,
        "device": "cuda",
    }
    matrix.configs["precompute_climatology"] = {
        "data": {**TARGET_ARCHIVES, "sequence": SEQUENCE},
        "product": "mdn_chla",
    }
    manifests = pretrained_variants(matrix) | scratch_variants(matrix) | learned_baselines(matrix)
    manifests |= reference_forecasts(matrix)
    matrix.evaluate(
        "evaluate", "evaluation_transfer", manifests, skill_reference="persistence", mdn_weights=MDN_WEIGHTS
    )
    learning_curve(matrix)
    checkpoint_diagnostics(matrix)
    cross_retrieval(matrix)
    return matrix


# The main comparison: the proposed model and the six baselines, their data, and the evaluation.
PAPER_CONFIGS = [
    "precompute_pseudo_labels",
    "precompute_climatology",
    "pretrain_ours",
    *(f"{stage}_{water}" for stage in ("adapt_ours", "infer_ours") for water in EVALUATION_WATERS),
    "infer_persistence",
    "infer_climatology",
    "baseline_rf",
    "baseline_xgboost",
    *(f"{stage}_scratch_convlstm_{water}" for stage in ("train", "infer") for water in EVALUATION_WATERS),
    "pretrain_pix2pix",
    "infer_pix2pix_zero",
    "evaluate",
]


def write_yaml(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def generate(out_dir: Path, paper_dir: Path | None = None) -> None:
    """Write every config and plan into ``out_dir``, and the ``PAPER_CONFIGS`` into ``paper_dir``."""
    matrix = build_matrix()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, config in matrix.configs.items():
        write_yaml(out_dir / f"{name}.yaml", config)
    for name, tasks in matrix.plans.items():
        write_jsonl(out_dir / f"{name}.jsonl", tasks)
    print(f"wrote {len(matrix.configs)} configs and {len(matrix.plans)} plans to {out_dir}")
    if paper_dir is not None:
        paper_dir.mkdir(parents=True, exist_ok=True)
        for name in PAPER_CONFIGS:
            write_yaml(paper_dir / f"{name}.yaml", matrix.configs[name])
        print(f"wrote the {len(PAPER_CONFIGS)} configs of the main comparison to {paper_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(CONFIG_DIR), help=f"All configs (default {CONFIG_DIR})")
    parser.add_argument("--paper-out", type=Path, default=Path(PAPER_DIR), help=f"Paper configs (default {PAPER_DIR})")
    args = parser.parse_args()
    generate(args.out, args.paper_out)


if __name__ == "__main__":
    main()
