"""Cross-retrieval study: the same comparison with the NDCI and three-band pseudo-labels instead of the MDN's.

Three stages over a diagnostics plan: audit the alternative pseudo-labels on
the adaptation sets, select checkpoints on validation only, and aggregate
the train/test gaps per retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from chla_prediction.config import EvaluateForecastConfig, TrainForecasterConfig
from chla_prediction.evaluation.diagnostics_summary import select_matched, summarize_diagnostics
from chla_prediction.evaluation.evaluate import seed_summary
from chla_prediction.imagery.archive import load_reflectance
from chla_prediction.imagery.masks import water_extent_masks
from chla_prediction.imagery.waters import load_waters
from chla_prediction.io import file_sha256, read_jsonl, write_jsonl
from chla_prediction.recipe import CHLA_RANGE
from chla_prediction.retrieval.chla import RETRIEVAL_BANDS, ChlaRetrieval, log10_chla

__all__ = ["RETRIEVALS", "audit", "finish", "select"]

RETRIEVALS = ("ndci", "three_band")


def audit(tasks: list[dict], output: Path) -> None:
    """Per-scene coverage and value range of each retrieval on the adaptation-set target scenes."""
    retrieve = ChlaRetrieval()
    rows, metadata, seen = [], [], set()
    for task in tasks:
        identity = (task["retrieval"], task["water_id"])
        if identity in seen:
            continue
        seen.add(identity)
        train = TrainForecasterConfig.from_yaml(task["train_config"])
        evaluation = EvaluateForecastConfig.from_yaml(task["evaluation_config"])
        evaluation = evaluation.model_copy(update={"data": train.data})
        archives, splits = load_waters(train.data)
        masks = water_extent_masks(archives, evaluation)
        for archive in archives:
            samples = splits[archive.water_id]["train"]
            if not samples or not splits[archive.water_id]["val"]:
                raise ValueError(f"Empty training/validation split: {identity}")
            scene_indices = sorted({s.target_index for s in samples})
            eligible = 0
            for index in scene_indices:
                scene = archive.scenes[index]
                name = f"{task['retrieval']}_chla"
                bands = RETRIEVAL_BANDS[name]
                reflectance, valid = load_reflectance(scene, bands)
                valid &= masks[archive.water_id]
                concentration = retrieve(name, reflectance, bands, valid)
                usable = np.isfinite(concentration)
                values = concentration[usable]
                fraction = float(usable.sum() / masks[archive.water_id].sum())
                eligible += fraction >= evaluation.min_valid_fraction
                row = {
                    "retrieval": task["retrieval"],
                    "water_id": archive.water_id,
                    "date": scene.scene_date.isoformat(),
                    "base_pixels": int(valid.sum()),
                    "usable_pixels": int(usable.sum()),
                    "water_mask_fraction": fraction,
                    "below_clip": int((values < CHLA_RANGE[0]).sum()),
                    "above_clip": int((values > CHLA_RANGE[1]).sum()),
                }
                if values.size:
                    row.update(
                        dict(zip(("p01", "p50", "p99"), np.quantile(values, [0.01, 0.5, 0.99]).tolist(), strict=True))
                    )
                    other = retrieve("ndci_chla", reflectance, bands, usable)[usable]
                    # Subsample to about 10k pixels to bound memory.
                    step = max(1, len(values) // 10000)
                    x, y = log10_chla(values[::step]), np.log10(other[::step])
                    row["log_correlation_with_ndci"] = (
                        float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else None
                    )
                rows.append(row)
            if not eligible:
                raise ValueError(f"No training target scenes pass the existing coverage criterion: {identity}")
            metadata.append(
                {
                    "retrieval": task["retrieval"],
                    "water_id": archive.water_id,
                    "train_sequences": len(samples),
                    "validation_sequences": len(splits[archive.water_id]["val"]),
                    "audited_train_scenes": len(scene_indices),
                    "eligible_train_scenes": int(eligible),
                    "archive_manifest_sha256": file_sha256(archive.manifest_path),
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "training_label_audit.csv", index=False)
    (output / "label_audit.json").write_text(
        json.dumps(
            {
                "split": "train",
                "coefficients_fitted": False,
                "three_band_formula": "232.329 * (1/B4 - 1/B5) * B6 + 23.174",
                "source": "https://doi.org/10.3390/rs13081542",
                "invalid_policy": "Nonpositive or undefined 3BDA concentrations excluded; positive products clipped to [0.1, 1000] before log10.",
                "coverage": metadata,
            },
            indent=2,
        )
    )
    print(json.dumps(metadata, indent=2), flush=True)


def select(tasks: list[dict], output: Path) -> None:
    """Select each retrieval's checkpoints on validation; write the selected tasks as one plan."""
    selected = []
    for retrieval in RETRIEVALS:
        directory = output / retrieval / "paper"
        select_matched([t for t in tasks if t["retrieval"] == retrieval], directory)
        selected.extend(t | {"retrieval": retrieval} for t in read_jsonl(directory / "diagnostics_selected.jsonl"))
    write_jsonl(output / "diagnostics_selected.jsonl", selected)


def finish(tasks: list[dict], output: Path) -> None:
    """Aggregate the train/test gaps of the selected checkpoints per retrieval."""
    frames = []
    for retrieval in RETRIEVALS:
        directory = output / retrieval / "paper"
        summarize_diagnostics([t for t in tasks if t["retrieval"] == retrieval], directory, "matched")
        frame = pd.read_csv(directory / "matched_gaps_per_water.csv")
        frame["retrieval"] = retrieval
        frames.append(frame)
    gaps = pd.concat(frames, ignore_index=True)
    paper = output / "paper"
    paper.mkdir(exist_ok=True)
    gaps.to_csv(paper / "generalization_gap_per_water_seed.csv", index=False)
    metrics = ["train", "val", "test", "test_minus_train", "val_minus_train"]
    overall = gaps.groupby(["retrieval", "regime", "seed"])[metrics].mean().reset_index()
    overall.to_csv(paper / "generalization_gap_overall_seeds.csv", index=False)
    summary = seed_summary(overall, ["retrieval", "regime"], metrics)
    summary.to_csv(paper / "generalization_gap_summary.csv", index=False)
    print(summary.to_string(), flush=True)
