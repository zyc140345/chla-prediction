"""Evaluate prediction manifests: score every sample, then summarize per water, overall and per forecast-lead bin.

The overall score of a method is the unweighted mean over waters of its
per-water mean over samples.
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from chla_prediction.config import EvaluateForecastConfig
from chla_prediction.evaluation.bootstrap import mean_ci
from chla_prediction.evaluation.scoring import CHLA_METRICS, OUTPUT_METRICS, SampleScorer
from chla_prediction.imagery.masks import water_extent_masks
from chla_prediction.imagery.waters import load_waters
from chla_prediction.io import read_jsonl

BOOTSTRAP_ITERATIONS = 2000

__all__ = ["EvaluationOutput", "horizon_bins", "overall_means", "run_evaluation", "seed_summary"]


@dataclass(frozen=True)
class EvaluationOutput:
    run_dir: Path
    per_sample_csv: Path
    summary_csv: Path


def overall_means(per_water: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    """Unweighted mean over waters per ``keys``, with the number of waters."""
    groups = per_water.groupby(keys)
    return groups[metrics].mean().join(groups.size().rename("n_waters")).reset_index()


def seed_summary(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    """Mean and standard deviation over seeds per ``keys``, as ``<metric>_mean`` and ``<metric>_std``."""
    summary = frame.groupby(keys)[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    return summary.reset_index()


def _summaries(per_sample: pd.DataFrame, reference_model: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tables pooled over samples, per water and overall, ranked by RMSE_log.

    Skill and paired differences use the Chl-a metrics on the samples shared
    with the skill reference; the output-space ``rmse`` is in each model's
    own units and is not compared across models. Bootstrap intervals
    resample samples.
    """
    metric_columns = OUTPUT_METRICS + CHLA_METRICS
    reference = per_sample[per_sample.model == reference_model].set_index("sample_id")
    rng = np.random.default_rng(0)
    rows = []
    for model_name, group in per_sample.groupby("model"):
        shared = group[group.sample_id.isin(reference.index)]
        row = {"model": model_name, "n_samples": len(group)}
        row.update({name: group[name].mean() for name in metric_columns})
        for name, skill in [("rmse_log", "skill_vs_reference"), ("chla_rmse", "chla_skill_vs_reference")]:
            reference_mean = reference.loc[shared.sample_id, name].mean() if len(shared) else np.nan
            row[skill] = float(1.0 - shared[name].mean() / reference_mean) if len(shared) else np.nan
        row["rmse_log_ci_low"], row["rmse_log_ci_high"] = mean_ci(
            group["rmse_log"].to_numpy(), rng=rng, n_resamples=BOOTSTRAP_ITERATIONS
        )
        paired = shared["rmse_log"].to_numpy() - reference.loc[shared.sample_id, "rmse_log"].to_numpy()
        paired = paired[np.isfinite(paired)]
        row["rmse_log_diff_vs_reference"] = float(paired.mean()) if paired.size else np.nan
        row["rmse_log_diff_ci_low"], row["rmse_log_diff_ci_high"] = mean_ci(
            paired, rng=rng, n_resamples=BOOTSTRAP_ITERATIONS
        )
        rows.append(row)
    pooled = pd.DataFrame(rows).sort_values("rmse_log")
    per_water = per_sample.groupby(["model", "water_id"])[metric_columns].mean().reset_index()
    overall = overall_means(per_water, ["model"], metric_columns)
    return pooled, per_water, overall.sort_values("rmse_log")


def horizon_bins(horizon_days: np.ndarray, edges: list[int]) -> pd.Categorical:
    """Forecast-lead bin of every sample; edges ``[7, 15, 30]`` give ``<=7d``, ``8-15d``, ``16-30d`` and ``>30d``."""
    spans = [f"{low + 1}-{high}d" for low, high in zip(edges, edges[1:], strict=False)]
    names = [f"<={edges[0]}d", *spans, f">{edges[-1]}d"]
    index = np.searchsorted(edges, horizon_days, side="left")
    return pd.Categorical([names[i] for i in index], categories=names, ordered=True)


def _horizon_summaries(per_sample: pd.DataFrame, reference_model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pooled and overall tables per forecast-lead bin, computed like the all-lead tables."""
    pooled_frames, overall_frames = [], []
    for label, group in per_sample.groupby("horizon_bin", sort=True, observed=True):
        if reference_model not in set(group.model):
            continue
        pooled, _, overall = _summaries(group, reference_model)
        pooled_frames.append(pooled.assign(horizon_bin=label))
        overall_frames.append(overall.assign(horizon_bin=label))
    if not pooled_frames:  # no bin contains the skill reference
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(pooled_frames, ignore_index=True), pd.concat(overall_frames, ignore_index=True)


def run_evaluation(config_path: Path) -> EvaluationOutput:
    config = EvaluateForecastConfig.from_yaml(config_path)
    if not config.prediction_manifests:
        raise ValueError(f"{config_path} names no prediction manifests")
    run_dir = config.output_dir / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    archives, _ = load_waters(config.data)
    masks = water_extent_masks(archives, config)
    scorer = SampleScorer(config)

    frames: list[dict] = []
    skipped: Counter[str] = Counter()
    for model_name, manifest_path in config.prediction_manifests.items():
        # A configured variant without predictions is left out of the tables.
        if not Path(manifest_path).exists() or Path(manifest_path).stat().st_size == 0:
            warnings.warn(f"Skipping {model_name}: no predictions at {manifest_path}", stacklevel=2)
            continue
        for row in read_jsonl(manifest_path):
            # A merged manifest may cover waters this evaluation does not score.
            if row["water_id"] not in masks:
                skipped[row["water_id"]] += 1
                continue
            metrics = scorer.score(row, masks[row["water_id"]])
            if metrics is None or metrics["water_mask_fraction"] < config.min_valid_fraction:
                continue
            metrics["model"] = model_name
            frames.append(metrics)
    for water_id, count in sorted(skipped.items()):
        print(f"Skipped {count} prediction rows of {water_id}, which this config does not evaluate")
    per_sample = pd.DataFrame(frames)
    if per_sample.empty:
        raise ValueError("No evaluable samples; check manifests and mask thresholds")
    per_sample["horizon_bin"] = horizon_bins(per_sample.horizon_days.to_numpy(), config.horizon_bins)
    per_sample_csv = run_dir / "per_sample_metrics.csv"
    per_sample.to_csv(per_sample_csv, index=False)

    pooled, per_water, overall = _summaries(per_sample, config.skill_reference)
    summary_csv = run_dir / "summary_metrics.csv"
    pooled.to_csv(summary_csv, index=False)
    per_water.to_csv(run_dir / "per_water_metrics.csv", index=False)
    overall.to_csv(run_dir / "overall_summary_metrics.csv", index=False)
    pooled_horizon, overall_horizon = _horizon_summaries(per_sample, config.skill_reference)
    pooled_horizon.to_csv(run_dir / "summary_by_horizon.csv", index=False)
    overall_horizon.to_csv(run_dir / "overall_summary_by_horizon.csv", index=False)
    return EvaluationOutput(run_dir=run_dir, per_sample_csv=per_sample_csv, summary_csv=summary_csv)
