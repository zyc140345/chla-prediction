"""The paper's tables from an evaluation run.

Tables report the Chl-a metrics only and rank by the overall score, the
unweighted mean over waters. Each claim shows the paired difference pooled
over samples with its bootstrap interval, beside the difference per water.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from chla_prediction.evaluation.bootstrap import mean_ci

__all__ = ["CLAIMS", "LADDER", "make_paper_tables"]

BOOTSTRAP = 10000
# Rows of the main comparison as (variant, group, label), in print order;
# variants absent from the evaluation are dropped. The figures reuse the labels.
LADDER = [
    ("persistence", "Reference", "Persistence"),
    ("climatology", "Reference", "Climatology"),
    ("xgboost_pixel", "Per-pixel ML", "XGBoost"),
    ("rf_pixel", "Per-pixel ML", "Random forest"),
    ("scratch_convlstm", "Target-only DL", "ConvLSTM"),
    ("pix2pix_zero", "Cross-water DL", "pix2pix"),
    ("ours_adapted", "Ours", "Ours"),
]
# Component ablation of pretraining and few-shot transfer with one architecture.
PARADIGM = [
    ("scratch_ours", "w/o pretraining", "Proposed model trained from scratch on the target water"),
    ("ours_zero", "w/o target adaptation", "Pretrained model applied zero-shot"),
    ("ours_adapted", "Full method", "Pretraining + few-shot transfer"),
]
# Design choices: the proposed model's encoder and temporal module against alternatives.
ABLATIONS = [
    ("Encoder", ["ours", "rand", "resnet18", "resnet50", "convnext", "convnext_l", "clay_lora"]),
    ("Temporal module", ["ours", "convgru", "convlstm", "simvp", "stt"]),
]
CORPUS_SIZES = [25, 50, 100, 200]
# (variant, against, claim); negative differences support the claim.
CLAIMS = [
    ("ours_adapted", "persistence", "Beats Persistence"),
    ("ours_adapted", "climatology", "Beats Climatology"),
    ("ours_adapted", "xgboost_pixel", "Beats XGBoost"),
    ("ours_adapted", "rf_pixel", "Beats Random forest"),
    ("ours_adapted", "scratch_convlstm", "Beats ConvLSTM"),
    ("ours_adapted", "scratch_ours", "Pretraining helps (w/o pretraining)"),
    ("ours_adapted", "ours_zero", "Target adaptation helps (w/o target adaptation)"),
    ("ours_adapted", "notime_adapted", "Time conditioning helps after transfer (w/o time conditioning)"),
    ("ours_adapted", "pix2pix_zero", "Beats pix2pix"),
    ("ours_zero", "rand_zero", "MDN initialization helps (zero-shot)"),
    ("ours_adapted", "rand_adapted", "MDN initialization helps (after transfer)"),
]


def _paired(wide: pd.DataFrame, a: str, b: str, seed: int = 0) -> tuple[float, float, float, int]:
    """Mean paired difference ``a - b`` over shared samples, its bootstrap interval, and the sample count."""
    if a == b:
        return 0.0, 0.0, 0.0, int(wide[a].notna().sum())
    both = wide[[a, b]].dropna()
    difference = (both[a] - both[b]).to_numpy()
    low, high = mean_ci(difference, rng=np.random.default_rng(seed), n_resamples=BOOTSTRAP)
    mean = float(difference.mean()) if difference.size >= 2 else np.nan
    return mean, low, high, int(difference.size)


def _per_water_diffs(wide: pd.DataFrame, water_of: pd.Series, waters: list[str], a: str, b: str) -> dict:
    """Mean paired difference ``a - b`` per water, and in how many waters it is negative."""
    row, negative = {}, 0
    for water in waters:
        both = wide.loc[water_of == water, [a, b]].dropna()
        value = float((both[a] - both[b]).mean()) if len(both) else np.nan
        row[f"diff_{water}"] = value
        negative += value < 0
    row["waters_agreeing"] = f"{negative}/{len(waters)}"
    return row


def _water_means(per_sample: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Metric per model and water, plus the overall score (mean over waters) as ``overall``."""
    table = per_sample.pivot_table(index="model", columns="water_id", values=metric, aggfunc="mean")
    table["overall"] = table.mean(axis=1)
    return table


def main_table(
    per_sample: pd.DataFrame, wide: pd.DataFrame, waters: list[str], ladder: list[tuple[str, str, str]]
) -> pd.DataFrame:
    rmse_log = _water_means(per_sample, "rmse_log")
    rmse = _water_means(per_sample, "chla_rmse")
    mae = _water_means(per_sample, "chla_mae")
    pearson = _water_means(per_sample, "chla_pearson_r")
    reference = rmse_log.loc["persistence", "overall"]
    rows = []
    for variant, group, label in ladder:
        if variant not in rmse_log.index:
            continue
        mean, low, high, _ = _paired(wide, variant, "persistence")
        row = {
            "group": group,
            "model": label,
            "variant": variant,
            "rmse_log": rmse_log.loc[variant, "overall"],
            "rmse_mg_m3": rmse.loc[variant, "overall"],
            "mae_mg_m3": mae.loc[variant, "overall"],
            "pearson_r": pearson.loc[variant, "overall"],
            "skill_vs_persistence": 1.0 - rmse_log.loc[variant, "overall"] / reference,
            "paired_diff_vs_persistence": mean,
            "diff_ci_low": low,
            "diff_ci_high": high,
        }
        row.update({f"rmse_log_{water}": rmse_log.loc[variant, water] for water in waters})
        rows.append(row)
    return pd.DataFrame(rows)


def ablation_table(per_sample: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    rmse_log = _water_means(per_sample, "rmse_log")
    rows = []
    for factor, variants in ABLATIONS:
        baseline = f"{variants[0]}_adapted"
        for variant in variants:
            for suffix in ("zero", "adapted"):
                name = f"{variant}_{suffix}"
                if name not in rmse_log.index:
                    continue
                mean, low, high, _ = _paired(wide, name, baseline)
                rows.append(
                    {
                        "factor": factor,
                        "variant": variant,
                        "regime": suffix,
                        "rmse_log": rmse_log.loc[name, "overall"],
                        "diff_vs_reference": mean,
                        "diff_ci_low": low,
                        "diff_ci_high": high,
                    }
                )
    return pd.DataFrame(rows)


def claims_table(wide: pd.DataFrame, water_of: pd.Series, waters: list[str]) -> pd.DataFrame:
    rows = []
    for better, worse, label in CLAIMS:
        if better not in wide.columns or worse not in wide.columns:
            continue
        mean, low, high, count = _paired(wide, better, worse)
        row = {
            "claim": label,
            "variant": better,
            "against": worse,
            "n_samples": count,
            "mean_diff": mean,
            "ci_low": low,
            "ci_high": high,
            "supported": bool(high < 0),
        }
        rows.append(row | _per_water_diffs(wide, water_of, waters, better, worse))
    return pd.DataFrame(rows)


def horizon_table(per_sample: pd.DataFrame) -> pd.DataFrame:
    """Overall RMSE_log per forecast-lead bin for the rows of the main comparison."""
    present = [(variant, label) for variant, _, label in LADDER if variant in set(per_sample.model)]
    rows = []
    for variant, label in present:
        subset = per_sample[per_sample.model == variant]
        row = {"model": label, "variant": variant}
        for name, group in subset.groupby("horizon_bin", observed=True):
            row[str(name)] = group.groupby("water_id").rmse_log.mean().mean()
        rows.append(row)
    return pd.DataFrame(rows)


def corpus_size_table(per_sample: pd.DataFrame) -> pd.DataFrame:
    """Overall RMSE_log against pretraining-corpus size, the full corpus of 441 waters last."""
    rmse_log = _water_means(per_sample, "rmse_log")
    rows = []
    for size in [*CORPUS_SIZES, 441]:
        variant = "ours" if size == 441 else f"corpus{size}"
        row = {"corpus_waters": size}
        for suffix in ("zero", "adapted"):
            name = f"{variant}_{suffix}"
            row[suffix] = rmse_log.loc[name, "overall"] if name in rmse_log.index else np.nan
        if not (np.isnan(row["zero"]) and np.isnan(row["adapted"])):
            rows.append(row)
    return pd.DataFrame(rows)


def _markdown(frame: pd.DataFrame, title: str) -> str:
    """Render a Markdown table (``DataFrame.to_markdown`` needs the optional tabulate package)."""
    if frame.empty:
        return f"### {title}\n\n_No variants of this table were present in the evaluation._\n"

    def cell(value: object) -> str:
        if isinstance(value, float):
            return "" if pd.isna(value) else f"{value:.4f}"
        return str(value)

    header = "| " + " | ".join(frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([f"### {title}", "", header, rule, *body, ""])


def _expected_variants() -> set[str]:
    regimes = ("zero", "adapted")
    wanted = {variant for table in (LADDER, PARADIGM) for variant, _, _ in table}
    wanted |= {f"{variant}_{regime}" for _, variants in ABLATIONS for variant in variants for regime in regimes}
    return wanted | {variant for claim in CLAIMS for variant in claim[:2]}


def make_paper_tables(run_dir: Path, output_dir: Path) -> None:
    """Write every table as CSV and all of them as ``paper_tables.md``."""
    per_sample = pd.read_csv(run_dir / "per_sample_metrics.csv")
    wide = per_sample.pivot_table(index="sample_id", columns="model", values="rmse_log")
    water_of = per_sample.drop_duplicates("sample_id").set_index("sample_id").water_id.reindex(wide.index)
    waters = sorted(per_sample.water_id.unique())
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "main": (main_table(per_sample, wide, waters, LADDER), "Main comparison (overall RMSE_log)"),
        "paradigm": (
            main_table(per_sample, wide, waters, PARADIGM),
            "Component ablation (w/o pretraining, w/o target adaptation)",
        ),
        "ablations": (ablation_table(per_sample, wide), "Design choices"),
        "claims": (claims_table(wide, water_of, waters), "Claim-by-claim paired tests"),
        "by_horizon": (horizon_table(per_sample), "Overall RMSE_log by forecast lead"),
        "corpus_size": (corpus_size_table(per_sample), "Pretraining-corpus size"),
    }
    # The evaluation only warns about a missing manifest, so name what is absent.
    missing = sorted(_expected_variants() - set(wide.columns))
    counts = per_sample.drop_duplicates("sample_id").water_id.value_counts().sort_index()
    sections = [
        "# Paper tables",
        "",
        f"Evaluation run: `{run_dir}`. Test samples per water: "
        + ", ".join(f"{water} {count}" for water, count in counts.items())
        + f" (total {int(counts.sum())}).",
        "",
        "All errors are RMSE_log (log10 mg/m^3) against the MDN pseudo-label of the real future scene,",
        "inside the fixed Sentinel-1 water-extent mask. Negative differences favor the first variant.",
        "",
    ]
    if missing:
        sections += [f"Variants not present in this evaluation: {', '.join(missing)}.", ""]
    for name, (frame, title) in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
        sections.append(_markdown(frame, title))
    (output_dir / "paper_tables.md").write_text("\n".join(sections), encoding="utf-8")
    print("\n".join(sections))
    if missing:
        print(f"\nWARNING: {len(missing)} expected variants absent: {', '.join(missing)}")
    print(f"\nwrote {output_dir}/paper_tables.md and the CSVs")
