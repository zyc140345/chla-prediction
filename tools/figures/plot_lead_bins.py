r"""Overall RMSE_log of the main methods in forecast-lead bins, as a heatmap.

Reads ``by_horizon.csv`` of the paper tables (one row per method, one column
per lead bin) and ``per_sample_metrics.csv`` of the evaluation run, and draws
one row per lead bin and one column per method with the value in each cell,
the best of each bin in bold, and the number of test samples per bin in the
row labels.

Usage:
    python tools/figures/plot_lead_bins.py --tables-dir outputs/runs/paper \
        --evaluation-dir outputs/runs/evaluation_transfer --output figures/lead_bins
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _style import INK, METHODS, save, use_style
from matplotlib.colors import PowerNorm

BINS = [("<=7d", r"$\leq$7 d"), ("8-15d", "8–15 d"), ("16-30d", "16–30 d"), (">30d", ">30 d")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables-dir", type=Path, required=True, help="Directory of the paper tables")
    parser.add_argument("--evaluation-dir", type=Path, required=True, help="Evaluation run of the paper tables")
    parser.add_argument("--output", type=Path, required=True, help="Output path without extension")
    args = parser.parse_args()

    by_horizon = pd.read_csv(args.tables_dir / "by_horizon.csv").set_index("variant")
    scores = np.array([[by_horizon.loc[variant, lead] for variant, _ in METHODS] for lead, _ in BINS])
    samples = pd.read_csv(args.evaluation_dir / "per_sample_metrics.csv", usecols=["sample_id", "horizon_bin"])
    counts = samples.drop_duplicates("sample_id").horizon_bin.value_counts()

    use_style(axes=False)
    fig, ax = plt.subplots(figsize=(5.9, 1.6))
    # The gamma stretches the high-error end, where most methods sit.
    norm = PowerNorm(1.3, vmin=scores.min() - 0.01, vmax=scores.max() + 0.01)
    image = ax.imshow(scores, cmap="viridis_r", aspect="auto", norm=norm)
    for row, col in np.ndindex(scores.shape):
        value = scores[row, col]
        ax.text(
            col,
            row,
            f"{value:.3f}",
            ha="center",
            va="center",
            fontsize=7.5,
            color="white" if norm(value) > 0.36 else INK,
            fontweight="bold" if value == scores[row].min() else "normal",
        )
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([label for _, label in METHODS], fontsize=7.5)
    ax.set_yticks(range(len(BINS)))
    ax.set_yticklabels([f"{label} ({counts[lead]})" for lead, label in BINS], fontsize=7.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # White gap before the proposed model, the last column.
    ax.axvline(len(METHODS) - 1.5, color="white", linewidth=2)
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(r"$\mathrm{RMSE}_{\log}$", fontsize=8)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.outline.set_linewidth(0.4)
    fig.tight_layout(pad=0.4)
    save(fig, args.output)


if __name__ == "__main__":
    main()
