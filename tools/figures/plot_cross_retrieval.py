r"""Transfer across retrieval algorithms: test RMSE_log per water and retrieval algorithm.

Reads ``generalization_gap_per_water_seed.csv`` and
``generalization_gap_overall_seeds.csv`` of the cross-retrieval runs and
draws, for each retrieval algorithm, paired bars (trained from scratch against
pretrained and transferred) per water and for the mean over waters, with the
standard deviation over seeds as error bars. The waters are those of the runs.

Usage:
    python tools/figures/plot_cross_retrieval.py --tables-dir outputs/runs/cross_retrieval/paper \
        --output figures/cross_retrieval --figsize 5.45 1.8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _style import INK, REGIMES, WATERS, clean_axes, save, use_style

PRODUCTS = [("ndci", "NDCI"), ("three_band", "3BDA")]
# Tick labels break at spaces; a one-word name needs its own break.
TICK_LABELS = {"loweswater": "Lowes-\nwater"}
MEAN = "__mean__"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables-dir", type=Path, required=True, help="Directory of the cross-retrieval tables")
    parser.add_argument("--output", type=Path, required=True, help="Output path without extension")
    parser.add_argument("--figsize", type=float, nargs=2, default=(5.45, 2.4))
    parser.add_argument(
        "--ymin",
        type=float,
        nargs="*",
        default=None,
        help="Lower y-limit per panel (default: lowest bar minus its deviation and a margin, on a 0.05 grid)",
    )
    args = parser.parse_args()

    per_water = pd.read_csv(args.tables_dir / "generalization_gap_per_water_seed.csv")
    overall = pd.read_csv(args.tables_dir / "generalization_gap_overall_seeds.csv")
    groups = [
        (water, TICK_LABELS.get(water, name.replace(" ", "\n")))
        for water, name in WATERS
        if water in set(per_water.water_id)
    ] + [(MEAN, "Mean")]

    use_style()
    fig, axes = plt.subplots(1, 2, figsize=tuple(args.figsize), sharey=False)
    width = 0.36
    x = np.arange(len(groups))
    for panel, (ax, (product, title)) in enumerate(zip(axes, PRODUCTS, strict=True)):
        lows = []
        for k, (regime, (label, color, _)) in enumerate(REGIMES.items()):
            water_runs = per_water[(per_water.retrieval == product) & (per_water.regime == regime)]
            overall_runs = overall[(overall.retrieval == product) & (overall.regime == regime)]
            seeds = [
                (overall_runs if water == MEAN else water_runs[water_runs.water_id == water]).test.values
                for water, _ in groups
            ]
            mean = [scores.mean() for scores in seeds]
            std = [scores.std(ddof=1) for scores in seeds]
            lows.extend(m - sd for m, sd in zip(mean, std, strict=True))
            ax.bar(
                x + (k - 0.5) * width,
                mean,
                width=width,
                color=color,
                alpha=0.85,
                linewidth=0,
                label=label,
                zorder=2,
                yerr=std,
                error_kw={"ecolor": INK, "elinewidth": 0.7, "capsize": 2, "capthick": 0.7},
            )
        ax.axvline(len(groups) - 1.5, color="0.6", linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in groups], fontsize=7.5)
        ax.set_title(title, fontsize=8.5, pad=4)
        clean_axes(ax)
        if args.ymin:
            ymin = args.ymin[min(panel, len(args.ymin) - 1)]
        else:
            ymin = max(0.0, np.floor((min(lows) - 0.02) / 0.05) * 0.05)
        ax.set_ylim(bottom=ymin)
    axes[0].set_ylabel(r"Test $\mathrm{RMSE}_{\log}$")
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=7.5,
        bbox_to_anchor=(0.5, -0.005),
        handletextpad=0.5,
        columnspacing=1.5,
    )
    fig.tight_layout(pad=0.4, w_pad=1.2, rect=(0, 0.1, 1, 1))
    save(fig, args.output)


if __name__ == "__main__":
    main()
