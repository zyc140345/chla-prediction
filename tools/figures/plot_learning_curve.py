r"""Data-amount learning curve: test RMSE_log against the size of the adaptation set.

Reads ``learning_curve_seeds.csv`` (one row per regime, data amount and seed),
``learning_curve_epochs.csv`` (the training sequences of each run) and
``paradigm.csv`` (the zero-shot score) from the paper tables. Draws the mean
and standard deviation over seeds of training from scratch on the target
water and of pretraining + few-shot transfer, against the adaptation
sequences summed over the evaluation waters, with the zero-shot score as a
horizontal line.

Usage:
    python tools/figures/plot_learning_curve.py --tables-dir outputs/runs/paper \
        --output figures/learning_curve
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from _style import MUTED, REGIMES, add_layout_arguments, clean_axes, place_legend, save, use_style


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables-dir", type=Path, required=True, help="Directory of the paper tables")
    parser.add_argument("--output", type=Path, required=True, help="Output path without extension")
    add_layout_arguments(parser)
    args = parser.parse_args()

    seeds = pd.read_csv(args.tables_dir / "learning_curve_seeds.csv")
    stats = seeds.groupby(["regime", "data_amount"])["rmse_log"].agg(["mean", "std"])
    epochs = pd.read_csv(args.tables_dir / "learning_curve_epochs.csv")
    # Adaptation sequences per data amount, summed over the waters; the x axis in increasing order.
    sequences = epochs.drop_duplicates(["data_amount", "water_id"]).groupby("data_amount").train_sequences.sum()
    sequences = sequences.sort_values()
    paradigm = pd.read_csv(args.tables_dir / "paradigm.csv").set_index("variant")
    zero_shot = float(paradigm.loc["ours_zero", "rmse_log"])

    use_style()
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    x = list(range(len(sequences)))
    for regime, (label, color, marker) in REGIMES.items():
        ax.errorbar(
            x,
            [stats.loc[(regime, amount), "mean"] for amount in sequences.index],
            yerr=[stats.loc[(regime, amount), "std"] for amount in sequences.index],
            color=color,
            marker=marker,
            markersize=3.2,
            linewidth=1.2,
            capsize=2.5,
            capthick=0.7,
            elinewidth=0.7,
            label=label,
            zorder=3,
        )
    ax.axhline(zero_shot, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2, label="Zero-shot")

    ax.set_xticks(x)
    ax.set_xticklabels([str(count) for count in sequences])
    ax.set_xlim(-0.35, len(sequences) - 0.65)
    ax.set_xlabel("Adaptation data volume")
    ax.set_ylabel(r"Test $\mathrm{RMSE}_{\log}$", fontsize=8)
    ax.set_ylim(0.28, 0.64)
    clean_axes(ax)
    handles, _ = ax.get_legend_handles_labels()
    place_legend(fig, ax, handles, args)
    save(fig, args.output)


if __name__ == "__main__":
    main()
