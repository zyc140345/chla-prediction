r"""Limits of pseudo-label supervision: two panels written as separate files.

``<output>_insitu``: MDN pseudo-label against same-day in-situ Chl-a at Hushan
Reservoir, from ``mdn_insitu_matchups.csv`` (``export_insitu_matchups.py``).
``<output>_scaling``: overall test RMSE_log against the number of pretraining
waters, from ``corpus_size.csv``.

Both panels get the same axes top and bottom in figure fractions, so they
align when placed side by side at the same height.

Usage:
    python tools/figures/plot_label_limits.py --tables-dir outputs/runs/paper \
        --output figures/label_limits --figsize 3.85 1.75
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _style import MUTED, REGIMES, TEAL, TRANSFER, clean_axes, save, use_style
from matplotlib.axes import Axes

SCALING_LINES = [("zero", "Zero-shot", TEAL, "^"), ("adapted", *REGIMES["adapted"])]


def draw_matchups(ax: Axes, matchups: pd.DataFrame) -> None:
    """Same-day matchups only: a sample a day or two from the scene mixes real change into the comparison."""
    matchups = matchups[(matchups.days_offset == 0) & np.isfinite(matchups.mdn) & np.isfinite(matchups.insitu)]
    low, high = 10.0, 300.0
    ax.plot([low, high], [low, high], color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    ax.scatter(
        matchups.insitu,
        matchups.mdn,
        s=16,
        marker="o",
        facecolors=TRANSFER,
        edgecolors=TRANSFER,
        linewidths=0.7,
        zorder=3,
    )
    log_label, log_insitu = np.log10(matchups.mdn), np.log10(matchups.insitu)
    rmse = np.sqrt(np.mean((log_label - log_insitu) ** 2))
    bias = np.mean(log_label - log_insitu)
    pearson_r = np.corrcoef(log_label, log_insitu)[0, 1]
    ax.text(
        0.96,
        0.04,
        f"N = {len(matchups)}\n$\\mathrm{{RMSE}}_{{\\log}}$ = {rmse:.3f}\nbias = {bias:+.3f}\n$r$ = {pearson_r:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel(r"In situ ($\mathrm{mg\,m^{-3}}$)")
    ax.set_ylabel(r"Pseudo-label ($\mathrm{mg\,m^{-3}}$)")
    clean_axes(ax, grid="both")
    ax.set_aspect("equal")


def draw_scaling(ax: Axes, corpus: pd.DataFrame) -> None:
    corpus = corpus.sort_values("corpus_waters")
    for column, label, color, marker in SCALING_LINES:
        ax.plot(
            corpus.corpus_waters,
            corpus[column],
            color=color,
            marker=marker,
            markersize=3.6,
            linewidth=1.2,
            label=label,
            zorder=3,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(corpus.corpus_waters)
    ax.set_xticklabels([str(int(size)) for size in corpus.corpus_waters])
    ax.minorticks_off()
    ax.set_xlabel("Pretraining waters")
    ax.set_ylabel(r"Overall test $\mathrm{RMSE}_{\log}$")
    ax.legend(loc="upper right", frameon=False, fontsize=7.5, handletextpad=0.4)
    clean_axes(ax)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables-dir", type=Path, required=True, help="Directory of the paper tables")
    parser.add_argument("--output", type=Path, required=True, help="Output stem, suffixed _insitu and _scaling")
    parser.add_argument(
        "--matchups", type=Path, default=None, help="Matchup CSV (default: <tables-dir>/mdn_insitu_matchups.csv)"
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(5.45, 2.35),
        help="Total width of the two panels and their common height, inches",
    )
    args = parser.parse_args()

    matchups = pd.read_csv(args.matchups or args.tables_dir / "mdn_insitu_matchups.csv")
    corpus = pd.read_csv(args.tables_dir / "corpus_size.csv")
    use_style()
    width, height = args.figsize
    insitu_fig, insitu_ax = plt.subplots(figsize=(height, height))
    draw_matchups(insitu_ax, matchups)
    scaling_fig, scaling_ax = plt.subplots(figsize=(width - height, height))
    draw_scaling(scaling_ax, corpus)
    panels = [("_insitu", insitu_fig, insitu_ax), ("_scaling", scaling_fig, scaling_ax)]
    for _, fig, _ in panels:
        fig.tight_layout(pad=0.4)
    bottom = max(ax.get_position().y0 for _, _, ax in panels)
    top = min(ax.get_position().y1 for _, _, ax in panels)
    for suffix, fig, ax in panels:
        position = ax.get_position()
        ax.set_position([position.x0, bottom, position.width, top - bottom])
        save(fig, args.output.with_name(args.output.name + suffix))


if __name__ == "__main__":
    main()
