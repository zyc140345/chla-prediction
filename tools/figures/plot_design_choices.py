r"""Design-choice ablation: encoder and temporal-module alternatives.

Reads ``ablations.csv`` and draws, for the ``Encoder`` and ``Temporal module``
factors, one row per variant with (left) the zero-shot and adapted overall
RMSE_log joined as a dumbbell and (right) the adapted paired difference to
the proposed model with its 95% bootstrap interval. ``--params`` appends the
trainable parameters from ``count_parameters.py`` to each row label.

Usage:
    python tools/figures/plot_design_choices.py --tables-dir outputs/runs/paper \
        --output figures/design_choices
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from _style import DESIGN_CHOICES, INK, MUTED, REGIMES, clean_axes, save, use_style
from matplotlib.lines import Line2D

ADAPTED = REGIMES["adapted"]
OURS_BACKGROUND = "#f1f1f1"
ZERO_SHOT_MARKER = {
    "marker": "o",
    "markersize": 5,
    "markerfacecolor": "white",
    "markeredgecolor": ADAPTED.color,
    "markeredgewidth": 1.0,
    "linestyle": "none",
}


def parameter_label(counts: pd.DataFrame, factor: str, variant: str) -> str:
    """Trainable parameters in millions; a star marks a module with many more frozen parameters."""
    row = counts.loc[(factor, variant)]
    millions = row["trainable_M"]
    text = f"{millions:.1f}M" if millions >= 1 else f"{millions:.2f}M"
    return text + ("*" if row["total_M"] > row["trainable_M"] * 1.5 else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables-dir", type=Path, required=True, help="Directory of the paper tables")
    parser.add_argument("--output", type=Path, required=True, help="Output path without extension")
    parser.add_argument(
        "--params", type=Path, default=None, help="module_parameters.csv written by count_parameters.py"
    )
    args = parser.parse_args()

    ablations = pd.read_csv(args.tables_dir / "ablations.csv").set_index(["factor", "variant", "regime"])
    counts = pd.read_csv(args.params).set_index(["group", "variant"]) if args.params is not None else None

    use_style()
    heights = [len(variants) for _, variants in DESIGN_CHOICES]
    fig, axes = plt.subplots(
        len(heights),
        2,
        figsize=(5.45, 3.15),
        gridspec_kw={"height_ratios": heights, "width_ratios": [1.35, 1]},
        sharex="col",
    )
    for (ax_l, ax_r), (factor, variants) in zip(axes, DESIGN_CHOICES, strict=True):
        for ax in (ax_l, ax_r):
            clean_axes(ax, grid="x")
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
        ys = list(range(len(variants)))[::-1]
        for y, (variant, _) in zip(ys, variants, strict=True):
            zero = ablations.loc[(factor, variant, "zero"), "rmse_log"]
            adapted = ablations.loc[(factor, variant, "adapted")]
            if variant == "ours":
                for ax in (ax_l, ax_r):
                    ax.axhspan(y - 0.45, y + 0.45, color=OURS_BACKGROUND, zorder=0, linewidth=0)
            ax_l.plot([zero, adapted.rmse_log], [y, y], color=MUTED, linewidth=1.2, zorder=2)
            ax_l.plot(zero, y, zorder=3, **ZERO_SHOT_MARKER)
            ax_l.plot(adapted.rmse_log, y, marker="o", markersize=5, color=ADAPTED.color, linestyle="none", zorder=4)
            if variant != "ours":
                low, high = adapted.diff_ci_low, adapted.diff_ci_high
                ax_r.plot([low, high], [y, y], color=INK, linewidth=1.2, zorder=3)
                for end in (low, high):
                    ax_r.plot([end, end], [y - 0.12, y + 0.12], color=INK, linewidth=0.9, zorder=3)
                ax_r.plot(
                    adapted.diff_vs_reference, y, marker="o", markersize=4.5, color=INK, linestyle="none", zorder=4
                )
            else:
                ax_r.plot(0, y, marker="|", markersize=8, color=INK, linestyle="none", zorder=4)
        labels = [
            label + (f" ({parameter_label(counts, factor, variant)})" if counts is not None else "")
            for variant, label in variants
        ]
        ax_l.set_yticks(ys)
        ax_l.set_yticklabels(labels, fontsize=8)
        ax_r.set_yticks(ys)
        ax_r.set_yticklabels([])
        ax_r.axvline(0, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
        for ax in (ax_l, ax_r):
            ax.set_ylim(-0.6, len(variants) - 0.4)
        ax_l.set_title(factor, loc="left", fontsize=8.5, pad=3)
    axes[0][0].set_xlim(0.374, 0.425)
    axes[0][1].set_xlim(-0.012, 0.055)
    axes[1][0].set_xlabel(r"Overall $\mathrm{RMSE}_{\log}$")
    axes[1][1].set_xlabel("Difference to ours")
    handles = [
        Line2D([], [], label="Zero-shot", **ZERO_SHOT_MARKER),
        Line2D([], [], marker="o", markersize=5, color=ADAPTED.color, linestyle="none", label=ADAPTED.label),
        Line2D(
            [],
            [],
            marker="o",
            markersize=4.5,
            color=INK,
            linestyle="-",
            linewidth=1.2,
            label="Mean and 95% CI over test samples",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=7.5,
        bbox_to_anchor=(0.5, -0.005),
        handletextpad=0.5,
        columnspacing=1.5,
    )
    fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.8, rect=(0, 0.07, 1, 1))
    save(fig, args.output)


if __name__ == "__main__":
    main()
