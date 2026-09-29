r"""Training and validation loss against epochs.

With ``--epochs-log`` (``learning_curve_epochs.csv``, the logged losses of the
three-seed data-amount runs) one panel per water and data amount compares
training from scratch on the target water with pretraining + few-shot
transfer. Without ``--seed`` the curves are the mean over the seeds still
training, which early stopping ends at different epochs, with their min-max band.

With ``--run-log`` (the ``training_log.jsonl`` of one run) the figure shows
that run; ``--smooth N`` draws a centered N-epoch running mean over the faint
raw curves.

Usage (the two panels of the paper figure):
    python tools/figures/plot_training_curves.py \
        --run-log outputs/runs/train_scratch_convlstm_georges/training_log.jsonl \
        --output figures/training_curves_convlstm
    python tools/figures/plot_training_curves.py \
        --epochs-log outputs/runs/paper/learning_curve_epochs.csv \
        --seed 42 --waters georges --amounts all --output figures/training_curves
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from _style import REGIMES, TEAL, WATERS, add_layout_arguments, clean_axes, place_legend, save, use_style
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator

AMOUNTS = [("all", "Full record"), ("1y", "Last year")]
LINESTYLES = {"val": "solid", "train": (0, (3, 2))}
RUN_COLOR, RUN_MARKER = TEAL, "v"


def draw_loss(
    ax: Axes,
    epoch: pd.Series,
    loss: pd.Series,
    split: str,
    color: str,
    marker: str,
    markevery: int,
    label: str | None = None,
) -> Line2D:
    """One loss curve: validation solid, training dashed and lighter."""
    (line,) = ax.plot(
        epoch,
        loss,
        color=color,
        linestyle=LINESTYLES[split],
        linewidth=1.2,
        marker=marker,
        markersize=3.2,
        markevery=markevery,
        alpha=0.95 if split == "val" else 0.85,
        label=label,
    )
    return line


def plot_run(args: argparse.Namespace) -> None:
    """Training and validation loss of one run."""
    log = pd.read_json(args.run_log, lines=True)
    log = log[log.phase == "forecast"].assign(epoch=lambda frame: frame.epoch + 1)

    use_style()
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    handles = []
    for split, label in (("val", "Val."), ("train", "Train")):
        loss = log[f"{split}_loss"]
        if args.smooth > 1:
            ax.plot(log.epoch, loss, color=RUN_COLOR, linestyle=LINESTYLES[split], linewidth=0.6, alpha=0.25, zorder=1)
            loss = loss.rolling(args.smooth, center=True, min_periods=1).mean()
        handles.append(draw_loss(ax, log.epoch, loss, split, RUN_COLOR, RUN_MARKER, args.markevery, label))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss", fontsize=8)
    ax.set_ylim(*(args.ylim or (0.1, 0.7)))
    ax.xaxis.set_major_locator(MultipleLocator(50))
    clean_axes(ax)
    place_legend(fig, ax, handles, args)
    save(fig, args.output)


def plot_learning_curve_runs(args: argparse.Namespace) -> None:
    """Scratch against transfer, one panel per water and data amount."""
    waters = [(water, name) for water, name in WATERS if args.waters is None or water in args.waters]
    amounts = [(amount, label) for amount, label in AMOUNTS if args.amounts is None or amount in args.amounts]
    epochs = pd.read_csv(args.epochs_log)
    epochs = epochs[epochs.phase == "forecast"]
    if args.seed is not None:
        epochs = epochs[epochs.seed == args.seed]
    epochs = epochs.assign(epoch=epochs.epoch + 1)

    use_style()
    n_panels = len(waters) * len(amounts)
    # One row of panels when either list has one entry, else a data amount by water grid.
    single_row = len(waters) == 1 or len(amounts) == 1
    if single_row:
        size = tuple(args.figsize) if n_panels == 1 else (3.0 * n_panels, 2.7)
        fig, axes = plt.subplots(1, n_panels, figsize=size, sharey=True, squeeze=False)
    else:
        fig, axes = plt.subplots(len(amounts), len(waters), figsize=(6.8, 4.0), sharex=True, sharey=True)
    panels = [(row, col, amount, water) for row, amount in enumerate(amounts) for col, water in enumerate(waters)]
    for ax, (row, col, (amount, amount_label), (water, name)) in zip(axes.ravel(), panels, strict=True):
        runs = epochs[(epochs.data_amount == amount) & (epochs.water_id == water)]
        for regime, (_, color, marker) in REGIMES.items():
            for split in ("train", "val"):
                curve = runs[runs.regime == regime].groupby("epoch")[f"{split}_loss"].agg(["mean", "min", "max"])
                curve = curve.reset_index()
                if args.seed is None:
                    ax.fill_between(curve.epoch, curve["min"], curve["max"], color=color, alpha=0.15, linewidth=0)
                draw_loss(ax, curve.epoch, curve["mean"], split, color, marker, 1 if len(curve) <= 6 else 2)
        if single_row:
            if n_panels > 1:
                ax.set_title(f"{name}, {amount_label.lower()}", fontsize=8.5, pad=4)
            ax.set_xlabel("Epoch")
            if row == 0 and col == 0:
                ax.set_ylabel("Loss", fontsize=8)
        else:
            if row == 0:
                ax.set_title(name, fontsize=8.5, pad=4)
            if col == 0:
                ax.set_ylabel(f"{amount_label}\nLoss", fontsize=8)
            if row == len(amounts) - 1:
                ax.set_xlabel("Epoch")
        ax.set_ylim(*(args.ylim or (0.2, 0.85)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        clean_axes(ax)

    val_word, train_word = ("val.", "train.") if args.short_labels else ("validation", "training")
    handles = [
        Line2D(
            [],
            [],
            color=regime.color,
            linewidth=1.2,
            linestyle=LINESTYLES[split],
            marker=regime.marker,
            markersize=3.2,
            label=f"{regime.label}, {word}",
        )
        for split, word in (("val", val_word), ("train", train_word))
        for regime in REGIMES.values()
    ]
    if n_panels == 1:
        place_legend(fig, axes[0, 0], handles, args, columnspacing=0.6, handlelength=1.3)
    else:
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=2 if single_row else 4,
            frameon=False,
            fontsize=7.5,
            bbox_to_anchor=(0.5, args.legend_y),
            handletextpad=0.5,
            columnspacing=1.0,
        )
        fig.tight_layout(pad=0.4, rect=(0, 0.16 if single_row else 0.06, 1, 1))
        if args.vertical:
            fig.subplots_adjust(top=args.vertical[0], bottom=args.vertical[1])
    save(fig, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--epochs-log", type=Path, help="learning_curve_epochs.csv of the data-amount runs")
    source.add_argument("--run-log", type=Path, help="training_log.jsonl of one run")
    parser.add_argument("--output", type=Path, required=True, help="Output path without extension")
    parser.add_argument(
        "--ylim", type=float, nargs=2, default=None, help="Loss axis range (default 0.1-0.7 for a run, else 0.2-0.85)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Draw one seed instead of the mean and range")
    parser.add_argument("--waters", nargs="*", default=None, help="Water ids to draw (default: all)")
    parser.add_argument("--amounts", nargs="*", default=None, help="Data amounts to draw (default: all, 1y)")
    parser.add_argument("--short-labels", action="store_true", help="Abbreviate validation and training")
    parser.add_argument("--markevery", type=int, default=10, help="Marker spacing in epochs of a single run")
    parser.add_argument("--smooth", type=int, default=0, help="Running-mean window over a single run (0: raw only)")
    add_layout_arguments(parser)
    args = parser.parse_args()
    if args.run_log is not None:
        plot_run(args)
    else:
        plot_learning_curve_runs(args)


if __name__ == "__main__":
    main()
