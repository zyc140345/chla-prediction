"""Shared look and vocabulary of the paper figures.

Colors, the training regimes, the evaluation waters (``configs/target_waters.csv``),
the methods of the main comparison and the design-choice variants, plus the
axes, legend and file conventions every figure script uses.

Bold labels need TeX Gyre Pagella; matplotlib on macOS finds only the
regular face of Palatino.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import NamedTuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from chla_prediction.evaluation.paper_tables import ABLATIONS, LADDER

# Agg on every host: the macOS backend shifts tight layouts by a pixel or two.
matplotlib.use("Agg")

INK = "#1f1f1f"
MUTED = "#8a8a8a"
GRID = "#d9d9d9"
SCRATCH = "#c8552d"
TRANSFER = "#1f4e9c"
TEAL = "#3f7f8c"


class Regime(NamedTuple):
    label: str
    color: str
    marker: str


# Trained from scratch on the target water, and pretrained then transferred.
REGIMES = {
    "scratch": Regime("From scratch", SCRATCH, "s"),
    "adapted": Regime("Transfer", TRANSFER, "o"),
}

with (Path(__file__).resolve().parents[2] / "configs/target_waters.csv").open() as _stream:
    WATERS = [(row["water_id"], row["name"]) for row in csv.DictReader(_stream)]
WATER_NAMES = dict(WATERS)


def _figure_methods() -> list[tuple[str, str]]:
    """The main comparison as (variant, label): LADDER order with random forest moved before XGBoost."""
    methods = [(variant, label) for variant, _, label in LADDER]
    random_forest = next(method for method in methods if method[0] == "rf_pixel")
    methods.remove(random_forest)
    methods.insert([variant for variant, _ in methods].index("xgboost_pixel"), random_forest)
    return methods


METHODS = _figure_methods()

_DESIGN_LABELS = {
    "Encoder": {
        "ours": "MDN pretrained (ours)",
        "rand": "MDN from scratch",
        "resnet18": "ResNet-18",
        "resnet50": "ResNet-50",
        "convnext": "ConvNeXt",
        "convnext_l": "ConvNeXt-L",
        "clay_lora": "Clay v1.5 + LoRA",
    },
    "Temporal module": {
        "ours": "Temporal attention (ours)",
        "convgru": "ConvGRU",
        "convlstm": "ConvLSTM",
        "simvp": "SimVP + TAU",
        "stt": "Space-time Transformer",
    },
}
# Encoder and temporal-module ablations as (factor, [(variant, label), ...]), in the order of ABLATIONS.
DESIGN_CHOICES = [
    (factor, [(variant, _DESIGN_LABELS[factor][variant]) for variant in variants])
    for factor, variants in ABLATIONS
    if factor in _DESIGN_LABELS
]


def use_style(font_size: float = 8.5, axes: bool = True) -> None:
    """Serif type for every figure; ``axes`` also draws the axes in ink."""
    params = {
        "font.family": "serif",
        "font.serif": ["Palatino", "TeX Gyre Pagella", "DejaVu Serif"],
        "font.size": font_size,
        "mathtext.fontset": "stix",
    }
    if axes:
        params |= {"axes.edgecolor": INK, "axes.linewidth": 0.6}
    plt.rcParams.update(params)


def clean_axes(ax: Axes, grid: str = "y") -> None:
    """Light grid along ``grid`` ("x", "y" or "both") behind the data, no top or right spine, short ticks."""
    ax.grid(True, axis=grid, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=3, width=0.6, labelsize=7.5)


def add_layout_arguments(parser: argparse.ArgumentParser) -> None:
    """Size and legend options of a single-panel figure, so figures placed side by side align."""
    parser.add_argument("--figsize", type=float, nargs=2, default=(3.3, 3.0), help="Width and height in inches")
    parser.add_argument("--legend-inside", default=None, help="Place the legend inside the axes at this location")
    parser.add_argument("--legend-cols", type=int, default=2, help="Columns of the legend below the axes")
    parser.add_argument("--legend-y", type=float, default=-0.01, help="Figure fraction of the legend bottom")
    parser.add_argument(
        "--bottom",
        type=float,
        default=None,
        help="Figure fraction reserved for the legend below the axes (default 0.13 for two columns, else 0.24)",
    )
    parser.add_argument(
        "--vertical",
        type=float,
        nargs=2,
        default=None,
        metavar=("TOP", "BOTTOM"),
        help="Fixed top and bottom of the axes as figure fractions",
    )


def place_legend(fig: Figure, ax: Axes, handles: list[Artist], args: argparse.Namespace, **below: float) -> None:
    """Legend inside ``ax`` or below it as set by ``add_layout_arguments``, then the layout.

    ``below`` overrides the spacing of the legend below the axes.
    """
    if args.legend_inside:
        ax.legend(
            handles=handles,
            loc=args.legend_inside,
            frameon=False,
            fontsize=6.5,
            handletextpad=0.4,
            labelspacing=0.3,
            handlelength=1.4,
        )
        fig.tight_layout(pad=0.4)
    else:
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=args.legend_cols,
            frameon=False,
            fontsize=7.5,
            bbox_to_anchor=(0.5, args.legend_y),
            handletextpad=0.4,
            **({"columnspacing": 0.8, "handlelength": 1.6} | below),
        )
        bottom = args.bottom if args.bottom is not None else (0.13 if args.legend_cols == 2 else 0.24)
        fig.tight_layout(pad=0.4, rect=(0, bottom, 1, 1))
    if args.vertical:
        fig.subplots_adjust(top=args.vertical[0], bottom=args.vertical[1])


def save(fig: Figure, output: Path, png_dpi: int = 200, pdf_dpi: float | str = "figure") -> None:
    """Write ``output`` (a path without extension) as PDF and PNG."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), dpi=pdf_dpi)
    fig.savefig(output.with_suffix(".png"), dpi=png_dpi)
