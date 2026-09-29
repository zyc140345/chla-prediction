r"""Forecast Chl-a fields of the main methods, one row per evaluation water.

Reads the ``.npz`` maps written by ``export_field_maps.py`` and draws the MDN
pseudo-label of the target scene followed by one column per method. Each row
shares one logarithmic color scale with its own colorbar, panels are cropped
to the water-extent mask, and each forecast shows its RMSE_log in a corner.

Usage:
    python tools/figures/plot_field_comparison.py --maps outputs/runs/field_maps \
        --output figures/field_maps --panel-width 0.78 \
        --cases baogu_20250607_20250915 loweswater_20250508_20250513 \
                georges_20250920_20251002 wentzel_20241119_20241124 \
                hushan_20250106_20250111
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from _style import INK, METHODS, WATER_NAMES, save, use_style
from matplotlib import ticker
from matplotlib.colors import LogNorm

COLUMNS = [("reference", "MDN label"), *METHODS]


def crop_bounds(valid: np.ndarray, margin: int) -> tuple[slice, slice]:
    """Rows and columns of the valid pixels' bounding box, widened by ``margin``."""
    rows, cols = np.where(valid)
    return (
        slice(max(rows.min() - margin, 0), rows.max() + margin + 1),
        slice(max(cols.min() - margin, 0), cols.max() + margin + 1),
    )


def iso_date(compact: str) -> str:
    return f"{compact[:4]}-{compact[4:6]}-{compact[6:]}"


def load_case(path: Path, margin: int) -> tuple[dict, dict[str, np.ndarray]]:
    """Metadata and the cropped field of every column, NaN outside the valid pixels."""
    data = np.load(path)
    window = crop_bounds(data["valid"], margin)
    valid = data["valid"][window]
    fields = {}
    for key, _ in COLUMNS:
        field = data["reference" if key == "reference" else f"model_{key}"][window].astype(float)
        field[~valid] = np.nan
        fields[key] = field
    return json.loads(str(data["meta"])), fields


def colorbar_ticks(vmin: float, vmax: float) -> np.ndarray:
    """1-2-5 ticks when the range spans enough, otherwise every integer mantissa, so ticks stay round."""
    coarse = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], dtype=float)
    fine = np.array([k * 10.0**e for e in range(4) for k in range(1, 10)])
    ticks = coarse[(coarse >= vmin) & (coarse <= vmax)]
    if len(ticks) < 2:
        ticks = fine[(fine >= vmin) & (fine <= vmax)]
        ticks = ticks[:: max(1, len(ticks) // 3)]
    return ticks if len(ticks) >= 2 else np.round(np.geomspace(vmin, vmax, 3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maps", type=Path, required=True, help="Directory of exported .npz field maps")
    parser.add_argument("--cases", nargs="+", required=True, help="Sample ids, one per row, top to bottom")
    parser.add_argument("--output", type=Path, required=True, help="Output path without extension")
    parser.add_argument("--margin", type=int, default=4, help="Pixels kept around the valid pixels")
    parser.add_argument("--panel-width", type=float, default=1.22, help="Panel width in inches")
    args = parser.parse_args()

    use_style(8, axes=False)
    cases = [load_case(args.maps / f"{case}.npz", args.margin) for case in args.cases]

    n_rows, n_cols = len(cases), len(COLUMNS)
    panel_w = args.panel_width
    fig_w = n_cols * panel_w + 1.1  # row labels on the left, colorbars on the right
    fig_h = n_rows * panel_w * 0.92 + 0.35
    fig = plt.figure(figsize=(fig_w, fig_h))
    grid = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1] * n_cols + [0.07],
        left=0.115,
        right=0.952,
        top=0.955,
        bottom=0.015,
        wspace=0.06,
        hspace=0.12,
    )
    cmap = matplotlib.colormaps["viridis"].with_extremes(bad="0.92")

    for row, (meta, fields) in enumerate(cases):
        pooled = np.concatenate([field[np.isfinite(field)] for field in fields.values()])
        low, high = np.percentile(pooled, [2, 98])
        norm = LogNorm(vmin=max(low, meta["chla_range"][0]), vmax=max(high, low * 1.5))
        for col, (key, title) in enumerate(COLUMNS):
            ax = fig.add_subplot(grid[row, col])
            image = ax.imshow(fields[key], norm=norm, cmap=cmap, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("0.6")
            if row == 0:
                ax.set_title(title, fontsize=8, pad=3)
            if key != "reference":
                ax.text(
                    0.03,
                    0.04,
                    f"{meta['rmse_log'][key]:.3f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=6.5,
                    color=INK,
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
                )
            if col == 0:
                water = WATER_NAMES.get(meta["water_id"], meta["water_id"])
                ax.set_ylabel(
                    f"{water}\n{iso_date(meta['origin_date'])} $\\rightarrow$\n{iso_date(meta['target_date'])} "
                    f"({meta['horizon_days']} d)",
                    fontsize=6.8,
                    labelpad=5,
                )
        cbar = fig.colorbar(image, cax=fig.add_subplot(grid[row, n_cols]))
        cbar.set_ticks(colorbar_ticks(norm.vmin, norm.vmax))
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        cbar.minorticks_off()
        cbar.ax.tick_params(labelsize=6, length=2, width=0.4, pad=1.5)
        cbar.outline.set_linewidth(0.4)
    fig.text(0.984, 0.5, "Chl-a (mg m$^{-3}$)", rotation=90, ha="center", va="center", fontsize=7.5)
    save(fig, args.output, png_dpi=220, pdf_dpi=300)


if __name__ == "__main__":
    main()
