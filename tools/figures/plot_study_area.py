"""Study area: world map of the pretraining and evaluation waters, and a panel per evaluation water.

The base map uses the Natural Earth 110m shapefiles in ``data/naturalearth``
(public domain, https://www.naturalearthdata.com) and highlights the countries
of the evaluation waters. Pretraining waters are the candidates in
``configs/pretraining_corpus.csv`` listed in ``--corpus-list``. Each panel
shows the extent of one evaluation water, the union of the valid masks of its
field maps (``outputs/runs/field_maps``, the registered bounding box at
10 m), on a longitude-latitude grid; panels are numbered from west to east.

Usage:
    python tools/figures/plot_study_area.py --output figures/study_area
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _style import INK, TRANSFER, save, use_style
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle

LAND = "#c4c4c4"
LAND_HIGHLIGHT = "#f7dc8e"
OCEAN = "#e3f4f9"
GRATICULE = "#6e6e6e"
TARGET = "#d62728"
WATER = "#2b6cb0"
WATER_EDGE = "#0f2f55"
WEDGE = "#5fb6dc"
# Sentinel-2 L2A scenes downloaded per evaluation water; no config file lists them.
EVAL_SCENES = {
    "baogu": 115,
    "loweswater": 447,
    "georges": 261,
    "wentzel": 461,
    "hushan": 298,
}
# Hushan has no HydroLAKES record, so target_waters.csv has no area; it follows Miao et al. (2024).
AREA_OVERRIDE_KM2 = {"hushan": 4.94}
# HydroLAKES lake_type codes.
WATER_TYPE = {1: "Lake", 2: "Reservoir", 3: "Reservoir"}
# Extra padding (pixels at 10 m) around a water whose default window leaves no
# corner free of water for the name and the scale bar.
PAD = {"loweswater": 22}
# (name corner, scale-bar corner) where the automatic choice still overlaps
# water. Corners: tl, tr, bl, br; "bl_up" raises the bar in the bottom-left corner.
PLACEMENT = {
    "georges": ("tr", "bl_up"),
}


def read_waters(configs: Path, corpus_list: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pretraining and evaluation waters with the centers of their bounding boxes."""
    corpus = pd.read_csv(configs / "pretraining_corpus.csv")
    targets = pd.read_csv(configs / "target_waters.csv")
    keep = {line.strip() for line in corpus_list.read_text().splitlines() if line.strip()}
    corpus = corpus[~corpus.water_id.isin(targets.hydrolakes_id) & corpus.water_id.isin(keep)]

    def with_center(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(lon=(frame.west + frame.east) / 2, lat=(frame.south + frame.north) / 2)

    return with_center(corpus), with_center(targets)


def draw_world_map(
    fig: Figure, corpus: pd.DataFrame, targets: pd.DataFrame, naturalearth: Path, map_width: float
) -> tuple[Axes, ccrs.Projection]:
    """World map in the cylindrical Miller projection, so graticule labels stay simple."""
    fig_w, fig_h = fig.get_size_inches()
    lon_min, lon_max, lat_min, lat_max = -135, 160, -50, 72
    proj = ccrs.Miller(central_longitude=10)
    # Height of the Miller extent relative to its width, so the axes box fits the map.
    y_lo = proj.transform_point(0, lat_min, ccrs.PlateCarree())[1]
    y_hi = proj.transform_point(0, lat_max, ccrs.PlateCarree())[1]
    x_lo = proj.transform_point(lon_min, 0, ccrs.PlateCarree())[0]
    x_hi = proj.transform_point(lon_max, 0, ccrs.PlateCarree())[0]
    map_h = map_width * fig_w / fig_h * ((y_hi - y_lo) / (x_hi - x_lo))
    ax = fig.add_axes([0.5 - map_width / 2, 1 - 0.02 - map_h, map_width, map_h], projection=proj)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_facecolor(OCEAN)
    ax.add_feature(
        ShapelyFeature(
            Reader(naturalearth / "ne_110m_land.shp").geometries(),
            ccrs.PlateCarree(),
            facecolor=LAND,
            edgecolor="none",
        ),
        zorder=0,
    )
    countries = set(targets.country)
    ax.add_feature(
        ShapelyFeature(
            [
                record.geometry
                for record in Reader(naturalearth / "ne_110m_admin_0_countries.shp").records()
                if record.attributes["NAME"] in countries
            ],
            ccrs.PlateCarree(),
            facecolor=LAND_HIGHLIGHT,
            edgecolor=INK,
            linewidth=0.3,
        ),
        zorder=1,
    )
    ax.add_feature(
        ShapelyFeature(
            Reader(naturalearth / "ne_110m_coastline.shp").geometries(),
            ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="#6a6a6a",
            linewidth=0.3,
        ),
        zorder=1,
    )
    gridlines = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        xlocs=range(-120, 181, 60),
        ylocs=range(-60, 91, 30),
        color=GRATICULE,
        linewidth=0.3,
        alpha=0.7,
        zorder=2,
    )
    gridlines.xlabel_style = {"size": 6, "color": INK}
    gridlines.ylabel_style = {"size": 6, "color": INK}
    gridlines.top_labels = False
    gridlines.right_labels = False
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.scatter(
        corpus.lon,
        corpus.lat,
        s=7,
        color=TRANSFER,
        edgecolors="white",
        linewidths=0.3,
        transform=ccrs.PlateCarree(),
        zorder=3,
        label="Pretraining waters",
    )
    ax.scatter(
        targets.lon,
        targets.lat,
        s=40,
        marker="o",
        facecolors="none",
        edgecolors=TARGET,
        linewidths=0.9,
        transform=ccrs.PlateCarree(),
        zorder=4,
        label="Evaluation waters",
    )
    ax.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        fontsize=6.5,
        handletextpad=0.4,
        borderaxespad=0.4,
        markerscale=1.2,
    )
    return ax, proj


def water_outline(maps: Path, water_id: str) -> np.ndarray:
    """Union of the valid masks of a water's field maps."""
    masks = [np.load(path, allow_pickle=True)["valid"].astype(bool) for path in sorted(maps.glob(f"{water_id}_*.npz"))]
    return np.logical_or.reduce(masks)


def square_window(outline: np.ndarray, pad: int) -> tuple[float, float, int]:
    """Center row, center column and side in pixels of a square around the water, ``pad`` beyond its extent."""
    rows = np.where(outline.any(axis=1))[0]
    cols = np.where(outline.any(axis=0))[0]
    top, bottom = rows[0] - pad, rows[-1] + pad
    left, right = cols[0] - pad, cols[-1] + pad
    return (top + bottom) / 2, (left + right) / 2, max(bottom - top, right - left)


def window_mask(outline: np.ndarray, top: float, left: float, side: int) -> np.ndarray:
    """Water mask of the displayed square, padded with land where it leaves the array."""
    height, width = outline.shape
    window = np.zeros((side, side), dtype=bool)
    row0, col0 = int(round(top)), int(round(left))
    inside = outline[max(row0, 0) : min(row0 + side, height), max(col0, 0) : min(col0 + side, width)]
    window[max(-row0, 0) : max(-row0, 0) + inside.shape[0], max(-col0, 0) : max(-col0, 0) + inside.shape[1]] = inside
    return window


def water_fraction(window: np.ndarray, x_from: float, y_from: float, x_to: float, y_to: float) -> float:
    """Share of water in a box of axes fractions (origin bottom left)."""
    height, width = window.shape
    block = window[int((1 - y_to) * height) : int((1 - y_from) * height), int(x_from * width) : int(x_to * width)]
    return block.mean() if block.size else 0.0


def label_corners(window: np.ndarray, name: str) -> tuple[str, str]:
    """Corners of the name and the scale bar that cover the least water, preferring the order listed."""
    name_w, name_h = min(0.7, 0.045 * len(name) + 0.06), 0.14
    bar_w, bar_h = 0.52, 0.17
    name_spots = {
        "tl": (0.03, 1 - name_h, 0.03 + name_w, 1.0),
        "tr": (1 - 0.03 - name_w, 1 - name_h, 0.97, 1.0),
        "bl": (0.03, 0.0, 0.03 + name_w, name_h),
    }
    bar_spots = {
        "bl": (0.03, 0.0, 0.03 + bar_w, bar_h),
        "br": (0.82 - bar_w, 0.0, 0.82, bar_h),
        "tl": (0.03, 1 - bar_h, 0.03 + bar_w, 1.0),
        "tr": (1 - 0.03 - bar_w, 1 - bar_h, 0.97, 1.0),
    }
    pairs = [(name_corner, bar_corner) for name_corner in name_spots for bar_corner in bar_spots]
    return min(
        (pair for pair in pairs if pair[0] != pair[1]),
        key=lambda pair: round(
            water_fraction(window, *name_spots[pair[0]]) + water_fraction(window, *bar_spots[pair[1]]), 2
        ),
    )


def nice_ticks(lo: float, hi: float) -> tuple[np.ndarray, float]:
    """Two round ticks inside the inner 85% of the range, at least 45% of the range apart.

    The coarsest step that gives such a pair wins, so labels stay round and legible.
    """
    span = hi - lo
    inner_lo, inner_hi = lo + 0.075 * span, hi - 0.075 * span
    fallback = None
    for step in sorted({m * 10.0**k for k in range(-4, 2) for m in (1, 2, 5)}, reverse=True):
        candidates = np.arange(np.ceil(inner_lo / step), np.floor(inner_hi / step) + 1) * step
        if len(candidates) >= 2:
            pair = np.array([candidates[0], candidates[-1]])
            if pair[1] - pair[0] >= 0.45 * span:
                return pair, step
            if fallback is None:
                fallback = (pair, step)
    return fallback if fallback is not None else (np.array([lo, hi]), span)


def deg_label(value: float, axis: str, decimals: int) -> str:
    hemisphere = ("E" if value >= 0 else "W") if axis == "lon" else ("N" if value >= 0 else "S")
    return f"{abs(value):.{decimals}f}°{hemisphere}"


def scale_bar(ax: Axes, lon0: float, lat0: float, lon_per_km: float, km: float, fontsize: float) -> None:
    """Black-and-white bar of ``km`` kilometres in two halves, with its lower-left corner at (lon0, lat0)."""
    half = km / 2 * lon_per_km
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.035
    ax.add_patch(Rectangle((lon0, lat0), half, h, facecolor="black", edgecolor="black", linewidth=0.4, zorder=6))
    ax.add_patch(Rectangle((lon0 + half, lat0), half, h, facecolor="white", edgecolor="black", linewidth=0.4, zorder=6))
    xr = ax.get_xlim()[1] - ax.get_xlim()[0]
    labels = [(0, lon0), (km / 2, lon0 + half), (km, lon0 + 2 * half)]
    if half / xr < 0.11:  # bar too short for three labels
        labels = [labels[0], labels[2]]
    for k, x in labels:
        ax.text(
            x, lat0 + 1.4 * h, f"{k:g}", ha="center", va="bottom", fontsize=fontsize, color=INK, zorder=6, clip_on=False
        )
    ax.text(
        lon0 + 2 * half + 0.045 * xr,
        lat0 + 1.4 * h,
        "km",
        ha="left",
        va="bottom",
        fontsize=fontsize,
        color=INK,
        zorder=6,
        clip_on=False,
    )


def draw_extent(pax: Axes, row: pd.Series, outline: np.ndarray) -> np.ndarray:
    """Water extent in a square window on a longitude-latitude grid; returns the water mask of the window."""
    height, width = outline.shape

    def lon_of(col):
        return row.west + (row.east - row.west) * col / width

    def lat_of(line):
        return row.north - (row.north - row.south) * line / height

    row_mid, col_mid, span = square_window(outline, PAD.get(row.water_id, 12))
    top, bottom = row_mid - span / 2, row_mid + span / 2
    left, right = col_mid - span / 2, col_mid + span / 2
    pax.set_facecolor(LAND_HIGHLIGHT)
    # Water as a vector polygon: contourf and contour of the mask at the same level
    # share their boundary, so the fill and the shoreline coincide exactly. The
    # padding closes the shoreline polygons at the array border.
    lons = np.concatenate([[lon_of(-0.5)], lon_of(np.arange(width) + 0.5), [lon_of(width + 0.5)]])
    lats = np.concatenate([[lat_of(-0.5)], lat_of(np.arange(height) + 0.5), [lat_of(height + 0.5)]])
    field = np.pad(outline.astype(float), 1)
    pax.contourf(lons, lats, field, levels=[0.5, 1.5], colors=[WATER], antialiased=True, zorder=2)
    pax.contour(lons, lats, field, levels=[0.5], colors=[WATER_EDGE], linewidths=0.4, zorder=3)
    pax.set_xlim(lon_of(left), lon_of(right))
    pax.set_ylim(lat_of(bottom), lat_of(top))
    xticks, xstep = nice_ticks(*pax.get_xlim())
    yticks, ystep = nice_ticks(*pax.get_ylim())
    xdecimals, ydecimals = max(0, int(-np.floor(np.log10(xstep)))), max(0, int(-np.floor(np.log10(ystep))))
    pax.set_xticks(xticks)
    pax.set_yticks(yticks)
    pax.set_xticklabels([deg_label(v, "lon", xdecimals) for v in xticks], fontsize=5.5)
    pax.set_yticklabels([deg_label(v, "lat", ydecimals) for v in yticks], fontsize=5.5, rotation=90, va="center")
    pax.tick_params(length=2, width=0.5, pad=1.5, top=True, right=True, labeltop=False, labelright=False)
    pax.grid(True, color=GRATICULE, linewidth=0.3, alpha=0.7, zorder=4)
    for spine in pax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor(INK)
    return window_mask(outline, top, left, int(round(span)))


def draw_scale_bar(pax: Axes, latitude: float, corner: str) -> None:
    """A 1 or 2 km scale bar in a corner of the panel."""
    lon_per_km = 1.0 / (111.32 * np.cos(np.deg2rad(latitude)))
    xlim, ylim = pax.get_xlim(), pax.get_ylim()
    km = 2 if (xlim[1] - xlim[0]) / lon_per_km > 4.5 else 1
    bar_len = km * lon_per_km
    xr = xlim[1] - xlim[0]
    yr = ylim[1] - ylim[0]
    bx = {
        "bl": xlim[0] + 0.05 * xr,
        "tl": xlim[0] + 0.05 * xr,
        "tr": xlim[1] - 0.05 * xr - bar_len,
        "bl_up": xlim[0] + 0.03 * xr,
        "br": xlim[1] - 0.30 * xr - bar_len,
    }[corner]
    by = {
        "bl": ylim[0] + 0.06 * yr,
        "br": ylim[0] + 0.06 * yr,
        "tl": ylim[1] - 0.15 * yr,
        "tr": ylim[1] - 0.15 * yr,
        "bl_up": ylim[0] + 0.125 * yr,
    }[corner]
    scale_bar(pax, bx, by, lon_per_km, km, fontsize=5.5)


def annotate_panel(pax: Axes, row: pd.Series, window: np.ndarray, number: int) -> None:
    """Name and scale bar in the corners with the least water, the number at the bottom right, facts below."""
    name = row["name"]
    name_corner, bar_corner = PLACEMENT.get(row.water_id, label_corners(window, name))
    nx, ny = (0.05, 0.95) if name_corner == "tl" else (0.95, 0.95) if name_corner == "tr" else (0.05, 0.05)
    pax.text(
        nx,
        ny,
        name,
        transform=pax.transAxes,
        ha="left" if name_corner in ("tl", "bl") else "right",
        va="top" if name_corner in ("tl", "tr") else "bottom",
        fontsize=6.5,
        fontweight="bold",
        color=INK,
        zorder=6,
    )
    draw_scale_bar(pax, row.lat, bar_corner)
    box = 0.13
    pax.add_patch(
        Rectangle(
            (1 - box, 0),
            box,
            box,
            transform=pax.transAxes,
            facecolor="black",
            edgecolor="none",
            zorder=6,
            clip_on=False,
        )
    )
    pax.text(
        1 - box / 2,
        box / 2,
        f"{number}",
        transform=pax.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        color="white",
        zorder=7,
    )
    area = AREA_OVERRIDE_KM2.get(row.water_id, row.area_km2)
    info = f"{area:.2f} km$^2$ \u00b7 {EVAL_SCENES[row.water_id]} scenes \u00b7 {WATER_TYPE[int(row.lake_type)]}"
    pax.text(0.5, -0.19, info, transform=pax.transAxes, ha="center", va="top", fontsize=6, color=INK)


def draw_wedges(fig: Figure, world: Axes, proj: ccrs.Projection, panels: list[tuple[pd.Series, Axes]]) -> None:
    """Wedges from each water on the world map to the top edge of its panel, on a transparent overlay."""
    overlay = fig.add_axes([0, 0, 1, 1], zorder=5)
    overlay.axis("off")
    overlay.patch.set_alpha(0)
    to_figure = fig.transFigure.inverted()
    for row, pax in panels:
        marker = world.transData.transform(proj.transform_point(row.lon, row.lat, ccrs.PlateCarree()))
        x, y = to_figure.transform(marker)
        box = pax.get_position()
        overlay.add_patch(
            Polygon(
                [[x, y], [box.x0, box.y1], [box.x1, box.y1]],
                closed=True,
                transform=fig.transFigure,
                facecolor=WEDGE,
                edgecolor="none",
                alpha=0.35,
            )
        )
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--naturalearth", type=Path, default=Path("data/naturalearth"))
    parser.add_argument("--maps", type=Path, default=Path("outputs/runs/field_maps"))
    parser.add_argument("--corpus-list", type=Path, default=Path("configs/pretraining_waters.txt"))
    parser.add_argument("--output", type=Path, required=True, help="Output path without extension")
    parser.add_argument("--figsize", type=float, nargs=2, default=(7.2, 4.3))
    parser.add_argument("--map-width", type=float, default=0.66, help="World-map width as a figure fraction")
    args = parser.parse_args()

    corpus, targets = read_waters(args.configs, args.corpus_list)
    print(f"{len(corpus)} pretraining waters, {len(targets)} evaluation waters")

    use_style(axes=False)
    fig = plt.figure(figsize=tuple(args.figsize))
    fig.patch.set_facecolor("white")
    world, proj = draw_world_map(fig, corpus, targets, args.naturalearth, args.map_width)

    # Square panels in one row, placed from west to east so the wedges do not cross.
    fig_w, fig_h = args.figsize
    gap = 0.04
    panel_w = (0.91 - (len(targets) - 1) * gap) / len(targets)
    panels = []
    for slot, index in enumerate(targets.sort_values("lon").index):
        row = targets.loc[index]
        pax = fig.add_axes([0.05 + slot * (panel_w + gap), 0.135, panel_w, panel_w * fig_w / fig_h])
        window = draw_extent(pax, row, water_outline(args.maps, row.water_id))
        annotate_panel(pax, row, window, slot + 1)
        panels.append((row, pax))
    draw_wedges(fig, world, proj, panels)
    save(fig, args.output)


if __name__ == "__main__":
    main()
