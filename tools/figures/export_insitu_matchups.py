"""Point-level check of the MDN against in-situ Chl-a measurements.

Each in-situ sample (date, lon, lat, chla in mg/m^3) is matched to the
nearest clear archive scene within ``--max-days``; the MDN Chl-a at the
3x3 pixel window around the sample is compared with the measurement.
Reports RMSE_log, median ratio, Pearson r in log space and MAE, for all
matchups and for same-day matchups only. Optionally the same statistics are
computed for third-party product rasters (``<products-dir>/chl_<date>.tif``,
any CRS) to compare an operational retrieval on the same points.

    uv run python tools/figures/export_insitu_matchups.py \\
        --manifest data/evaluation/hushan/stac_download_manifest.jsonl \\
        --insitu data/insitu/hushan_insitu_chla.csv \\
        --mdn-weights data/models/mdn/mdn_msi_chl.npz \\
        --output outputs/runs/paper/mdn_insitu_matchups.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject, transform

from chla_prediction.evaluation.scoring import rmse_log
from chla_prediction.imagery.archive import SceneArchive, load_reflectance
from chla_prediction.retrieval.chla import ChlaRetrieval, log10_chla
from chla_prediction.retrieval.mdn import MDN_BANDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--water-id", default="hushan")
    parser.add_argument("--insitu", type=Path, required=True, help="CSV with date (YYYYMMDD), lon, lat, chla")
    parser.add_argument("--mdn-weights", type=Path, required=True)
    parser.add_argument("--products-dir", type=Path, default=None)
    parser.add_argument("--max-days", type=int, default=2)
    parser.add_argument("--max-cloud-fraction", type=float, default=0.2, help="Scene cloud fraction limit (0-1)")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV with one row per matchup")
    return parser.parse_args()


def _window_median(field: np.ndarray | None, window: tuple[slice, slice]) -> float:
    """Median of a sample window; NaN when the field misses it entirely."""
    if field is None or not np.isfinite(field[window]).any():
        return float("nan")
    return float(np.nanmedian(field[window]))


def _stats(label: str, estimate: np.ndarray, insitu: np.ndarray) -> None:
    ok = np.isfinite(estimate) & np.isfinite(insitu)
    if ok.sum() < 3:
        print(f"{label}: too few matchups ({ok.sum()})")
        return
    est, obs = estimate[ok], insitu[ok]
    log_est, log_obs = log10_chla(est), log10_chla(obs)
    print(
        f"{label}: N={ok.sum()} log10-RMSE={rmse_log(est, obs):.3f} "
        f"median-ratio={np.median(est / obs):.2f} pearson(log)={np.corrcoef(log_est, log_obs)[0, 1]:.3f} "
        f"MAE={np.mean(np.abs(est - obs)):.1f} mg/m3 (in-situ median {np.median(obs):.1f})"
    )


def main() -> None:
    args = parse_args()
    retrieve = ChlaRetrieval(args.mdn_weights)
    archive = SceneArchive.from_stac_manifest(args.manifest, args.water_id)
    points = pd.read_csv(args.insitu, dtype={"date": str})
    rows = []
    for date_str, group in points.groupby("date"):
        day = datetime.strptime(date_str, "%Y%m%d").date()
        candidates = [
            (abs((s.scene_date - day).days), s)
            for s in archive.scenes
            if abs((s.scene_date - day).days) <= args.max_days and s.cloud_fraction <= args.max_cloud_fraction
        ]
        if not candidates:
            print(f"{date_str}: no clear scene within {args.max_days} days")
            continue
        scene = min(candidates, key=lambda c: c[0])[1]
        reflectance, valid = load_reflectance(scene, MDN_BANDS)
        chla = retrieve("mdn_chla", reflectance, MDN_BANDS, valid)
        with rasterio.open(scene.path) as src:
            profile = src.profile
            xs, ys = transform("EPSG:4326", src.crs, group.lon.values, group.lat.values)
            pixels = [src.index(x, y) for x, y in zip(xs, ys, strict=True)]
        product = None
        if args.products_dir is not None:
            product_path = args.products_dir / f"chl_{scene.scene_date:%Y%m%d}.tif"
            if product_path.exists():
                with rasterio.open(product_path) as src:
                    product = np.full((profile["height"], profile["width"]), np.nan, np.float32)
                    reproject(
                        src.read(1),
                        product,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=profile["transform"],
                        dst_crs=profile["crs"],
                        resampling=Resampling.bilinear,
                        dst_nodata=np.nan,
                    )
        for (row, col), (_, point) in zip(pixels, group.iterrows(), strict=True):
            window = (slice(max(row - 1, 0), row + 2), slice(max(col - 1, 0), col + 2))
            rows.append(
                {
                    "date": date_str,
                    "sample": point.get("sample", ""),
                    "scene_date": scene.scene_date.strftime("%Y%m%d"),
                    "days_offset": int((scene.scene_date - day).days),
                    "insitu": point.chla,
                    "mdn": _window_median(chla, window),
                    "product": _window_median(product, window),
                }
            )
    table = pd.DataFrame(rows)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
    same_day = table.date == table.scene_date
    _stats("MDN, all matchups", table.mdn.values, table.insitu.values)
    _stats("MDN, same-day", table.mdn[same_day].values, table.insitu[same_day].values)
    if args.products_dir is not None:
        _stats("product, all matchups", table["product"].values, table.insitu.values)
        _stats("product, same-day", table["product"][same_day].values, table.insitu[same_day].values)


if __name__ == "__main__":
    main()
