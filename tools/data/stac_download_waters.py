# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pystac-client>=0.8",
#   "planetary-computer>=1.0",
#   "rasterio>=1.3",
#   "numpy>=1.26",
# ]
# ///
"""Download Sentinel-2 L2A chip time series for a list of waters.

Reads a water CSV (``water_id``, ``west``, ``south``, ``east``, ``north``) and
pulls every usable scene per water from the Planetary Computer STAC catalog,
parallelized across waters. Chips are uint16 DN with scale 1e-4 and the BOA
offset harmonized, with per-scene SCL rasters and one resumable JSONL
manifest per water, in ``<out>/<water_id>/``; the manifest lists its files
relative to that directory. Chips carry B1-B12 (B1 resampled from 60 m) so
the MDN can be applied.

    uv run --script tools/data/stac_download_waters.py --waters configs/pretraining_corpus.csv \\
        --water-ids configs/pretraining_waters.txt --output-dir data/corpus
    uv run --script tools/data/stac_download_waters.py \\
        --waters configs/target_waters.csv --output-dir data/evaluation
"""

import argparse
import csv
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

# Remote COG reads over high-latency links: skip sibling-file probing on
# open, ingest a large header chunk in one request, and retry transient
# HTTP failures. Must be set before rasterio/GDAL initialize.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_INGESTED_BYTES_AT_OPEN", "32768")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "5")

import numpy as np  # noqa: E402
import planetary_computer  # noqa: E402
import pystac_client  # noqa: E402
import rasterio  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.vrt import WarpedVRT  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
# B01 (60 m) comes last; every archive of the corpus stores the bands in this order.
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "B01"]
OUT_BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "B1"]
SCL_CLOUDY = (3, 8, 9, 10)
BOA_OFFSET = 1000
RESOLUTION = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--waters",
        type=Path,
        required=True,
        help="Water CSV (select_pretraining_waters.py or configs/target_waters.csv)",
    )
    parser.add_argument(
        "--water-ids", type=Path, default=None, help="Only these waters: a file of whitespace-separated water ids"
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="One subdirectory per water")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--max-cloud-cover", type=float, default=70.0, help="Tile eo:cloud_cover limit (percent)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit-waters", type=int, default=None, help="Only the first N waters (smoke tests)")
    parser.add_argument("--limit-scenes", type=int, default=None, help="At most N scenes per water (smoke tests)")
    parser.add_argument(
        "--shard",
        default="0/1",
        help="I/K: only scene dates with YYYYMMDD %% K == I, so K processes can share a water "
        "(the manifest is append-only and resume reads it at water start)",
    )
    return parser.parse_args()


def read_band(href: str, vrt_options: dict, resampling: Resampling) -> np.ndarray:
    for attempt in range(3):
        try:
            with rasterio.open(href) as src, WarpedVRT(src, resampling=resampling, **vrt_options) as vrt:
                return vrt.read(1)
        except rasterio.errors.RasterioIOError:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def target_grid(item, roi: tuple[float, float, float, float]) -> dict:
    code = item.properties.get("proj:code")
    epsg = code.removeprefix("EPSG:") if code else item.properties["proj:epsg"]
    dst_crs = f"EPSG:{epsg}"
    west, south, east, north = transform_bounds("EPSG:4326", dst_crs, *roi)
    west = np.floor(west / RESOLUTION) * RESOLUTION
    south = np.floor(south / RESOLUTION) * RESOLUTION
    east = np.ceil(east / RESOLUTION) * RESOLUTION
    north = np.ceil(north / RESOLUTION) * RESOLUTION
    transform = rasterio.transform.from_origin(west, north, RESOLUTION, RESOLUTION)
    return {
        "crs": dst_crs,
        "transform": transform,
        "width": round((east - west) / RESOLUTION),
        "height": round((north - south) / RESOLUTION),
    }


def pick_item(items: list, roi: tuple[float, float, float, float]):
    def covers(item) -> bool:
        w, s, e, n = item.bbox
        return w <= roi[0] and s <= roi[1] and e >= roi[2] and n >= roi[3]

    def key(item):
        return (covers(item), -item.properties.get("eo:cloud_cover", 100.0), item.id)

    return sorted(items, key=key)[-1]


def download_water(water: dict, args: argparse.Namespace) -> str:
    water_id = water["water_id"]
    roi = (float(water["west"]), float(water["south"]), float(water["east"]), float(water["north"]))
    out_dir = args.output_dir / water_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "stac_download_manifest.jsonl"
    done = set()
    if manifest_path.exists():
        done = {json.loads(line)["scene_date"] for line in manifest_path.read_text().splitlines()}

    # Items are signed per scene right before reading (SAS tokens expire
    # after about an hour; a water with hundreds of scenes runs longer).
    catalog = pystac_client.Client.open(STAC_URL)
    # The STAC API rate-limits bursts of searches (many workers starting at
    # once) and occasionally drops connections; back off and retry instead
    # of failing the whole water.
    for attempt in range(6):
        try:
            search = catalog.search(collections=[COLLECTION], bbox=roi, datetime=f"{args.start}/{args.end}")
            items = list(search.items())
            break
        except Exception:  # noqa: BLE001 - APIError on 429, dropped connections; both transient
            if attempt == 5:
                raise
            time.sleep(30 * (attempt + 1) + random.uniform(0, 10))
    by_date: dict[str, list] = {}
    for item in items:
        by_date.setdefault(item.datetime.strftime("%Y%m%d"), []).append(item)

    # Shuffle deterministically per water so an interrupted or time-boxed run
    # leaves uniform coverage of the whole date range instead of only the
    # earliest years.
    ordered = sorted(by_date)
    random.Random(water_id).shuffle(ordered)
    shard_index, shard_count = args.shard
    ordered = [d for d in ordered if int(d) % shard_count == shard_index]
    n_written = n_failed = 0
    for scene_date in ordered:
        if scene_date in done:
            continue
        if args.limit_scenes is not None and n_written >= args.limit_scenes:
            break
        item = pick_item(by_date[scene_date], roi)
        cloud = item.properties.get("eo:cloud_cover")
        if cloud is not None and cloud > args.max_cloud_cover:
            continue
        scene_start = time.time()
        try:
            write_scene(planetary_computer.sign(item), water_id, scene_date, roi, out_dir, manifest_path, cloud)
        except Exception as exc:  # noqa: BLE001 - one bad scene must not abort the water; a rerun fills the gap
            n_failed += 1
            print(f"{water_id} {scene_date} FAILED {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            continue
        n_written += 1
        print(f"{water_id} {scene_date} in {time.time() - scene_start:.0f}s", flush=True)
    return f"{water_id}: {n_written} new scenes ({len(done)} resumed, {n_failed} failed)"


def write_scene(item, water_id: str, scene_date: str, roi, out_dir: Path, manifest_path: Path, cloud) -> None:
    grid = target_grid(item, roi)
    vrt_options = {"crs": grid["crs"], "transform": grid["transform"], "width": grid["width"], "height": grid["height"]}
    baseline = item.properties.get("s2:processing_baseline", "")
    harmonize = baseline >= "04.00"

    stack = np.zeros((len(BANDS), grid["height"], grid["width"]), np.uint16)
    for i, band in enumerate(BANDS):
        data = read_band(item.assets[band].href, vrt_options, Resampling.bilinear)
        if harmonize:
            data = np.where(data > BOA_OFFSET, data - BOA_OFFSET, np.where(data > 0, 1, 0)).astype(np.uint16)
        stack[i] = data
    scl = read_band(item.assets["SCL"].href, vrt_options, Resampling.nearest).astype(np.uint8)
    roi_cloud = float(np.isin(scl, SCL_CLOUDY).mean())

    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "count": len(BANDS),
        "crs": grid["crs"],
        "transform": grid["transform"],
        "width": grid["width"],
        "height": grid["height"],
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "nodata": 0,
    }
    refl_path = out_dir / f"{water_id}_s2_{scene_date}_{item.id}.tif"
    with rasterio.open(refl_path, "w", **profile) as dst:
        dst.write(stack)
        for i, name in enumerate(OUT_BAND_NAMES, start=1):
            dst.set_band_description(i, name)
    scl_path = out_dir / f"{water_id}_s2_{scene_date}_{item.id}_SCL.tif"
    scl_profile = profile | {"dtype": "uint8", "count": 1, "predictor": 1, "nodata": None}
    with rasterio.open(scl_path, "w", **scl_profile) as dst:
        dst.write(scl, 1)

    row = {
        "scene_date": scene_date,
        "item_id": item.id,
        "water_id": water_id,
        "sensor": "sentinel2_l2a",
        "source": "planetary_computer",
        "bands": OUT_BAND_NAMES,
        "crs": grid["crs"],
        "resolution_m": RESOLUTION,
        "processing_baseline": baseline,
        "boa_offset_subtracted": harmonize,
        "scene_cloud_cover": cloud,
        "roi_cloud_shadow_fraction": round(roi_cloud, 4),
        "is_product_date": False,
        "local_path": refl_path.name,
        "scl_path": scl_path.name,
        "created_at": date.today().isoformat(),
    }
    with manifest_path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    with args.waters.open() as stream:
        waters = list(csv.DictReader(stream))
    if args.water_ids is not None:
        keep = set(args.water_ids.read_text().split())
        waters = [water for water in waters if water["water_id"] in keep]
    if args.limit_waters is not None:
        waters = waters[: args.limit_waters]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.shard = tuple(int(x) for x in args.shard.split("/"))
    print(f"{len(waters)} waters, {args.workers} workers")
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_water, water, args): water["water_id"] for water in waters}
        for future in as_completed(futures):
            # One failing water must not stop the others; the run still fails at the end.
            try:
                print(future.result(), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"{futures[future]}: FAILED {exc}", flush=True)
                failed.append(futures[future])
    if failed:
        raise SystemExit(f"{len(failed)} waters failed: {', '.join(sorted(failed))}")
    print("all waters done")


if __name__ == "__main__":
    main()
