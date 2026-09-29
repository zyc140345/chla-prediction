"""Fetch water-extent layers onto each evaluation water's Sentinel-2 grid.

One GeoTIFF per water, ``<water_id>_gsw.tif``, stacks JRC Global Surface
Water (Pekel et al. 2016), ESA WorldCover water, Sentinel-1 water frequencies
and the Dynamic World water frequency over the training period. The
evaluation thresholds ``s1_median_water``; the scene classification (SCL) is
not used because it mislabels turbid and algal water.

The imagery and the Earth Engine credentials may live on different hosts, so
this runs in two steps:

    # where the imagery is
    uv run python tools/data/fetch_water_extent.py dump-grids configs/paper/evaluate.yaml grids.json
    # where the credentials are, after copying grids.json
    uv run python tools/data/fetch_water_extent.py fetch grids.json data/water_masks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import rasterio
import yaml

from chla_prediction.imagery.archive import read_bands
from chla_prediction.io import read_jsonl
from chla_prediction.recipe import SEQUENCE

GSW = "JRC/GSW1_4/GlobalSurfaceWater"
GSW_BANDS = ["occurrence", "seasonality", "recurrence", "max_extent"]
# ESA WorldCover v200 (2021, 10 m); class 80 is permanent water.
WORLDCOVER = "ESA/WorldCover/v200"
WATER_CLASS = 80
# Sentinel-1 VV below these thresholds counts as water. Open water is dark at
# C band whatever grows in it. The share of scenes below -17.5 dB is the rule
# "per-pixel median below -17.5 dB" (a true median exceeds the Earth Engine
# memory limit); -16 dB lies on the upper tail of water, where wind pushes it.
S1 = "COPERNICUS/S1_GRD"
S1_WATER_DB = -16.0
S1_MEDIAN_DB = -17.5
DYNAMIC_WORLD = "GOOGLE/DYNAMICWORLD/V1"
DW_WATER_LABEL = 0
BANDS = [*GSW_BANDS, "worldcover_water", "s1_water_pct", "s1_median_water", "dw_water_pct"]


def _bounds(grid: dict) -> list[float]:
    """Raster bounds in the water's own CRS, from its affine transform."""
    a, _, west, _, e, north = grid["transform"]
    return [west, north + e * grid["height"], west + a * grid["width"], north]


def _frequency(collection, is_water, name: str):
    """Percentage of the collection's images in which a pixel is water."""
    return collection.map(is_water).mean().multiply(100).rename(name).unmask(0)


def dump_grids(config_path: Path, destination: Path) -> None:
    """Record the CRS, affine transform and shape of every water of an evaluation config."""
    grids = []
    for entry in yaml.safe_load(config_path.read_text())["data"]["archives"]:
        manifest = Path(entry["manifest_path"])
        with rasterio.open(manifest.parent / read_jsonl(manifest)[0]["scl_path"]) as src:
            grids.append(
                {
                    "water_id": entry["water_id"],
                    "crs": str(src.crs),
                    "transform": list(src.transform)[:6],
                    "width": src.width,
                    "height": src.height,
                }
            )
    destination.write_text(json.dumps(grids, indent=2), encoding="utf-8")
    print(f"wrote {len(grids)} water grids to {destination}")


def fetch(grids_path: Path, output_dir: Path, start: str, train_end: str) -> None:
    """Download the band stack onto each water grid, matching its Sentinel-2 rasters exactly."""
    # Earth Engine is needed only here, so dump-grids runs on hosts without it.
    import ee
    import requests
    from _gee import initialize

    initialize()
    output_dir.mkdir(parents=True, exist_ok=True)
    for grid in json.loads(grids_path.read_text(encoding="utf-8")):
        region = ee.Geometry.Rectangle(_bounds(grid), proj=grid["crs"], evenOdd=False)
        sar = (
            ee.ImageCollection(S1)
            .filterBounds(region)
            .filterDate(start, train_end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .select("VV")
        )
        dynamic = ee.ImageCollection(DYNAMIC_WORLD).filterBounds(region).filterDate(start, train_end).select("label")

        # GSW (30 m) is resampled nearest-neighbor onto the 10 m grid; every
        # band is a percentage, so the stack fits uint8.
        stack = (
            ee.Image(GSW)
            .select(GSW_BANDS)
            .unmask(0)
            .addBands(
                [
                    ee.ImageCollection(WORLDCOVER).first().eq(WATER_CLASS).multiply(100).unmask(0),
                    _frequency(sar, lambda image: image.lt(S1_WATER_DB), "s1_water_pct"),
                    _frequency(sar, lambda image: image.lt(S1_MEDIAN_DB), "s1_median_water"),
                    _frequency(dynamic, lambda image: image.eq(DW_WATER_LABEL), "dw_water_pct"),
                ]
            )
            .toUint8()
        )
        url = stack.getDownloadURL(
            {
                "format": "GEO_TIFF",
                "crs": grid["crs"],
                "crs_transform": grid["transform"],
                "dimensions": f"{grid['width']}x{grid['height']}",
            }
        )
        response = requests.get(url, timeout=600)
        response.raise_for_status()
        destination = output_dir / f"{grid['water_id']}_gsw.tif"
        destination.write_bytes(response.content)
        # Earth Engine exports unnamed bands; readers select them by description.
        with rasterio.open(destination, "r+") as dst:
            dst.descriptions = tuple(BANDS)
            if (str(dst.crs), dst.width, dst.height) != (grid["crs"], grid["width"], grid["height"]):
                raise ValueError(f"{grid['water_id']}: water grid {dst.crs} {dst.width}x{dst.height} != scene grid")
        recurrence, worldcover, s1_water, s1_median, dw_water = read_bands(
            destination, ["recurrence", "worldcover_water", "s1_water_pct", "s1_median_water", "dw_water_pct"]
        )
        print(
            f"{grid['water_id']}: recur>=90 {int((recurrence >= 90).sum()):6d}  "
            f"worldcover {int((worldcover >= 50).sum()):6d}  "
            f"s1>=75 {int((s1_water >= 75).sum()):6d} ({sar.size().getInfo()} scenes)  "
            f"dw>=75 {int((dw_water >= 75).sum()):6d} ({dynamic.size().getInfo()} scenes)  "
            f"s1med>=50 {int((s1_median >= 50).sum()):6d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    dump = sub.add_parser("dump-grids", help="record the water grids (run where the imagery is)")
    dump.add_argument("config", type=Path)
    dump.add_argument("destination", type=Path)
    get = sub.add_parser("fetch", help="download the layers onto those grids (run where the credentials are)")
    get.add_argument("grids", type=Path)
    get.add_argument("output_dir", type=Path)
    get.add_argument("--start", default="2019-01-01")
    get.add_argument("--train-end", default=SEQUENCE["train_end"], help="End of the period the layers are fitted on")
    args = parser.parse_args()
    if args.command == "dump-grids":
        dump_grids(args.config, args.destination)
    else:
        fetch(args.grids, args.output_dir, args.start, args.train_end)


if __name__ == "__main__":
    main()
