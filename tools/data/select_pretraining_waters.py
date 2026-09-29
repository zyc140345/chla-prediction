# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "geopandas>=1.0",
#   "numpy>=1.26",
# ]
# ///
"""Select the candidate waters of the pretraining corpus from HydroLAKES.

Filters to small-to-medium water bodies in the tropical-to-warm-temperate
band and samples two tiers: a China tier of southern reservoirs (the regime
of Hushan and Baogu) and a globally stratified tier (continent x
area bin). Water bounding boxes are padded to a minimum chip extent so every
downloaded scene supports the training crop size. The drawn candidates are
``configs/pretraining_corpus.csv``.

Evaluation waters drawn as candidates are downloaded to ``data/evaluation``
and never enter the corpus directory. Hushan has no HydroLAKES id, so its
neighborhood is excluded here.
"""

import argparse
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

MIN_CHIP_KM = 2.6  # >= 256 px at 10 m after padding
KM_PER_DEG_LAT = 111.0
AREA_BINS = [0.5, 2.0, 10.0, 50.0]

# (lon, lat, radius_deg, water) around evaluation waters without a HydroLAKES id.
EXCLUSION_ZONES = [
    (110.72, 19.94, 0.5, "hushan"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hydrolakes", type=Path, required=True, help="Path to HydroLAKES_polys_v10.shp")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV")
    parser.add_argument("--china-count", type=int, default=123)
    parser.add_argument("--global-count", type=int, default=337)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def padded_bbox(bounds: tuple[float, float, float, float], lat: float) -> tuple[float, float, float, float]:
    west, south, east, north = bounds
    min_lat_deg = MIN_CHIP_KM / KM_PER_DEG_LAT
    min_lon_deg = MIN_CHIP_KM / (KM_PER_DEG_LAT * max(0.2, math.cos(math.radians(lat))))
    pad_lat = max(0.003, (min_lat_deg - (north - south)) / 2)
    pad_lon = max(0.003, (min_lon_deg - (east - west)) / 2)
    return west - pad_lon, south - pad_lat, east + pad_lon, north + pad_lat


def stratified_sample(frame: gpd.GeoDataFrame, keys: list[str], count: int, rng: np.random.Generator):
    groups = list(frame.groupby(keys, sort=True))
    quota = {name: max(1, round(count * len(group) / len(frame))) for name, group in groups}
    picked = []
    for name, group in groups:
        take = min(quota[name], len(group))
        picked.append(group.sample(n=take, random_state=int(rng.integers(0, 2**31))))
    sampled = pd.concat(picked)
    if len(sampled) > count:
        sampled = sampled.sample(n=count, random_state=int(rng.integers(0, 2**31)))
    return sampled


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    columns = ["Hylak_id", "Lake_name", "Country", "Continent", "Lake_type", "Lake_area", "Pour_long", "Pour_lat"]
    waters = gpd.read_file(
        args.hydrolakes,
        columns=columns,
        where="Lake_area >= 0.5 AND Lake_area <= 50",
    )
    waters = waters[(waters.Pour_lat > -35.0) & (waters.Pour_lat < 45.0)]
    for lon, lat, radius, _name in EXCLUSION_ZONES:
        distance = np.hypot(waters.Pour_long - lon, waters.Pour_lat - lat)
        waters = waters[distance > radius]
    print(f"{len(waters)} candidate waters after filters")

    china = waters[(waters.Country == "China") & (waters.Pour_lat < 35.0)]
    china_reservoirs = china[china.Lake_type.isin([2, 3])]
    china_pool = china_reservoirs if len(china_reservoirs) >= args.china_count else china
    china_pool = china_pool.assign(area_bin=pd.cut(china_pool.Lake_area, AREA_BINS))
    china_pick = stratified_sample(china_pool, ["area_bin"], args.china_count, rng)

    world = waters[~waters.index.isin(china_pick.index)]
    world = world.assign(area_bin=pd.cut(world.Lake_area, AREA_BINS))
    world_pick = stratified_sample(world, ["Continent", "area_bin"], args.global_count, rng)

    picked = pd.concat([china_pick.assign(tier="china"), world_pick.assign(tier="global")])
    rows = []
    for _, row in picked.iterrows():
        west, south, east, north = padded_bbox(row.geometry.bounds, row.Pour_lat)
        rows.append(
            {
                "water_id": f"hylak_{int(row.Hylak_id)}",
                "name": row.Lake_name or "",
                "country": row.Country,
                "tier": row.tier,
                "lake_type": int(row.Lake_type),
                "area_km2": round(float(row.Lake_area), 3),
                "west": round(west, 6),
                "south": round(south, 6),
                "east": round(east, 6),
                "north": round(north, 6),
            }
        )
    table = pd.DataFrame(rows).sort_values("water_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(
        f"wrote {len(table)} waters ({(table.tier == 'china').sum()} china / {(table.tier == 'global').sum()} global)"
    )
    print(table.groupby("tier").area_km2.describe().round(2))


if __name__ == "__main__":
    main()
