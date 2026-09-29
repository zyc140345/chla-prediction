"""Write each water's eight-day pseudo-label climatology over the training period."""

from __future__ import annotations

import argparse
from pathlib import Path

from chla_prediction.config import PrecomputeClimatologyConfig
from chla_prediction.imagery.climatology import climatology_path, write_climatology
from chla_prediction.imagery.waters import load_waters


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config = PrecomputeClimatologyConfig.from_yaml(args.config)
    archives, _ = load_waters(config.data)
    for archive in archives:
        path = climatology_path(archive, config.product)
        write_climatology(archive, config.product, config.data.sequence.train_end, path)
        print(f"{archive.water_id}: {path}", flush=True)


if __name__ == "__main__":
    main()
