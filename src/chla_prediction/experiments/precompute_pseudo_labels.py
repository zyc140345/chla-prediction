"""Attach the MDN log10 Chl-a pseudo-label raster to every archive scene (``retrieval.precompute``)."""

from __future__ import annotations

import argparse
from pathlib import Path

from chla_prediction.retrieval.precompute import precompute_pseudo_labels


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    output = precompute_pseudo_labels(args.config)

    print(f"Wrote {output.n_written} pseudo-label rasters ({output.n_skipped} already present)")


if __name__ == "__main__":
    main()
