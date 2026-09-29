"""Train the pix2pix baseline on the pretraining corpus (``training.pix2pix``)."""

from __future__ import annotations

import argparse
from pathlib import Path

from chla_prediction.training.pix2pix import train_pix2pix


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    train_pix2pix(args.config, args.resume)


if __name__ == "__main__":
    main()
