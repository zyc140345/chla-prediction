"""Train one forecaster config: pretraining, few-shot transfer, or training from scratch (``training.train``)."""

from __future__ import annotations

import argparse
from pathlib import Path

from chla_prediction.training.train import train_forecaster


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    output = train_forecaster(args.config)

    print(f"Wrote training manifest: {output.manifest_path}")
    print(f"Sample counts: {output.sample_counts}")


if __name__ == "__main__":
    main()
