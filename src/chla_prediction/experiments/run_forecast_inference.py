"""Forecast every sample of one split on the full scene (``inference.predict``)."""

from __future__ import annotations

import argparse
from pathlib import Path

from chla_prediction.inference.predict import run_inference


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    output = run_inference(args.config)

    print(f"Wrote prediction manifest: {output.manifest_path} ({output.prediction_count} predictions)")


if __name__ == "__main__":
    main()
