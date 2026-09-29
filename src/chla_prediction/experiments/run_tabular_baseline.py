"""Fit and run the per-pixel random forest or XGBoost baseline (``baselines.tabular``)."""

from __future__ import annotations

import argparse
from pathlib import Path

from chla_prediction.baselines.tabular import run_tabular_baseline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    output = run_tabular_baseline(args.config)

    print(f"Wrote prediction manifest: {output.manifest_path} ({output.prediction_count} predictions)")


if __name__ == "__main__":
    main()
