"""Build the generalization-gap table from a scored plan directory (``evaluation.generalization_gap``)."""

from __future__ import annotations

import argparse
from pathlib import Path

from chla_prediction.evaluation.generalization_gap import summarize_generalization_gap


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path, help="Directory holding plan.json and the scored tasks")
    summarize_generalization_gap(parser.parse_args(argv).stage)


if __name__ == "__main__":
    main()
