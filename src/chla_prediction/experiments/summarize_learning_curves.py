"""Write the data-amount learning-curve tables of an experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from chla_prediction.evaluation.learning_curve import summarize_learning_curves


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Experiment output directory, e.g. outputs/runs")
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    evaluation_dir = args.evaluation_dir or args.root / "evaluation_learning_curve"
    output_dir = args.output_dir or args.root / "paper"
    summarize_learning_curves(args.root, evaluation_dir, output_dir)


if __name__ == "__main__":
    main()
