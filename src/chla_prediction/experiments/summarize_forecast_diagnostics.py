"""Aggregate the learning-curve checkpoint scores of a diagnostics plan into ``curve_*`` tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from chla_prediction.evaluation.diagnostics_summary import summarize_diagnostics
from chla_prediction.io import read_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize_diagnostics(read_jsonl(args.plan), args.output_dir, "curve")


if __name__ == "__main__":
    main()
