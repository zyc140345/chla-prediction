"""Run one task of a diagnostics plan (``evaluation.diagnostics.run_task``)."""

from __future__ import annotations

import argparse
from pathlib import Path

from chla_prediction.evaluation.diagnostics import run_task
from chla_prediction.io import read_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args(argv)
    run_task(read_jsonl(args.plan)[args.index])


if __name__ == "__main__":
    main()
