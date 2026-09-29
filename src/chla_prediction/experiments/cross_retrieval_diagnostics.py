"""Run one stage of the cross-retrieval study (``evaluation.cross_retrieval``)."""

from __future__ import annotations

import argparse
from pathlib import Path

from chla_prediction.evaluation.cross_retrieval import audit, finish, select
from chla_prediction.io import read_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["audit", "select", "finish"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    {"audit": audit, "select": select, "finish": finish}[args.mode](read_jsonl(args.plan), args.output_dir)


if __name__ == "__main__":
    main()
