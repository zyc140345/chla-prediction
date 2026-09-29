"""Score one baseline on one water of a generalization-gap plan (``evaluation.generalization_gap``)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chla_prediction.evaluation.generalization_gap import score_task


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    score_task(plan["tasks"][args.index], plan)


if __name__ == "__main__":
    main()
