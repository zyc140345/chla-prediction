"""Score prediction manifests inside the water-extent masks (``evaluation.evaluate``)."""

from __future__ import annotations

import argparse
from pathlib import Path

from chla_prediction.evaluation.evaluate import run_evaluation


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    output = run_evaluation(args.config)

    print(f"Wrote evaluation tables under: {output.run_dir}")


if __name__ == "__main__":
    main()
