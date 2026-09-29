"""Write the paper's tables from an evaluation run (``evaluation.paper_tables``).

    python -m chla_prediction.experiments.make_paper_tables outputs/runs/evaluation_transfer \\
        --output-dir outputs/runs/paper
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chla_prediction.evaluation.paper_tables import make_paper_tables


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="An evaluate_forecast run directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    make_paper_tables(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
