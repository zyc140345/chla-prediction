"""Merge one variant's per-water prediction manifests into the file an evaluation reads.

    python -m chla_prediction.experiments.merge_predictions \\
        --output outputs/runs/merged_ours_adapted.jsonl \\
        outputs/runs/infer_ours_{baogu,loweswater,georges,wentzel,hushan}/prediction_manifest.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chla_prediction.inference.manifest import merge_manifests


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(f"{args.output}: {merge_manifests(args.manifests, args.output)} rows")


if __name__ == "__main__":
    main()
