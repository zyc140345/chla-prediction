r"""Parameter counts of the encoders and temporal modules in the design-choices figure.

Builds each variant from its pretraining config and counts the parameters of
its encoder or temporal module, total and trainable. Weights are not loaded:
the MDN initialization and the Clay checkpoint change values, not counts.

Usage:
    python tools/figures/count_parameters.py --configs configs/generated \
        --output outputs/runs/parameter_counts/module_parameters.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from _style import DESIGN_CHOICES

from chla_prediction.config import TrainForecasterConfig
from chla_prediction.models.registry import build_model

MODULES = {"Encoder": "encoder", "Temporal module": "temporal"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", type=Path, default=Path("configs/generated"))
    parser.add_argument("--output", type=Path, required=True, help="CSV to write")
    args = parser.parse_args()
    rows = []
    for group, variants in DESIGN_CHOICES:
        module_name = MODULES[group]
        for variant, label in variants:
            config = TrainForecasterConfig.from_yaml(args.configs / f"pretrain_{variant}.yaml")
            model_config = config.model.model_copy(update={"mdn_weights": None, "clay_checkpoint": None})
            module = getattr(build_model(model_config, config.data), module_name)
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            rows.append(
                {
                    "group": group,
                    "variant": variant,
                    "label": label,
                    "module": module_name,
                    "total_parameters": total,
                    "trainable_parameters": trainable,
                    "total_M": total / 1e6,
                    "trainable_M": trainable / 1e6,
                }
            )
    counts = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(args.output, index=False)
    print(counts[["group", "variant", "trainable_parameters"]].to_string(index=False))


if __name__ == "__main__":
    main()
