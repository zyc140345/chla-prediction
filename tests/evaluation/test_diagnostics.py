import pandas as pd
import pytest

from chla_prediction.evaluation.diagnostics import select_validation_checkpoint


def test_checkpoint_selection_ignores_better_test_and_train_scores() -> None:
    frame = pd.DataFrame(
        [
            {"split": "val", "epoch": 3, "learning_rate": 0.0002, "rmse_log": 0.3},
            {"split": "val", "epoch": 1, "learning_rate": 0.00005, "rmse_log": 0.4},
            {"split": "test", "epoch": 1, "learning_rate": 0.00005, "rmse_log": 0.01},
            {"split": "train", "epoch": 40, "learning_rate": 0.0002, "rmse_log": 0.001},
        ]
    )
    selected = select_validation_checkpoint(frame)
    assert selected["epoch"] == 3
    assert selected["split"] == "val"
    with pytest.raises(ValueError, match="nonempty validation"):
        select_validation_checkpoint(frame[frame.split != "val"])
