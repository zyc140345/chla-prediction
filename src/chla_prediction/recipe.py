"""The paper's recipe: the proposed model and how it is pretrained, transferred, and scored.

Section 2 of the paper as configuration values. The config generator
(``tools/generate_forecast_configs.py``) builds the proposed model's runs
(variant ``ours``) from them; every other variant changes one of them. The
optimizer is AdamW with weight decay 1e-4 (``OptimConfig`` defaults).
"""

from __future__ import annotations

__all__ = [
    "CHLA_RANGE",
    "CROP_SIZE",
    "EVALUATION_WATERS",
    "FORECAST_OBJECTIVE",
    "MAX_SCENE_CLOUD_FRACTION",
    "MDN_WEIGHTS",
    "PRETRAINING_OPTIM",
    "PRETRAINING_PHASES",
    "PROPOSED_MODEL",
    "SEQUENCE",
    "SHORELINE_EROSION_PX",
    "TRANSFER_OPTIM",
    "TRANSFER_PHASES",
    "WATER_MASK_BAND",
    "WATER_MASK_THRESHOLD",
]

# The five evaluation waters (configs/target_waters.csv maps each to its HydroLAKES id).
EVALUATION_WATERS = ["baogu", "loweswater", "georges", "wentzel", "hushan"]

# Samples (Sections 2.2.1-2.2.3): scenes with at most 60% cloud, cirrus or
# shadow; up to six input scenes with at least four observed; chronological
# splits by target date into the pretraining or adaptation set, validation,
# and test.
MAX_SCENE_CLOUD_FRACTION = 0.6
SEQUENCE = {"input_window": 6, "min_input_observations": 4, "train_end": "2023-12-31", "val_end": "2024-06-30"}
CROP_SIZE = 256
# Pseudo-labels (Section 2.2.2) and the Chl-a metrics clip Chl-a to this
# range (mg/m^3) before the log.
CHLA_RANGE = (0.1, 1000.0)

# Model (Section 2.3): the spectral-spatial encoder whose spectral stage is
# initialized from the pretrained MDN, the target-relative temporal
# attention, and the lead-conditioned decoder (models/).
MDN_WEIGHTS = "data/models/mdn/mdn_msi_chl.npz"
PROPOSED_MODEL = {
    "name": "target_relative_attention_forecaster",
    "base_channels": 32,
    "latent_channels": 128,
    "hidden_channels": 128,
    "encoder": "mdn_spectral",
    "mdn_weights": MDN_WEIGHTS,
}

# Training (Section 2.4): masked L1 forecast loss plus the reconstruction
# loss at weight 0.3 (training/objectives.py, training/loop.py).
FORECAST_OBJECTIVE = {"reconstruction_weight": 0.3}
# Multi-water pretraining: a two-epoch warm-up that reconstructs a scene's
# pseudo-label from corrupted reflectance, then at most 12 forecasting epochs
# with early stopping on the validation loss.
PRETRAINING_PHASES = [
    {"name": "warmup", "objective": "reconstruction", "epochs": 2},
    {"name": "forecast", "objective": "forecast", "epochs": 12, "patience": 2},
]
PRETRAINING_OPTIM = {"learning_rate": 2.0e-4, "batch_size": 8}
# Few-shot transfer: three epochs on the target water's adaptation set,
# every parameter trainable, from the best pretrained checkpoint.
TRANSFER_PHASES = [{"name": "forecast", "objective": "forecast", "epochs": 3}]
TRANSFER_OPTIM = {"learning_rate": 5.0e-5, "batch_size": 4}

# Evaluation (Section 2.5.3): the fixed water-extent mask is the share of
# training-period Sentinel-1 acquisitions below -17.5 dB, thresholded at half.
WATER_MASK_BAND = "s1_median_water"
WATER_MASK_THRESHOLD = 50.0
# Rings of shoreline pixels dropped from the mask.
SHORELINE_EROSION_PX = 1
