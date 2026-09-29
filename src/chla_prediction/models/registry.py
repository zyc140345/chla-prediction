"""The model names of the configs, mapped to their classes."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from chla_prediction.baselines.pix2pix import Pix2PixForecaster
from chla_prediction.baselines.reference import ClimatologyForecaster, PersistenceForecaster
from chla_prediction.baselines.stacked_conv_lstm import StackedConvLSTMForecaster
from chla_prediction.config import DataConfig, ModelConfig
from chla_prediction.models.forecaster import Forecaster, IntervalAwareForecaster
from chla_prediction.models.temporal import (
    ConvGRU,
    ConvLSTM,
    SimVPTranslator,
    SpaceTimeTransformer,
    TargetRelativeAttention,
)

__all__ = ["MODELS", "build_model"]

MODELS: dict[str, Callable[[ModelConfig, DataConfig], Forecaster]] = {
    "target_relative_attention_forecaster": partial(IntervalAwareForecaster, temporal_cls=TargetRelativeAttention),
    "conv_gru_forecaster": partial(IntervalAwareForecaster, temporal_cls=ConvGRU),
    "conv_lstm_forecaster": partial(IntervalAwareForecaster, temporal_cls=ConvLSTM),
    "simvp_forecaster": partial(IntervalAwareForecaster, temporal_cls=SimVPTranslator),
    "space_time_transformer_forecaster": partial(IntervalAwareForecaster, temporal_cls=SpaceTimeTransformer),
    "persistence_forecaster": PersistenceForecaster,
    "climatology_forecaster": ClimatologyForecaster,
    "stacked_conv_lstm_forecaster": StackedConvLSTMForecaster,
    "pix2pix_forecaster": Pix2PixForecaster,
}


def build_model(model_config: ModelConfig, data_config: DataConfig) -> Forecaster:
    if model_config.name not in MODELS:
        raise KeyError(f"Unknown model {model_config.name!r}; known: {sorted(MODELS)}")
    return MODELS[model_config.name](model_config, data_config)
