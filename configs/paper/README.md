# Configs of the main comparison

The runs behind the paper's main comparison, as
`tools/generate_forecast_configs.py` writes them: the pseudo-labels and the
climatology (`precompute_*`), the proposed model's multi-water pretraining,
few-shot transfer and forecasts on each evaluation water (`pretrain_ours`,
`adapt_ours_<water>`, `infer_ours_<water>`), the six baselines, and the
evaluation. The generator also writes them, with every other run (ablations,
the data-amount study, diagnostics), into the untracked `configs/generated/`,
and a test keeps this directory equal to its output.
