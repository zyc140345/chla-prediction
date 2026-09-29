# Chlorophyll-a Field Forecasting for Data-Scarce Small Inland Waters by Multi-Water Pretraining and Few-Shot Transfer on Sentinel-2 Image Time Series

This repository contains the official implementation for the work
"Chlorophyll-a Field Forecasting for Data-Scarce Small Inland Waters by
Multi-Water Pretraining and Few-Shot Transfer on Sentinel-2 Image Time Series".

> Deep-learning forecasters of chlorophyll-a (Chl-a) fields, although valuable for bloom management, require long
> image records to learn spatiotemporal dynamics, which coarse-resolution satellites provide for large lakes and
> coastal waters. Small inland waters, key sources of drinking water and irrigation for local communities, rely on
> decametric satellites that were launched later, revisit less frequently, and lose entire scenes to cloud cover, so
> each water yields too few scenes to train a deep forecaster without overfitting. We address this problem with
> multi-water pretraining and few-shot transfer. We build a Sentinel-2 corpus of 441 lakes and reservoirs with more
> than 70,000 samples pseudo-labeled by a mixture density network, design a compact forecaster conditioned on the
> acquisition intervals and the forecast lead, pretrain it on the corpus, and adapt it to each target water with a
> few hundred local samples. On five evaluation waters, the method alleviates overfitting, reduces the log-space
> root-mean-square error by 13.3% relative to the strongest baseline, and attains the lowest error at every forecast
> lead. Multi-water pretraining thus makes field-scale Chl-a forecasting practical for small waters.

## Project Structure

```
.
├── src/chla_prediction/
│   ├── recipe.py                    # hyper-parameters of the proposed method (Sec. 2.5.2)
│   ├── config.py                    # typed configs
│   ├── imagery/                     # dataset construction (Sec. 2.2)
│   │   ├── archive.py               # Sentinel-2 scene archives and pixel validity
│   │   ├── masks.py                 # SCL masks and the fixed water-extent mask
│   │   ├── pseudo_labels.py         # log10 Chl-a pseudo-label frames
│   │   ├── sequences.py             # input windows, forecast targets, chronological splits
│   │   ├── waters.py                # archives and sample splits of the configured waters
│   │   └── dataset.py               # PyTorch datasets
│   ├── retrieval/                   # Chl-a pseudo-labels (Sec. 2.2.2)
│   │   ├── mdn.py                   # MDN inference
│   │   └── precompute.py            # pseudo-label raster of every scene
│   ├── models/                      # the proposed forecaster (Sec. 2.3)
│   │   ├── encoder.py               # spectral-spatial encoder initialized from the MDN
│   │   ├── temporal.py              # target-relative temporal attention and the ablated temporal modules
│   │   ├── decoder.py               # lead-conditioned Chl-a field decoder
│   │   └── forecaster.py            # encoder + temporal module + decoder
│   ├── training/                    # multi-water pretraining and few-shot transfer (Sec. 2.4)
│   ├── baselines/                   # Persistence, Climatology, Random forest, XGBoost, ConvLSTM, pix2pix (Sec. 2.5.1)
│   ├── inference/                   # forecast rasters and prediction manifests
│   ├── evaluation/                  # evaluation protocol (Sec. 2.5.3)
│   └── experiments/                 # command-line entry points
├── configs/
│   ├── target_waters.csv            # the five evaluation waters
│   ├── pretraining_corpus.csv       # candidate waters of the pretraining corpus
│   ├── pretraining_waters.txt       # the 441 waters that form the corpus
│   └── paper/                       # configs of the main comparison
├── tools/
│   ├── data/                        # data download
│   └── figures/                     # figure scripts
└── tests/
```

## Environment Setup

```shell
git clone https://github.com/zyc140345/chla-prediction.git && cd chla-prediction
uv sync --extra gee --extra figures
```

Training needs a CUDA GPU, and the pretraining corpus takes about 560 GB.

## Data Preparation

1. Sentinel-2 L2A scenes (Microsoft Planetary Computer)

```shell
# 441 pretraining waters
uv run --script tools/data/stac_download_waters.py --waters configs/pretraining_corpus.csv \
  --water-ids configs/pretraining_waters.txt --output-dir data/corpus
# 5 evaluation waters
uv run --script tools/data/stac_download_waters.py --waters configs/target_waters.csv --output-dir data/evaluation
```

2. Water-extent masks (Google Earth Engine)

```shell
uv run earthengine authenticate
export CHLA_GEE_PROJECT=<your-cloud-project>
# Download the water layers onto each evaluation water's grid
uv run python tools/data/fetch_water_extent.py dump-grids configs/paper/evaluate.yaml grids.json
uv run python tools/data/fetch_water_extent.py fetch grids.json data/water_masks
```

3. MDN weights ([reference implementation](https://github.com/BrandonSmithJ/MDN))

```shell
git clone https://github.com/BrandonSmithJ/MDN <parent>/MDN && git -C <parent>/MDN lfs pull
# In the reference implementation's environment (Python 3.9, TensorFlow 2.10, tensorflow-probability 0.18, scikit-learn 0.24)
python tools/data/export_mdn_weights.py <parent> data/models/mdn
```

4. MDN pseudo-labels

```shell
uv run python -m chla_prediction.experiments.precompute_pseudo_labels --config configs/paper/precompute_pseudo_labels.yaml
```

## Reproduce the Main Results

1. The proposed method

```shell
# Multi-water pretraining
uv run python -m chla_prediction.experiments.train_forecaster --config configs/paper/pretrain_ours.yaml
# Few-shot transfer and forecasts on each evaluation water
for water in baogu loweswater georges wentzel hushan; do
  uv run python -m chla_prediction.experiments.train_forecaster --config configs/paper/adapt_ours_${water}.yaml
  uv run python -m chla_prediction.experiments.run_forecast_inference --config configs/paper/infer_ours_${water}.yaml
done
# Merge the per-water forecasts
uv run python -m chla_prediction.experiments.merge_predictions --output outputs/runs/merged_ours_adapted.jsonl \
  outputs/runs/infer_ours_{baogu,loweswater,georges,wentzel,hushan}/prediction_manifest.jsonl
```

2. Baselines

```shell
# Persistence and Climatology (computed from each water's history)
uv run python -m chla_prediction.experiments.run_forecast_inference --config configs/paper/infer_persistence.yaml
uv run python -m chla_prediction.experiments.precompute_climatology --config configs/paper/precompute_climatology.yaml
uv run python -m chla_prediction.experiments.run_forecast_inference --config configs/paper/infer_climatology.yaml
# Random forest and XGBoost (fitted on each water)
uv run python -m chla_prediction.experiments.run_tabular_baseline --config configs/paper/baseline_rf.yaml
uv run python -m chla_prediction.experiments.run_tabular_baseline --config configs/paper/baseline_xgboost.yaml
# ConvLSTM (trained from scratch on each water)
for water in baogu loweswater georges wentzel hushan; do
  uv run python -m chla_prediction.experiments.train_forecaster --config configs/paper/train_scratch_convlstm_${water}.yaml
  uv run python -m chla_prediction.experiments.run_forecast_inference --config configs/paper/infer_scratch_convlstm_${water}.yaml
done
uv run python -m chla_prediction.experiments.merge_predictions --output outputs/runs/merged_scratch_convlstm.jsonl \
  outputs/runs/infer_scratch_convlstm_{baogu,loweswater,georges,wentzel,hushan}/prediction_manifest.jsonl
# pix2pix (pretrained on the corpus, applied zero-shot)
uv run python -m chla_prediction.experiments.train_pix2pix --config configs/paper/pretrain_pix2pix.yaml
uv run python -m chla_prediction.experiments.run_forecast_inference --config configs/paper/infer_pix2pix_zero.yaml
```

3. Evaluation

```shell
# Score the forecasts (variants not run are skipped)
uv run python -m chla_prediction.experiments.evaluate_forecast --config configs/paper/evaluate.yaml
# Paper tables
uv run python -m chla_prediction.experiments.make_paper_tables outputs/runs/evaluation_transfer \
  --output-dir outputs/runs/paper
```

4. Figures

```shell
# One script per figure in tools/figures/, e.g. the forecast-lead bins
uv run python tools/figures/plot_lead_bins.py --tables-dir outputs/runs/paper \
  --evaluation-dir outputs/runs/evaluation_transfer --output figures/lead_bins
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
