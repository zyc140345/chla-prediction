"""Export the reference MDN (MSI, chl) weights for ``chla_prediction.retrieval.MDN``.

Runs the original implementation (https://github.com/BrandonSmithJ/MDN,
GPL-3.0; TensorFlow 2.10 + tensorflow-probability 0.18 + scikit-learn 0.24
on Python 3.9 is a working stack) and writes:

    mdn_msi_chl.npz    round-wise MLP weights, scaler parameters, wavelengths
    mdn_reference.npz  Rrs inputs and the reference Chl-a outputs for tests

Run it in an environment where the cloned repository, with its LFS weights
pulled, is importable as ``MDN``; copy ``mdn_msi_chl.npz`` to
``data/models/mdn/``:

    python tools/data/export_mdn_weights.py <mdn_repo_parent> <output_dir>
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def reference_spectra(n: int = 4000) -> np.ndarray:
    """Log-uniform Rrs over a wide inland-water range in the seven MSI bands, plus bloom and flat spectra."""
    rng = np.random.default_rng(0)
    rrs = np.exp(rng.uniform(np.log(2e-4), np.log(6e-2), size=(n, 7))).astype(np.float32)
    rrs[:200, 4] *= 3.0
    rrs[200:400, :] = rrs[200:201, :] * np.linspace(0.5, 2, 200)[:, None]
    return rrs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mdn_parent", type=Path, help="Directory holding the cloned MDN repository")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.mdn_parent))
    from MDN.parameters import get_args
    from MDN.product_estimation import get_estimates

    rrs = reference_spectra()
    mdn_args = get_args(sensor="MSI", use_cmdline=False, product="chl", silent=True)
    outputs, _ = get_estimates(mdn_args, x_test=rrs, return_model=True, return_coefs=True)
    models = outputs["model"]
    estimates = np.array(outputs["estimates"])  # [rounds, n, n_targets]
    median = np.median(estimates, 0)
    print(f"{mdn_args.config_name}: {len(models)} rounds, median Chl-a {median.min():.3g}-{median.max():.3g}")

    export = {"wavelengths": np.array(mdn_args.wavelengths, dtype=np.float32), "n_rounds": len(models)}
    for r, model in enumerate(models):
        for i, weight in enumerate(model.model.get_weights()):
            export[f"round{r}_w{i}"] = weight
        x_scalers, y_scalers = model.scalerx.scalers, model.scalery.scalers
        names = [type(scaler).__name__ for scaler in x_scalers + y_scalers]
        if names != ["RobustScaler", "LogTransformer", "MinMaxScaler"]:
            raise ValueError(f"Unexpected MDN scalers: {names}")
        export[f"round{r}_x_center"] = x_scalers[0].center_
        export[f"round{r}_x_scale"] = x_scalers[0].scale_
        export[f"round{r}_y_min"] = y_scalers[1].min_
        export[f"round{r}_y_scale"] = y_scalers[1].scale_
        export[f"round{r}_n_mix"] = model.n_mix
        export[f"round{r}_n_targets"] = model.n_targets
        export[f"round{r}_epsilon"] = model.epsilon
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_dir / "mdn_msi_chl.npz", **export)
    np.savez(args.output_dir / "mdn_reference.npz", rrs=rrs, chl_median=median, chl_rounds=estimates)
    print("wrote", args.output_dir)


if __name__ == "__main__":
    main()
