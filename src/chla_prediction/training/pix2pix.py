"""Train the pix2pix baseline as in the TensorFlow pix2pix tutorial the reference code follows.

BCE adversarial loss plus 100 x L1, both gradients computed before either
optimizer steps, Adam with beta1 0.5 and TF's epsilon placement, and early
stopping on the validation L1 of the forecast frame. Of ``optim`` only the
learning rate, batch size, workers and device apply: the reference uses no
weight decay, gradient clipping or mixed precision.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from chla_prediction import config as config_module
from chla_prediction.baselines import frames as frames_module
from chla_prediction.baselines import pix2pix as pix2pix_module
from chla_prediction.baselines.frames import fill_gaps
from chla_prediction.baselines.pix2pix import FRAMES, PatchDiscriminator, Pix2PixForecaster, pack_frames, to_tanh
from chla_prediction.config import TrainForecasterConfig
from chla_prediction.imagery import dataset as dataset_module
from chla_prediction.io import append_jsonl, file_sha256
from chla_prediction.training import train as train_module
from chla_prediction.training.loop import restore_rng, rng_state, save_atomic
from chla_prediction.training.objectives import masked_l1
from chla_prediction.training.train import build_loaders, make_loader

__all__ = ["train_pix2pix"]

L1_WEIGHT = 100.0
ADAM_BETAS = (0.5, 0.999)


def _real_packing(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    input_targets = batch["input_targets"].to(device)
    input_masks = batch["input_target_masks"].to(device)
    target = batch["target"].to(device)
    target_valid = batch["target_valid"].to(device)
    frames = torch.cat([input_targets[:, -FRAMES + 1 :, 0], target[:, 0:1]], dim=1)
    valid = torch.cat([input_masks[:, -FRAMES + 1 :], target_valid[:, None]], dim=1)
    return to_tanh(fill_gaps(frames, valid)), valid.to(frames.dtype)


def _shared_update(generator, discriminator, condition, real, valid, optimizers):
    """Compute both gradients before either optimizer changes any parameter."""
    fake = generator(condition)
    scores_real = discriminator(condition, real)
    scores_fake = discriminator(condition, fake)
    adversarial = F.binary_cross_entropy_with_logits(scores_fake, torch.ones_like(scores_fake))
    loss_d = F.binary_cross_entropy_with_logits(scores_real, torch.ones_like(scores_real))
    loss_d = loss_d + F.binary_cross_entropy_with_logits(scores_fake, torch.zeros_like(scores_fake))
    reconstruction = L1_WEIGHT * masked_l1(fake, real, valid)
    loss_g = adversarial + reconstruction
    params_g, params_d = tuple(generator.parameters()), tuple(discriminator.parameters())
    grads_g = torch.autograd.grad(loss_g, params_g, retain_graph=True)
    grads_d = torch.autograd.grad(loss_d, params_d)
    norm_g = torch.stack([g.square().sum() for g in grads_g]).sum().sqrt()
    norm_d = torch.stack([g.square().sum() for g in grads_d]).sum().sqrt()
    values = torch.stack([loss_g, loss_d, adversarial, norm_g, norm_d])
    if not torch.isfinite(values).all():
        raise FloatingPointError(f"Non-finite GAN loss or gradient: {values.detach().tolist()}")
    optimizer_g, optimizer_d = optimizers
    for optimizer, params, grads in ((optimizer_g, params_g, grads_g), (optimizer_d, params_d, grads_d)):
        optimizer.zero_grad(set_to_none=True)
        # TF 2.1 Adam places epsilon before the second-moment bias correction.
        previous = optimizer.state.get(params[0], {}).get("step", 0)
        step = int(previous) + 1
        for group in optimizer.param_groups:
            group["eps"] = 1e-7 / math.sqrt(1.0 - ADAM_BETAS[1] ** step)
        for param, grad in zip(params, grads, strict=True):
            param.grad = grad
        optimizer.step()
    stats = dict(
        zip(("generator", "discriminator", "adversarial", "grad_g", "grad_d"), values.detach().tolist(), strict=True)
    )
    stats["d_real_probability"] = float(scores_real.detach().sigmoid().mean())
    stats["d_fake_probability"] = float(scores_fake.detach().sigmoid().mean())
    return fake.detach(), stats


def _epoch(generator, discriminator, loader, device, optimizers=None, log_path=None, epoch=0):
    training = optimizers is not None
    generator.train(training)
    discriminator.train(training)
    totals, batches = {}, 0
    for batch in loader:
        condition = pack_frames(batch["input_targets"].to(device), batch["input_target_masks"].to(device))
        real, valid = _real_packing(batch, device)
        if training:
            fake, stats = _shared_update(generator, discriminator, condition, real, valid, optimizers)
        else:
            with torch.no_grad():
                fake = generator(condition)
            stats = {}
        stats["l1_forecast"] = float(masked_l1(fake[:, FRAMES - 1 :], real[:, FRAMES - 1 :], valid[:, FRAMES - 1 :]))
        if not all(math.isfinite(value) for value in stats.values()):
            raise FloatingPointError(f"Non-finite metrics: {stats}")
        for name, value in stats.items():
            totals[name] = totals.get(name, 0.0) + value
        batches += 1
        if log_path and (batches == 1 or batches % 50 == 0):
            record = {"epoch": epoch, "batch": batches, **stats}
            append_jsonl(log_path, record)
            print(json.dumps(record), flush=True)
    return {name: value / batches for name, value in totals.items()}


def train_pix2pix(config_path: Path, resume: bool = False) -> Path:
    """Train one pix2pix config and return its run directory."""
    config = TrainForecasterConfig.from_yaml(config_path)
    torch.manual_seed(config.seed)
    run_dir = config.output_dir / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "training_log.jsonl").exists() and not resume:
        raise FileExistsError(f"Use a new run_name or --resume: {run_dir}")
    (phase,) = config.phases
    loaders, counts = build_loaders(config)
    # Validation forecasts one image at a time, as inference does, because
    # batch normalization always uses the current batch.
    loaders["forecast_val"] = make_loader(
        loaders["forecast_val"].dataset, config.optim, shuffle=False, batch_size=1, prefetch=2
    )
    if not len(loaders["forecast"]) or not len(loaders["forecast_val"]):
        raise ValueError("GAN training requires nonempty train and validation splits")
    device = torch.device(config.optim.device)
    forecaster = Pix2PixForecaster(config.model, config.data).to(device)
    discriminator = PatchDiscriminator().to(device)
    optimizer_g = torch.optim.Adam(
        forecaster.generator.parameters(), lr=config.optim.learning_rate, betas=ADAM_BETAS, eps=1e-7
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=config.optim.learning_rate, betas=ADAM_BETAS, eps=1e-7
    )
    epochs = phase.epochs
    history, best, stale, start = [], float("inf"), 0, 0
    # A resume under changed training code would not continue the same run.
    modules = (sys.modules[__name__], pix2pix_module, frames_module, dataset_module, train_module, config_module)
    source_hashes = {module.__name__: file_sha256(module.__file__) for module in modules}
    manifest = {
        "source_hashes": source_hashes,
        "config": json.loads(config.model_dump_json()),
        "config_path": str(config_path),
        "seed": config.seed,
        "sample_counts": counts,
        "epochs": epochs,
        # Commit of the Nagkoulis et al. repository the method settings were checked against.
        "source_commit": "8dac0f725852a4305592f1d82e616c84a5ff28cd",
        "validation_batch_size": 1,
        "method": "BCE, lambda_L1=100, shared pre-update gradients, TF Adam epsilon, current-batch BN and dropout",
        "generator_parameters": sum(p.numel() for p in forecaster.generator.parameters()),
        "discriminator_parameters": sum(p.numel() for p in discriminator.parameters()),
    }
    resume_path = run_dir / "training_state.pt"
    if resume:
        saved_manifest = json.loads((run_dir / "pretrain_manifest.json").read_text())
        for key in ("config", "epochs", "source_hashes"):
            if saved_manifest[key] != manifest[key]:
                raise ValueError(f"Resume mismatch: {key}")
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        forecaster.load_state_dict(saved["generator"])
        discriminator.load_state_dict(saved["discriminator"])
        optimizer_g.load_state_dict(saved["optim_g"])
        optimizer_d.load_state_dict(saved["optim_d"])
        history, best, stale = saved["history"], saved["best"], saved["stale"]
        start = len(history)
        restore_rng(saved["rng"])
    manifest_path = run_dir / "pretrain_manifest.json"
    manifest.update(epochs_run=len(history), history=history, best_val_l1_forecast=best if history else None)
    manifest_path.write_text(json.dumps(manifest | {"status": "running"}, indent=2))
    loaders["forecast_val"].generator = torch.Generator().manual_seed(config.seed + 11000)
    for epoch in range(start, epochs):
        if phase.patience is not None and stale >= phase.patience:
            break
        # Epoch-indexed loader RNGs let a resume continue the same data order.
        loaders["forecast"].sampler.generator = torch.Generator().manual_seed(config.seed + epoch)
        loaders["forecast"].generator = torch.Generator().manual_seed(config.seed + epoch + 10000)
        train_stats = _epoch(
            forecaster.generator,
            discriminator,
            loaders["forecast"],
            device,
            (optimizer_g, optimizer_d),
            run_dir / "step_log.jsonl",
            epoch,
        )
        # Validation uses a fixed dropout stream and leaves the training RNG untouched.
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            torch.manual_seed(config.seed + 1000)
            val_stats = _epoch(forecaster.generator, discriminator, loaders["forecast_val"], device)
        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})
        record = {
            "phase": "gan",
            "epoch": epoch,
            "train_loss": train_stats["l1_forecast"],
            "val_loss": val_stats["l1_forecast"],
            "generator_loss": train_stats["generator"],
            "discriminator_loss": train_stats["discriminator"],
        }
        append_jsonl(run_dir / "training_log.jsonl", record)
        print(json.dumps(record), flush=True)
        if val_stats["l1_forecast"] < best:
            best, stale = val_stats["l1_forecast"], 0
            torch.save(forecaster.state_dict(), run_dir / "checkpoint_best.pt")
        else:
            stale += 1
        torch.save(forecaster.state_dict(), run_dir / "checkpoint_last.pt")
        state = {
            "generator": forecaster.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optim_g": optimizer_g.state_dict(),
            "optim_d": optimizer_d.state_dict(),
            "history": history,
            "best": best,
            "stale": stale,
            "rng": rng_state(),
        }
        save_atomic(resume_path, state)
        manifest.update(epochs_run=len(history), best_val_l1_forecast=best, history=history)
        manifest_path.write_text(json.dumps(manifest | {"status": "running"}, indent=2))
    manifest_path.write_text(json.dumps(manifest | {"status": "complete"}, indent=2))
    print(f"Wrote {run_dir}/checkpoint_best.pt (val forecast L1 {best:.4f})", flush=True)
    return run_dir
