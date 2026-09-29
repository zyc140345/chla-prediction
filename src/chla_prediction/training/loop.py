"""Phase-based training loop with epoch-level resume."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chla_prediction.config import TrainForecasterConfig
from chla_prediction.io import append_jsonl
from chla_prediction.training.objectives import masked_reconstruction_loss, next_scene_loss

__all__ = ["PhaseResult", "restore_rng", "rng_state", "run_training_phases", "save_atomic"]

RESUME_NAME = "checkpoint_resume.pt"
STEPS_NAME = "training_steps.jsonl"
TIMING_NAME = "timing_log.jsonl"


@dataclass(frozen=True)
class PhaseResult:
    name: str
    epochs: int
    final_train_loss: float
    final_val_loss: float | None


def _reconstruction_step(model, batch, config, device) -> torch.Tensor:
    return masked_reconstruction_loss(
        model,
        batch["image"].to(device),
        batch["valid"].to(device),
        batch["target"].to(device),
        batch["target_valid"].to(device),
        config.objective,
    )


def _forecast_step(model, batch, config, device) -> torch.Tensor:
    images = batch["images"].to(device)
    masks = batch["masks"].to(device)
    prediction = model.forecast(
        images,
        masks,
        batch["frame_valid"].to(device),
        batch["interval_days"].to(device),
        batch["lead_days"].to(device),
        input_targets=batch["input_targets"].to(device),
        input_target_masks=batch["input_target_masks"].to(device),
    )
    total = next_scene_loss(prediction, batch["target"].to(device), batch["target_valid"].to(device), config.objective)
    # Keep the reconstruction loss on the last input frame at reduced weight
    # so the encoder-decoder mapping does not drift.
    weight = config.objective.reconstruction_weight
    if weight > 0:
        reconstruction = masked_reconstruction_loss(
            model,
            images[:, -1],
            masks[:, -1],
            batch["input_targets"][:, -1].to(device),
            batch["input_target_masks"][:, -1].to(device),
            config.objective,
        )
        return total + weight * reconstruction
    return total


@dataclass(frozen=True)
class _EpochStats:
    loss: float
    grad_norm: float | None  # pre-clip total norm, mean over steps; None for validation
    seconds: float


class _StepWriter:
    """Step log: a rolling-mean record every ``interval`` training steps and a partial one closing the epoch."""

    def __init__(self, path: Path, interval: int, phase: str, epoch: int):
        self.path, self.interval, self.phase, self.epoch = path, interval, phase, epoch
        self.step, self.lines = 0, 0
        self._losses: list[float] = []
        self._norms: list[float] = []
        self._lr = 0.0

    def record(self, loss: float, grad_norm: float, lr: float) -> None:
        self.step += 1
        self._losses.append(loss)
        self._norms.append(grad_norm)
        self._lr = lr
        if len(self._losses) >= self.interval:
            self._flush()

    def _flush(self) -> None:
        record = {
            "phase": self.phase,
            "epoch": self.epoch,
            "step": self.step,
            "loss": sum(self._losses) / len(self._losses),
            "grad_norm": sum(self._norms) / len(self._norms),
            "learning_rate": self._lr,
        }
        append_jsonl(self.path, record)
        self.lines += 1
        self._losses, self._norms = [], []

    def close(self) -> None:
        if self._losses:
            self._flush()


def _run_epoch(model, loader, step_fn, config, device, optimizer=None, step_writer=None) -> _EpochStats:
    training = optimizer is not None
    model.train(training)
    total, norm_total, count = 0.0, 0.0, 0
    amp = config.optim.amp and device.type == "cuda"
    start = time.perf_counter()
    with torch.set_grad_enabled(training):
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                loss = step_fn(model, batch, config, device)
            loss = loss.float()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip_norm))
                optimizer.step()
                norm_total += grad_norm
                step_writer.record(float(loss.detach()), grad_norm, optimizer.param_groups[0]["lr"])
            total += float(loss.detach())
            count += 1
    if count == 0:
        raise ValueError("Empty data loader; check archive filters and split boundaries")
    return _EpochStats(
        loss=total / count,
        grad_norm=norm_total / count if training else None,
        seconds=time.perf_counter() - start,
    )


def _config_hash(config: TrainForecasterConfig) -> str:
    return hashlib.sha256(config.model_dump_json().encode("utf-8")).hexdigest()


def rng_state() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([cuda.cpu() for cuda in state["cuda"]])


def save_atomic(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _truncate(path: Path, lines: int) -> None:
    """Keep the first ``lines`` records, those of the epochs whose resume state reached disk."""
    if path.exists():
        kept = path.read_text(encoding="utf-8").splitlines(keepends=True)[:lines]
        path.write_text("".join(kept), encoding="utf-8")


def run_training_phases(
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    config: TrainForecasterConfig,
    run_dir: Path,
) -> list[PhaseResult]:
    """Run the configured phases and write logs and checkpoints.

    ``loaders`` maps ``reconstruction`` (warm-up) / ``forecast`` (forecasting)
    / ``forecast_val`` to data loaders; a phase uses the loader matching its
    objective.

    Three log files per run: ``training_log.jsonl`` holds one record per
    epoch (losses, mean pre-clip gradient norm, learning rate),
    ``training_steps.jsonl`` a rolling-mean record every
    ``optim.step_log_interval`` training steps, and ``timing_log.jsonl``
    the wall-clock seconds per epoch. Timings are kept apart because a
    resumed run repeats the lost epochs with different timings.

    Interruption safety: the end of every epoch atomically rewrites
    ``checkpoint_resume.pt`` with the model, optimizer (including the
    scheduler-reduced learning rate), plateau-scheduler state, RNG streams,
    and early-stopping counters. When that file exists the run continues
    from it with the same data order and RNG state. The file is removed once
    every phase has finished, so rerunning a completed run starts fresh and
    replaces its checkpoints and logs. A resume file written by a different
    config raises an error; delete the run directory to retrain.
    """
    device = torch.device(config.optim.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
    )
    step_fns = {"reconstruction": _reconstruction_step, "forecast": _forecast_step}
    log_path = run_dir / "training_log.jsonl"
    steps_path = run_dir / STEPS_NAME
    timing_path = run_dir / TIMING_NAME
    resume_path = run_dir / RESUME_NAME
    config_hash = _config_hash(config)

    state: dict | None = None
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
        if state["config_hash"] != config_hash:
            raise ValueError(
                f"{resume_path} was written by a different config; delete {run_dir} to retrain from scratch"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        _truncate(log_path, state["log_lines"])
        _truncate(timing_path, state["log_lines"])  # one timing line per epoch record
        _truncate(steps_path, state["step_lines"])
        print(f"Resuming {run_dir.name} at phase {state['phase_index']} epoch {state['epoch']}")
    else:
        # Fresh start: drop old logs so two runs never interleave.
        for path in (log_path, steps_path, timing_path):
            path.unlink(missing_ok=True)

    results = [PhaseResult(**entry) for entry in state["results"]] if state else []
    best_val = state["best_val"] if state else float("inf")
    log_lines = state["log_lines"] if state else 0
    step_lines = state["step_lines"] if state else 0
    start_phase = state["phase_index"] if state else 0
    scheduler = None
    train_loss, val_loss = float("nan"), None
    best_phase_val, stale, epochs_done = float("inf"), 0, 0

    def save_state(next_phase: int, next_epoch: int) -> None:
        save_atomic(
            resume_path,
            {
                "config_hash": config_hash,
                "phase_index": next_phase,
                "epoch": next_epoch,
                "results": [asdict(result) for result in results],
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "best_val": best_val,
                "best_phase_val": best_phase_val,
                "stale": stale,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "log_lines": log_lines,
                "step_lines": step_lines,
                "rng": rng_state(),
            },
        )

    for phase_index, phase in enumerate(config.phases):
        if phase_index < start_phase:
            continue  # finished before the interruption; its result is restored
        step_fn = step_fns[phase.objective]
        scheduler = None
        if config.optim.lr_plateau_patience is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5, patience=config.optim.lr_plateau_patience
            )
        resuming = state is not None and phase_index == start_phase
        if resuming and scheduler is not None and state["scheduler"] is not None:
            scheduler.load_state_dict(state["scheduler"])
        train_loss = state["train_loss"] if resuming else float("nan")
        val_loss = state["val_loss"] if resuming else None
        best_phase_val = state["best_phase_val"] if resuming else float("inf")
        stale = state["stale"] if resuming else 0
        epochs_done = state["epoch"] if resuming else 0
        if epochs_done == 0 and 0 in phase.checkpoint_epochs:
            torch.save(model.state_dict(), run_dir / f"checkpoint_{phase.name}_epoch000.pt")
        for epoch in range(epochs_done, phase.epochs):
            if phase.patience is not None and stale >= phase.patience:
                break
            writer = _StepWriter(steps_path, config.optim.step_log_interval, phase.name, epoch)
            train_stats = _run_epoch(model, loaders[phase.objective], step_fn, config, device, optimizer, writer)
            writer.close()
            step_lines += writer.lines
            train_loss = train_stats.loss
            record = {"phase": phase.name, "epoch": epoch, "train_loss": train_loss, "grad_norm": train_stats.grad_norm}
            timing = {"phase": phase.name, "epoch": epoch, "train_seconds": round(train_stats.seconds, 2)}
            if phase.objective == "forecast" and len(loaders["forecast_val"].dataset) > 0:
                val_stats = _run_epoch(model, loaders["forecast_val"], step_fn, config, device)
                val_loss = val_stats.loss
                record["val_loss"] = val_loss
                record["learning_rate"] = optimizer.param_groups[0]["lr"]
                timing["val_seconds"] = round(val_stats.seconds, 2)
                if scheduler is not None:
                    scheduler.step(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model.state_dict(), run_dir / "checkpoint_best.pt")
                stale = 0 if val_loss < best_phase_val else stale + 1
                best_phase_val = min(best_phase_val, val_loss)
            append_jsonl(log_path, record)
            append_jsonl(timing_path, timing)
            log_lines += 1
            epochs_done = epoch + 1
            if epochs_done in phase.checkpoint_epochs:
                torch.save(model.state_dict(), run_dir / f"checkpoint_{phase.name}_epoch{epochs_done:03d}.pt")
            save_state(phase_index, epochs_done)
        best_path = run_dir / "checkpoint_best.pt"
        if phase.patience is not None and best_path.exists():
            # Early stopping: continue from the best validation weights.
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        torch.save(model.state_dict(), run_dir / f"checkpoint_{phase.name}.pt")
        results.append(
            PhaseResult(name=phase.name, epochs=epochs_done, final_train_loss=train_loss, final_val_loss=val_loss)
        )
        # Phase-scoped values reset before the boundary save, so a resume
        # landing on the next phase starts it clean.
        scheduler, train_loss, val_loss = None, float("nan"), None
        best_phase_val, stale, epochs_done = float("inf"), 0, 0
        save_state(phase_index + 1, 0)
    resume_path.unlink(missing_ok=True)
    return results
