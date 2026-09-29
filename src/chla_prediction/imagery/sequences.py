"""Leakage-safe sequence sampling over dated scene archives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

__all__ = ["SequenceSample", "build_samples", "split_samples"]


@dataclass(frozen=True)
class SequenceSample:
    water_id: str
    input_indices: tuple[int, ...]
    target_index: int
    origin_date: date
    target_date: date
    interval_days: tuple[int, ...]
    lead_days: int

    @property
    def first_input_date(self) -> date:
        return self.origin_date - timedelta(days=sum(self.interval_days))

    @property
    def days_before_target(self) -> list[int]:
        """Days between each input scene and the target scene."""
        days, total = [], self.lead_days
        for interval in reversed(self.interval_days):
            days.append(total)
            total += interval
        return days[::-1]


def build_samples(
    water_id: str,
    dates: list[date],
    input_window: int,
    min_input_observations: int,
    max_lead_days: int | None = None,
) -> list[SequenceSample]:
    """Build next-observation samples: up to ``input_window`` scenes ending at the origin predict the next scene.

    With ``max_lead_days`` every later scene within that many days of the
    origin is a target as well. Inputs fall on or before the origin and
    targets strictly after it; ``split_samples`` splits on the target date.
    """
    if list(dates) != sorted(dates):
        raise ValueError("Scene dates must be sorted ascending")
    samples = []
    for origin_index in range(min_input_observations - 1, len(dates) - 1):
        first_input = max(0, origin_index + 1 - input_window)
        input_indices = tuple(range(first_input, origin_index + 1))
        input_dates = dates[first_input : origin_index + 1]
        origin = input_dates[-1]
        intervals = tuple((input_dates[i] - input_dates[i - 1]).days if i > 0 else 0 for i in range(len(input_dates)))
        for target_index in range(origin_index + 1, len(dates)):
            lead = (dates[target_index] - origin).days
            if target_index > origin_index + 1 and (max_lead_days is None or lead > max_lead_days):
                break
            samples.append(
                SequenceSample(
                    water_id=water_id,
                    input_indices=input_indices,
                    target_index=target_index,
                    origin_date=origin,
                    target_date=dates[target_index],
                    interval_days=intervals,
                    lead_days=lead,
                )
            )
    return samples


def split_samples(
    samples: list[SequenceSample],
    train_end: date,
    val_end: date,
    train_start: date | None = None,
) -> dict[str, list[SequenceSample]]:
    """Split chronologically on the target date, so no training sample sees a scene after ``train_end``.

    ``train_start`` keeps only training samples whose inputs all fall on or
    after it; validation and test are unaffected.
    """
    splits: dict[str, list[SequenceSample]] = {"train": [], "val": [], "test": []}
    for sample in samples:
        if sample.target_date <= train_end:
            if train_start is None or sample.first_input_date >= train_start:
                splits["train"].append(sample)
        elif sample.target_date <= val_end:
            splits["val"].append(sample)
        else:
            splits["test"].append(sample)
    return splits
