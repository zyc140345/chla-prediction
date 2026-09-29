from datetime import date, timedelta

import pytest

from chla_prediction.imagery.sequences import build_samples, split_samples

DATES = [date(2021, 1, d) for d in (1, 6, 11, 16, 26)] + [date(2021, 2, 10)]


def test_inputs_precede_origin_and_target_follows() -> None:
    samples = build_samples("hushan", DATES, input_window=3, min_input_observations=2)
    assert samples
    for sample in samples:
        input_dates = [DATES[i] for i in sample.input_indices]
        assert all(d <= sample.origin_date for d in input_dates)
        assert sample.target_date > sample.origin_date
        assert sample.lead_days == (sample.target_date - sample.origin_date).days


def test_samples_respect_the_input_window_and_minimum_observations() -> None:
    samples = build_samples("hushan", DATES, input_window=3, min_input_observations=2)
    assert all(len(s.input_indices) <= 3 for s in samples)
    assert min(len(s.input_indices) for s in samples) == 2
    gaps = samples[-1].interval_days
    assert gaps[0] == 0
    assert all(gap > 0 for gap in gaps[1:])


def test_unsorted_dates_rejected() -> None:
    with pytest.raises(ValueError, match="sorted"):
        build_samples("hushan", list(reversed(DATES)), input_window=3, min_input_observations=2)


def test_multi_lead_samples_extend_next_observation_samples() -> None:
    dates = DATES + [date(2021, 3, 20)]
    single = build_samples("hushan", dates, input_window=3, min_input_observations=2)
    multi = build_samples("hushan", dates, input_window=3, min_input_observations=2, max_lead_days=30)
    assert len(multi) > len(single)
    assert {(s.origin_date, s.target_date) for s in single} <= {(s.origin_date, s.target_date) for s in multi}
    for sample in multi:
        assert sample.target_date > sample.origin_date
        assert all(dates[i] <= sample.origin_date for i in sample.input_indices)
    # The next observation is always a target even beyond the lead limit.
    origins = {s.origin_date for s in single}
    assert origins == {s.origin_date for s in multi}
    long_lead = [s for s in multi if s.origin_date == date(2021, 2, 10)]
    assert [s.lead_days for s in long_lead] == [38]
    within = [s for s in multi if s.origin_date == date(2021, 1, 11)]
    assert [s.lead_days for s in within] == [5, 15, 30]


def test_days_before_target_count_back_from_the_target() -> None:
    dates = [date(2021, 1, day) for day in (1, 6, 16, 23)]
    sample = build_samples("water", dates, input_window=3, min_input_observations=3)[0]
    assert sample.interval_days == (0, 5, 10)
    assert sample.lead_days == 7
    assert sample.days_before_target == [22, 17, 7]


def test_chronological_split_has_no_target_leakage() -> None:
    samples = build_samples("hushan", DATES, input_window=3, min_input_observations=2)
    splits = split_samples(samples, train_end=date(2021, 1, 16), val_end=date(2021, 1, 31))
    assert all(s.target_date <= date(2021, 1, 16) for s in splits["train"])
    assert all(date(2021, 1, 16) < s.target_date <= date(2021, 1, 31) for s in splits["val"])
    assert all(s.target_date > date(2021, 1, 31) for s in splits["test"])
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(samples)


def test_train_start_limits_the_training_split_only() -> None:
    dates = [date(2021, 1, 1) + timedelta(days=10 * i) for i in range(40)]
    samples = build_samples("water", dates, input_window=3, min_input_observations=2)
    full = split_samples(samples, date(2021, 9, 1), date(2021, 11, 1))
    limited = split_samples(samples, date(2021, 9, 1), date(2021, 11, 1), train_start=date(2021, 5, 1))
    assert len(limited["train"]) < len(full["train"])
    assert all(s.first_input_date >= date(2021, 5, 1) for s in limited["train"])
    assert limited["val"] == full["val"]
    assert limited["test"] == full["test"]
