import numpy as np
import pytest
from scipy import stats

from smartphone_stats.data import generate_sample_dataset
from smartphone_stats.estimation import estimate_by_brand, mean_battery_life


def test_estimate_matches_manual_t_interval():
    values = [10.0, 12.0, 14.0, 11.0, 13.0, 15.0, 12.0, 14.0]
    est = mean_battery_life(values, confidence=0.95)
    arr = np.asarray(values)
    n = arr.size
    sem = arr.std(ddof=1) / np.sqrt(n)
    critical = stats.t.ppf(0.975, n - 1)

    assert est.n == n
    assert est.mean == pytest.approx(arr.mean())
    assert est.sem == pytest.approx(sem)
    assert est.lower == pytest.approx(arr.mean() - critical * sem)
    assert est.upper == pytest.approx(arr.mean() + critical * sem)
    assert est.lower < est.mean < est.upper


def test_narrower_interval_with_more_data():
    rng = np.random.default_rng(0)
    small = mean_battery_life(rng.normal(12, 1, 10))
    large = mean_battery_life(rng.normal(12, 1, 400))
    assert (large.upper - large.lower) < (small.upper - small.lower)


def test_estimate_by_brand_includes_pool():
    df = generate_sample_dataset(n_per_brand=20, seed=2)
    estimates = estimate_by_brand(df)
    assert set(estimates) == {"Apex", "Bolt", "Cosmos", "Delta", "All brands"}
    assert estimates["All brands"].n == 80
    # Pooled estimate must sit within the range of the brand estimates.
    means = [e.mean for e in estimates.values()]
    assert min(means) < estimates["All brands"].mean < max(means)


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
def test_wider_interval_for_higher_confidence(confidence):
    est = mean_battery_life([10.0, 11.0, 12.0, 13.0, 14.0], confidence=confidence)
    assert est.confidence == confidence
    assert est.lower < est.upper


def test_too_few_values_raises():
    with pytest.raises(ValueError, match="at least two"):
        mean_battery_life([12.0])


def test_bad_confidence_raises():
    with pytest.raises(ValueError):
        mean_battery_life([1.0, 2.0], confidence=1.5)
