import numpy as np
import pytest

from smartphone_stats import claims
from smartphone_stats.claims import DEFAULT_CLAIMS
from smartphone_stats.data import generate_sample_dataset

# Underscore aliases: pytest collects any module-level ``test_*`` name, and
# these library functions are deliberately named ``test_*``.
_tmc = claims.test_manufacturer_claim
_tcb = claims.test_claims_by_brand


def test_t_statistic_matches_hand_calculation():
    values = [11.0, 12.5, 13.0, 11.5, 12.0, 13.5, 12.2, 11.8]
    claim = 12.5
    arr = np.asarray(values)
    expected_t = (arr.mean() - claim) / (arr.std(ddof=1) / np.sqrt(arr.size))

    result = _tmc(values, claim=claim)
    assert result.t_statistic == pytest.approx(expected_t)
    assert result.sample_mean == pytest.approx(arr.mean())
    assert 0.0 <= result.p_value <= 1.0


def test_zero_variance_claim_at_mean_is_not_rejected():
    result = _tmc([12.0, 12.0, 12.0, 12.0], claim=12.0)
    assert result.t_statistic == 0.0
    assert result.p_value == 1.0
    assert not result.rejected
    assert "consistent with the data" in result.verdict


def test_zero_variance_claim_away_from_mean_is_rejected():
    result = _tmc([12.0, 12.0, 12.0, 12.0], claim=10.0)
    assert result.rejected
    assert result.p_value == 0.0


def test_outrageous_claim_is_rejected():
    rng = np.random.default_rng(4)
    values = rng.normal(10.0, 0.5, 50)
    result = _tmc(values, claim=20.0)
    assert result.rejected
    assert "significantly below" in result.verdict


def test_one_sided_alternatives_split_the_two_sided_p():
    rng = np.random.default_rng(5)
    values = rng.normal(10.0, 0.5, 40)  # sample mean lands below the claim
    greater = _tmc(values, claim=10.0, alternative="greater")
    less = _tmc(values, claim=10.0, alternative="less")
    assert less.p_value < 0.05 < greater.p_value
    assert greater.p_value + less.p_value == pytest.approx(1.0)


def test_ci_matches_confidence_level():
    rng = np.random.default_rng(6)
    values = rng.normal(12.0, 1.0, 30)
    result95 = _tmc(values, claim=12.0)
    result80 = _tmc(values, claim=12.0, alpha=0.20)
    assert (result80.ci_upper - result80.ci_lower) < (result95.ci_upper - result95.ci_lower)


def test_default_claims_cover_every_brand():
    df = generate_sample_dataset(n_per_brand=20, seed=8)
    assert set(DEFAULT_CLAIMS) == set(df["brand"].unique())


def test_claims_by_brand_runs_and_flags():
    df = generate_sample_dataset(n_per_brand=30, seed=9)
    results = _tcb(df)
    assert set(results) == set(DEFAULT_CLAIMS)
    # Bolt's synthetic mean is 11.1 h against a 11.5 h claim - should be rejected.
    assert results["Bolt"].rejected


def test_missing_brand_raises():
    df = generate_sample_dataset(n_per_brand=30, seed=9)
    with pytest.raises(ValueError, match="fewer than two"):
        _tcb(df, claims={"Nonexistent": 10.0})


def test_bad_alpha_raises():
    with pytest.raises(ValueError):
        _tmc([1.0, 2.0], claim=10.0, alpha=0.0)


def test_bad_alternative_raises():
    with pytest.raises(ValueError, match="alternative"):
        _tmc([1.0, 2.0], claim=10.0, alternative="sideways")


def test_too_few_measurements_raise():
    with pytest.raises(ValueError, match="at least two"):
        _tmc([1.0], claim=10.0)
