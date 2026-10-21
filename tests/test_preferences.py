import numpy as np
import pandas as pd
import pytest

from smartphone_stats import preferences
from smartphone_stats.data import generate_sample_dataset

# Underscore alias: pytest collects module-level ``test_*`` names.
_tpa = preferences.test_preference_association


def _independent_frame(groups=2000, seed=0):
    """Preference drawn from the same distribution for every group."""
    rng = np.random.default_rng(seed)
    group = rng.choice(["A", "B", "C"], size=groups * 3)
    pref = rng.choice(["Apex", "Bolt", "Cosmos"], size=groups * 3, p=[0.4, 0.35, 0.25])
    return pd.DataFrame({"customer_group": group, "preference": pref})


def test_detects_dependence_in_bundled_data():
    df = generate_sample_dataset(seed=42)
    result = _tpa(df)
    assert result.p_value < 0.05
    assert not result.independent
    assert "differ significantly" in result.verdict


def test_independent_data_not_rejected():
    result = _tpa(_independent_frame())
    assert result.p_value > 0.05
    assert result.independent
    assert "no significant association" in result.verdict


def test_effect_size_bounds():
    result = _tpa(generate_sample_dataset(seed=42))
    assert 0.0 <= result.cramers_v <= 1.0
    assert result.dof == (3 - 1) * (4 - 1)
    assert result.n == result.contingency.values.sum()


def test_tables_are_consistent():
    result = _tpa(generate_sample_dataset(seed=42))
    assert result.contingency.shape == result.expected.shape == result.residuals.shape
    assert result.min_expected == pytest.approx(result.expected.values.min())


def test_too_few_groups_raises():
    df = pd.DataFrame(
        {
            "customer_group": ["A", "A", "A", "A"],
            "preference": ["Apex", "Bolt", "Apex", "Bolt"],
        }
    )
    with pytest.raises(ValueError, match="at least two customer groups"):
        _tpa(df)


def test_missing_column_raises():
    with pytest.raises(ValueError, match="missing required columns"):
        _tpa(pd.DataFrame({"preference": ["Apex"]}))


def test_deterministic_result():
    df = generate_sample_dataset(seed=42)
    a = _tpa(df)
    b = _tpa(df)
    assert a.chi2_statistic == np.float64(b.chi2_statistic)
    assert a.p_value == b.p_value
