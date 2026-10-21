import pandas as pd
import pytest

from smartphone_stats.data import generate_sample_dataset, load_dataset, validate_dataset
from smartphone_stats.descriptives import battery_life_anova, brand_summary


def test_generate_is_reproducible():
    a = generate_sample_dataset(n_per_brand=10, seed=7)
    b = generate_sample_dataset(n_per_brand=10, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_generate_shape_and_columns():
    df = generate_sample_dataset(n_per_brand=10, seed=1)
    assert len(df) == 40
    assert set(df["brand"]) == {"Apex", "Bolt", "Cosmos", "Delta"}
    assert set(df["customer_group"]) == {"Student", "Professional", "Senior"}
    assert (df.filter(like="hours").select_dtypes("number") >= 0).all().all()


def test_bundled_sample_loads_and_validates():
    df = load_dataset()
    validate_dataset(df)
    assert len(df) >= 100
    assert df["preference"].nunique() == 4


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.csv")


def test_brand_summary_has_one_row_per_brand():
    df = generate_sample_dataset(n_per_brand=15, seed=3)
    summary = brand_summary(df)
    assert list(summary.index) == ["Apex", "Bolt", "Cosmos", "Delta"]
    assert (summary["n"] == 15).all()
    assert "battery_life_hours__mean" in summary.columns
    assert "charging_time_minutes__std" in summary.columns


def test_anova_detects_brand_differences():
    df = generate_sample_dataset(n_per_brand=30, seed=5)
    result = battery_life_anova(df)
    assert result["n_groups"] == 4
    # Synthesised brands have genuinely different battery lives.
    assert result["p_value"] < 0.05
    assert result["f_statistic"] > 0


def test_anova_requires_two_groups():
    df = generate_sample_dataset(n_per_brand=5, seed=5)
    with pytest.raises(ValueError, match="at least two"):
        battery_life_anova(df[df["brand"] == "Apex"])
