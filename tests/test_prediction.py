import numpy as np
import pandas as pd
import pytest

from smartphone_stats.data import generate_sample_dataset
from smartphone_stats.prediction import fit_battery_model


def _exact_df(n=50, seed=0):
    """Synthetic data with zero noise so coefficients are recoverable exactly."""
    rng = np.random.default_rng(seed)
    charging = rng.uniform(40, 80, n)
    score = rng.uniform(900, 1300, n)
    battery = 10.0 + 0.02 * charging + 0.001 * score
    return pd.DataFrame(
        {
            "battery_life_hours": battery,
            "charging_time_minutes": charging,
            "processing_score": score,
        }
    )


def test_recovers_exact_coefficients_on_noise_free_data():
    model = fit_battery_model(_exact_df())
    # With centred features, the intercept equals the mean response at mean x.
    y_mean = _exact_df()["battery_life_hours"].mean()
    assert model.coefficients["intercept"].estimate == pytest.approx(y_mean)
    assert model.coefficients["charging_time_minutes"].estimate == pytest.approx(0.02)
    assert model.coefficients["processing_score"].estimate == pytest.approx(0.001)
    assert model.r_squared == pytest.approx(1.0, abs=1e-9)


def test_fits_sample_data_sensibly():
    df = generate_sample_dataset(n_per_brand=60, seed=42)
    model = fit_battery_model(df)
    assert model.n == len(df)
    assert 0.0 < model.r_squared <= 1.0
    assert model.adj_r_squared <= model.r_squared
    assert model.rmse > 0
    # Brand effects separate the confounded within-brand slopes.
    assert model.coefficients["processing_score"].estimate > 0
    assert model.coefficients["charging_time_minutes"].estimate < 0
    assert model.reference_brand == "Apex"
    assert "brand[Cosmos]" in model.coefficients


def test_brand_effects_absorb_brand_level_differences():
    df = generate_sample_dataset(seed=42)
    model = fit_battery_model(df)
    # Same phone specs, different brand: Cosmos carries a positive brand premium.
    apex = model.predict(60.0, 1100.0, brand="Apex")
    cosmos = model.predict(60.0, 1100.0, brand="Cosmos")
    assert cosmos["predicted_hours"] > apex["predicted_hours"]


def test_unknown_brand_raises():
    model = fit_battery_model(generate_sample_dataset(seed=42))
    with pytest.raises(ValueError, match="unknown brand"):
        model.predict(60.0, 1100.0, brand="Nope")


def test_prediction_interval_contains_point_estimate():
    model = fit_battery_model(generate_sample_dataset(seed=42))
    result = model.predict(60.0, 1100.0)
    assert result["interval_low"] < result["predicted_hours"] < result["interval_high"]


def test_interval_widens_away_from_the_data_centre():
    df = generate_sample_dataset(seed=42)
    model = fit_battery_model(df)
    centre_x = df["charging_time_minutes"].mean()
    centre_s = df["processing_score"].mean()
    near = model.predict(centre_x, centre_s)
    far = model.predict(
        df["charging_time_minutes"].max() + 20,
        df["processing_score"].max() + 300,
    )
    near_width = near["interval_high"] - near["interval_low"]
    far_width = far["interval_high"] - far["interval_low"]
    assert far_width > near_width


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="missing required columns"):
        fit_battery_model(pd.DataFrame({"battery_life_hours": [1.0, 2.0, 3.0]}))


def test_too_few_rows_raise():
    df = _exact_df(n=3)
    with pytest.raises(ValueError, match="need more"):
        fit_battery_model(df)


def test_bad_interval_raises():
    model = fit_battery_model(_exact_df())
    with pytest.raises(ValueError):
        model.predict(60.0, 1100.0, interval=1.2)
