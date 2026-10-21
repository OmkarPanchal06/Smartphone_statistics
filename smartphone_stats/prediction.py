"""Task 4: predict battery life from charging time and processing performance.

Ordinary least-squares regression of battery_life_hours on charging_time_minutes
and processing_score (mean-centred for conditioning), plus brand fixed effects
when a brand column is available. Charging time and processing score both vary
systematically between brands, so without brand controls their effects are
confounded; the dummies separate within-brand from between-brand variation.

Includes classical standard errors, t-tests on coefficients, and prediction
intervals that widen with leverage away from the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

FEATURES = ("charging_time_minutes", "processing_score")


@dataclass(frozen=True)
class Coefficient:
    name: str
    estimate: float
    std_error: float
    t_statistic: float
    p_value: float


@dataclass(frozen=True)
class BatteryModel:
    """Fitted linear model predicting battery life for one phone."""

    coefficients: dict[str, Coefficient]
    r_squared: float
    adj_r_squared: float
    rmse: float
    f_statistic: float
    f_p_value: float
    n: int
    n_params: int
    reference_brand: str | None
    _x_mean: np.ndarray = field(repr=False, compare=False)
    _beta: np.ndarray = field(repr=False, compare=False)
    _resid_sd: float = field(repr=False, compare=False)
    _xtx_pinv: np.ndarray = field(repr=False, compare=False)
    _brands: tuple[str, ...] = field(repr=False, compare=False)

    @property
    def brand_levels(self) -> tuple[str, ...]:
        """Brands the model was trained on (empty when no brand effects)."""
        return self._brands

    def predict(
        self,
        charging_time_minutes: float,
        processing_score: float,
        brand: str | None = None,
        interval: float = 0.95,
    ) -> dict[str, float]:
        """Predict battery life with a two-sided prediction interval.

        ``brand`` may name any brand the model was trained on; the reference
        brand (or ``None``) means "no brand effect added".
        """
        if not 0.0 < interval < 1.0:
            raise ValueError("interval must be strictly between 0 and 1")
        if brand is not None and self._brands and brand not in self._brands:
            raise ValueError(
                f"unknown brand {brand!r}; model knows {', '.join(self._brands)}"
            )

        features = np.array([charging_time_minutes, processing_score], dtype=float) - self._x_mean
        dummies = np.array(
            [1.0 if (b and b == brand) else 0.0 for b in self._brands[1:]]
        )
        design = np.concatenate([[1.0], features, dummies])[None, :]
        y_hat = float((design @ self._beta).item())

        leverage = float((design @ self._xtx_pinv @ design.T).item())
        pred_sd = self._resid_sd * np.sqrt(1.0 + leverage)
        dof = self.n - self.n_params
        critical = float(stats.t.ppf((1.0 + interval) / 2.0, dof))
        margin = critical * pred_sd
        return {
            "predicted_hours": y_hat,
            "interval_low": y_hat - margin,
            "interval_high": y_hat + margin,
            "interval": interval,
        }


def fit_battery_model(df: pd.DataFrame) -> BatteryModel:
    """Fit the OLS battery-life model on a validated dataset."""
    needed = ("battery_life_hours", *FEATURES)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")

    y = df["battery_life_hours"].to_numpy(dtype=float)
    x_raw = df[list(FEATURES)].to_numpy(dtype=float)
    if np.isnan(y).any() or np.isnan(x_raw).any():
        raise ValueError("model inputs contain missing values")
    n = y.size

    x_mean = x_raw.mean(axis=0)
    x = x_raw - x_mean

    names = ["intercept", *FEATURES]
    parts = [np.ones(n), x]
    brands: tuple[str, ...] = ()
    if "brand" in df.columns:
        brands = tuple(sorted(str(b) for b in df["brand"].unique()))
        if len(brands) > 1:
            brand_col = df["brand"].astype(str).to_numpy()
            # Drop-first encoding: first sorted brand is the reference level.
            for level in brands[1:]:
                parts.append((brand_col == level).astype(float))
                names.append(f"brand[{level}]")

    design = np.column_stack(parts)
    n_params = design.shape[1]
    if n <= n_params:
        raise ValueError(f"need more than {n_params} observations to fit the model")

    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ beta
    sse = float(residuals @ residuals)
    sst = float(((y - y.mean()) ** 2).sum())
    dof = n - n_params
    mse = sse / dof
    resid_sd = float(np.sqrt(mse))

    covariance = mse * np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.diag(covariance))
    t_vals = beta / se
    p_vals = 2.0 * stats.t.sf(np.abs(t_vals), dof)

    coefficients = {
        name: Coefficient(
            name=name,
            estimate=float(beta[i]),
            std_error=float(se[i]),
            t_statistic=float(t_vals[i]),
            p_value=float(p_vals[i]),
        )
        for i, name in enumerate(names)
    }

    r_squared = 1.0 - sse / sst if sst > 0 else 0.0
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / dof
    model_ss = sst - sse
    f_stat = (model_ss / (n_params - 1)) / mse if mse > 0 else 0.0
    f_p = float(stats.f.sf(f_stat, n_params - 1, dof))

    return BatteryModel(
        coefficients=coefficients,
        r_squared=float(r_squared),
        adj_r_squared=float(adj_r_squared),
        rmse=resid_sd,
        f_statistic=float(f_stat),
        f_p_value=f_p,
        n=n,
        n_params=n_params,
        reference_brand=brands[0] if brands else None,
        _x_mean=x_mean,
        _beta=beta,
        _resid_sd=resid_sd,
        _xtx_pinv=np.linalg.pinv(design.T @ design),
        _brands=brands,
    )
