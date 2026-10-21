"""Task 2: estimate the average battery life with a confidence interval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class BatteryEstimate:
    """Point estimate and confidence interval for a mean battery life."""

    n: int
    mean: float
    std: float
    sem: float
    confidence: float
    df: int
    critical_value: float
    lower: float
    upper: float

    @property
    def margin(self) -> float:
        return self.upper - self.mean


def mean_battery_life(
    values: pd.Series | np.ndarray | list[float],
    confidence: float = 0.95,
) -> BatteryEstimate:
    """Estimate the population mean battery life via a Student-t interval."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n < 2:
        raise ValueError("at least two measurements are required to estimate a mean")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    sem = std / np.sqrt(n)
    df = n - 1
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, df))
    margin = critical * sem
    return BatteryEstimate(
        n=n,
        mean=mean,
        std=std,
        sem=sem,
        confidence=confidence,
        df=df,
        critical_value=critical,
        lower=mean - margin,
        upper=mean + margin,
    )


def estimate_by_brand(
    df: pd.DataFrame, confidence: float = 0.95
) -> dict[str, BatteryEstimate]:
    """Battery-life estimate for each brand, plus the pooled estimate."""
    estimates = {
        str(brand): mean_battery_life(part["battery_life_hours"], confidence)
        for brand, part in df.groupby("brand", observed=True)
    }
    estimates["All brands"] = mean_battery_life(df["battery_life_hours"], confidence)
    return estimates
