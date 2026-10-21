"""Task 1: descriptive model of the performance measurements per brand."""

from __future__ import annotations

import pandas as pd
from scipy import stats

from .models import MEASUREMENT_COLUMNS


def brand_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-brand descriptive statistics for every measured variable.

    One row per brand; columns are ``n`` plus ``<variable>__<stat>`` pairs.
    """
    grouped = df.groupby("brand", observed=True)
    summary = grouped[list(MEASUREMENT_COLUMNS)].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{var}__{stat}" for var, stat in summary.columns]
    summary.insert(0, "n", grouped.size())
    return summary.round(3)


def battery_life_anova(df: pd.DataFrame) -> dict:
    """One-way ANOVA: do mean battery lives differ across brands?

    Returns the F statistic, p-value, and group means. Requires >= 2 brands.
    """
    groups = [
        part["battery_life_hours"].to_numpy()
        for _, part in df.groupby("brand", observed=True)
    ]
    if len(groups) < 2:
        raise ValueError("ANOVA needs at least two brands")
    if any(len(g) < 2 for g in groups):
        raise ValueError("each brand needs at least two measurements for ANOVA")
    f_stat, p_value = stats.f_oneway(*groups)
    means = df.groupby("brand", observed=True)["battery_life_hours"].mean().round(3)
    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "group_means": means.to_dict(),
        "n_groups": len(groups),
        "n_total": int(sum(len(g) for g in groups)),
    }
