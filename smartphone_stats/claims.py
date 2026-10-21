"""Task 3: test manufacturers' battery-life claims against measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# Advertised battery life (hours) each manufacturer claims for its own brand.
DEFAULT_CLAIMS = {
    "Apex": 12.0,
    "Bolt": 11.5,
    "Cosmos": 13.0,
    "Delta": 11.0,
}


@dataclass(frozen=True)
class ClaimTest:
    """Result of a one-sample t-test of a manufacturer's claim."""

    brand: str
    claimed_hours: float
    n: int
    sample_mean: float
    t_statistic: float
    p_value: float
    confidence: float
    ci_lower: float
    ci_upper: float
    alpha: float
    alternative: str

    @property
    def rejected(self) -> bool:
        """True when the claim is statistically contradicted at level alpha."""
        return self.p_value < self.alpha

    @property
    def verdict(self) -> str:
        if self.rejected:
            direction = "below" if self.sample_mean < self.claimed_hours else "above"
            return (
                f"claim of {self.claimed_hours:.1f} h is rejected - measured "
                f"average is significantly {direction} the claim (p={self.p_value:.4g})"
            )
        return (
            f"claim of {self.claimed_hours:.1f} h is consistent with the data "
            f"(p={self.p_value:.4g}); insufficient evidence to reject"
        )


def test_manufacturer_claim(
    values: pd.Series | np.ndarray,
    claim: float,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    brand: str = "brand",
) -> ClaimTest:
    """One-sample t-test: is the measured mean battery life equal to the claim?

    ``alternative`` is one of 'two-sided', 'greater', 'less'. The confidence
    interval reported is the (1-alpha) interval matching the alternative.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")

    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n < 2:
        raise ValueError("at least two measurements are required for a t-test")

    sample_mean = float(arr.mean())
    if float(arr.std(ddof=1)) == 0.0:
        # Zero variance: scipy returns nan; resolve the degenerate case exactly.
        if sample_mean == claim:
            t_stat, p_value = 0.0, 1.0
        else:
            above = sample_mean > claim
            t_stat = float("inf") if above else float("-inf")
            p_value = {
                "two-sided": 0.0,
                "greater": 0.0 if above else 1.0,
                "less": 1.0 if above else 0.0,
            }[alternative]
    else:
        t_stat, p_value = stats.ttest_1samp(arr, popmean=claim, alternative=alternative)
    # CI level/shape must match the alternative hypothesis.
    tail = {"two-sided": (1 + (1 - alpha)) / 2, "greater": 1 - alpha, "less": 1 - alpha}
    critical = float(stats.t.ppf(tail[alternative], n - 1))
    sem = float(arr.std(ddof=1)) / np.sqrt(n)
    if alternative == "two-sided":
        ci_lower, ci_upper = sample_mean - critical * sem, sample_mean + critical * sem
    elif alternative == "greater":
        ci_lower, ci_upper = sample_mean - critical * sem, np.inf
    else:
        ci_lower, ci_upper = -np.inf, sample_mean + critical * sem

    return ClaimTest(
        brand=brand,
        claimed_hours=float(claim),
        n=n,
        sample_mean=sample_mean,
        t_statistic=float(t_stat),
        p_value=float(p_value),
        confidence=1 - alpha,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        alpha=alpha,
        alternative=alternative,
    )


def test_claims_by_brand(
    df: pd.DataFrame,
    claims: dict[str, float] | None = None,
    alpha: float = 0.05,
) -> dict[str, ClaimTest]:
    """Test every manufacturer's claim against the measurements of its brand."""
    claims = DEFAULT_CLAIMS if claims is None else claims
    results: dict[str, ClaimTest] = {}
    for brand, claim in claims.items():
        part = df.loc[df["brand"] == brand, "battery_life_hours"]
        if len(part) < 2:
            raise ValueError(f"brand {brand!r} has fewer than two measurements")
        results[brand] = test_manufacturer_claim(
            part, claim=claim, alpha=alpha, brand=str(brand)
        )
    return results
