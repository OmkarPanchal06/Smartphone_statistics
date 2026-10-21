"""Task 5: do smartphone preferences differ across customer groups?

Chi-square test of independence between the preferred brand and the customer
group, with effect size (Cramer's V) and standardised residuals for the cells
that drive the association.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class PreferenceTest:
    """Result of the chi-square test of preference x customer group."""

    chi2_statistic: float
    p_value: float
    dof: int
    n: int
    cramers_v: float
    contingency: pd.DataFrame
    expected: pd.DataFrame
    residuals: pd.DataFrame
    min_expected: float
    cells_below_5: int
    alpha: float

    @property
    def independent(self) -> bool:
        """True when preference and customer group are not significantly associated."""
        return self.p_value >= self.alpha

    @property
    def verdict(self) -> str:
        if self.independent:
            return (
                "no significant association between preference and customer group "
                f"(p={self.p_value:.4g}); preferences look alike across groups"
            )
        return (
            "preferences differ significantly across customer groups "
            f"(p={self.p_value:.4g}, Cramer's V={self.cramers_v:.3f})"
        )


def test_preference_association(df: pd.DataFrame, alpha: float = 0.05) -> PreferenceTest:
    """Chi-square test of independence: brand preference vs customer group."""
    needed = ("preference", "customer_group")
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")

    contingency = pd.crosstab(df["customer_group"], df["preference"])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        raise ValueError("need at least two customer groups and two preferred brands")

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    n = int(contingency.values.sum())
    min_dim = min(contingency.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * min_dim)))

    residuals = (contingency.values - expected) / np.sqrt(expected)
    cells_below_5 = int((expected < 5).sum())

    return PreferenceTest(
        chi2_statistic=float(chi2),
        p_value=float(p_value),
        dof=int(dof),
        n=n,
        cramers_v=cramers_v,
        contingency=contingency,
        expected=pd.DataFrame(expected, index=contingency.index, columns=contingency.columns),
        residuals=pd.DataFrame(
            np.round(residuals, 3), index=contingency.index, columns=contingency.columns
        ),
        min_expected=float(expected.min()),
        cells_below_5=cells_below_5,
        alpha=alpha,
    )
