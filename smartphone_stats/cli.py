"""Command-line interface for the smartphone statistics system.

Examples
--------
    python -m smartphone_stats all
    python -m smartphone_stats estimate --brand Cosmos
    python -m smartphone_stats claim --brand Apex --claim 12.5 --alpha 0.01
    python -m smartphone_stats predict --charging 60 --score 1100 --brand Cosmos
    python -m smartphone_stats preference
"""

from __future__ import annotations

import argparse
import math

import pandas as pd

from .claims import DEFAULT_CLAIMS, test_claims_by_brand
from .data import load_dataset
from .descriptives import battery_life_anova, brand_summary
from .estimation import estimate_by_brand, mean_battery_life
from .preferences import test_preference_association
from .prediction import fit_battery_model


def _fmt_p(p: float) -> str:
    return "<1e-16" if p <= 0 else f"{p:.4g}"


def _fmt_ci(lower: float, upper: float) -> str:
    low = "-inf" if lower == -math.inf else f"{lower:.2f}"
    high = "inf" if upper == math.inf else f"{upper:.2f}"
    return f"[{low}, {high}]"


def _report_descriptives(df: pd.DataFrame) -> None:
    print("== Performance measurements by brand ==")
    print(brand_summary(df).to_string())
    anova = battery_life_anova(df)
    print(
        f"\nOne-way ANOVA of battery life across brands: F={anova['f_statistic']:.3f}, "
        f"p={_fmt_p(anova['p_value'])} "
        f"(k={anova['n_groups']} brands, n={anova['n_total']})"
    )
    print(
        "Interpretation: brand means "
        + ("differ significantly." if anova["p_value"] < 0.05 else "are not significantly different.")
    )


def _report_estimation(df: pd.DataFrame, confidence: float, brand: str | None) -> None:
    print(f"== Average battery life ({confidence:.0%} confidence) ==")
    if brand is None:
        estimates = estimate_by_brand(df, confidence)
    else:
        part = df.loc[df["brand"] == brand, "battery_life_hours"]
        if part.empty:
            raise SystemExit(f"brand {brand!r} not present in the dataset")
        estimates = {str(brand): mean_battery_life(part, confidence)}
    ci_label = f"{confidence:.0%} CI"
    print(f"{'group':<14}{'n':>5}{'mean':>9}{'sd':>8}{ci_label:>22}")
    for name, est in estimates.items():
        ci = _fmt_ci(est.lower, est.upper)
        print(f"{name:<14}{est.n:>5}{est.mean:>9.2f}{est.std:>8.2f}{ci:>22}")


def _report_claims(df: pd.DataFrame, alpha: float, brand: str | None, claim: float | None) -> None:
    print(f"== Manufacturer claim tests (alpha={alpha}) ==")
    if brand is not None and claim is not None:
        from .claims import test_manufacturer_claim

        part = df.loc[df["brand"] == brand, "battery_life_hours"]
        if part.empty:
            raise SystemExit(f"brand {brand!r} not present in the dataset")
        tests = {
            str(brand): test_manufacturer_claim(
                part, claim=claim, alpha=alpha, brand=str(brand)
            )
        }
    else:
        claims = DEFAULT_CLAIMS
        if claim is not None:
            claims = {str(b): claim for b in df["brand"].unique()}
        tests = test_claims_by_brand(df, claims=claims, alpha=alpha)
    for name, test in tests.items():
        print(
            f"{name:<9} claim={test.claimed_hours:.1f} h  mean={test.sample_mean:.2f} h  "
            f"t={test.t_statistic:.3f}  p={_fmt_p(test.p_value)}  "
            f"CI={_fmt_ci(test.ci_lower, test.ci_upper)}"
        )
        print(f"          {test.verdict}")


def _report_prediction(
    df: pd.DataFrame,
    charging: float | None,
    score: float | None,
    interval: float,
    brand: str | None = None,
) -> None:
    print("== Battery-life prediction model ==")
    model = fit_battery_model(df)
    controls = " + brand fixed effects" if model.brand_levels else ""
    print(
        f"OLS on n={model.n}: battery_life ~ charging_time_minutes + "
        f"processing_score{controls} (features centred)"
    )
    for name, coef in model.coefficients.items():
        print(
            f"  {name:<24} {coef.estimate:>8.4f}  se={coef.std_error:.4f}  "
            f"t={coef.t_statistic:>6.3f}  p={_fmt_p(coef.p_value)}"
        )
    print(
        f"R^2={model.r_squared:.3f}  adj R^2={model.adj_r_squared:.3f}  "
        f"RMSE={model.rmse:.3f} h  F={model.f_statistic:.2f} p={_fmt_p(model.f_p_value)}"
    )
    if charging is not None and score is not None:
        result = model.predict(charging, score, brand=brand, interval=interval)
        label = f", brand={brand}" if brand else ""
        print(
            f"\nPrediction for charging={charging:.0f} min, score={score:.0f}{label}: "
            f"{result['predicted_hours']:.2f} h "
            f"({_fmt_ci(result['interval_low'], result['interval_high'])} "
            f"{interval:.0%} prediction interval)"
        )


def _report_preferences(df: pd.DataFrame, alpha: float) -> None:
    print(f"== Preference vs customer group (chi-square, alpha={alpha}) ==")
    result = test_preference_association(df, alpha=alpha)
    print("Contingency (observed counts):")
    print(result.contingency.to_string())
    print(
        f"\nchi2={result.chi2_statistic:.3f}  dof={result.dof}  p={_fmt_p(result.p_value)}  "
        f"Cramer's V={result.cramers_v:.3f}  min expected={result.min_expected:.1f}"
    )
    if result.cells_below_5:
        print(
            f"Warning: {result.cells_below_5} expected cell count(s) below 5 - "
            "chi-square approximation may be unreliable."
        )
    print("Standardised residuals (|z| > 2 flags a driving cell):")
    print(result.residuals.to_string())
    print(f"\nInterpretation: {result.verdict}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smartphone_stats",
        description="Statistical analysis of smartphone battery life, performance and preferences.",
    )
    parser.add_argument("--csv", default=None, help="path to a survey CSV (default: bundled sample)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("describe", help="per-brand descriptive statistics and ANOVA")

    p_est = sub.add_parser("estimate", help="average battery life with confidence interval")
    p_est.add_argument("--brand", default=None, help="restrict to one brand")
    p_est.add_argument("--confidence", type=float, default=0.95)

    p_claim = sub.add_parser("claim", help="test manufacturers' battery-life claims")
    p_claim.add_argument("--brand", default=None, help="restrict to one brand")
    p_claim.add_argument("--claim", type=float, default=None, help="claimed hours (per brand default)")
    p_claim.add_argument("--alpha", type=float, default=0.05)

    p_pred = sub.add_parser("predict", help="fit the battery-life model and predict")
    p_pred.add_argument("--charging", type=float, default=None, help="charging time in minutes")
    p_pred.add_argument("--score", type=float, default=None, help="processing score")
    p_pred.add_argument("--brand", default=None, help="brand for the fixed effect (default: reference)")
    p_pred.add_argument("--interval", type=float, default=0.95)

    p_pref = sub.add_parser("preference", help="chi-square test of preferences across groups")
    p_pref.add_argument("--alpha", type=float, default=0.05)

    p_all = sub.add_parser("all", help="run every analysis")
    p_all.add_argument("--confidence", type=float, default=0.95)
    p_all.add_argument("--alpha", type=float, default=0.05)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    df = load_dataset(args.csv)

    try:
        _dispatch(args, df)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    return 0


def _dispatch(args: argparse.Namespace, df: pd.DataFrame) -> None:
    if args.command == "describe":
        _report_descriptives(df)
    elif args.command == "estimate":
        _report_estimation(df, args.confidence, args.brand)
    elif args.command == "claim":
        if args.brand is None and args.claim is not None:
            raise SystemExit("--claim with a custom value needs --brand (or omit --claim)")
        _report_claims(df, args.alpha, args.brand, args.claim)
    elif args.command == "predict":
        _report_prediction(df, args.charging, args.score, args.interval, args.brand)
    elif args.command == "preference":
        _report_preferences(df, args.alpha)
    elif args.command == "all":
        _report_descriptives(df)
        print()
        _report_estimation(df, args.confidence, None)
        print()
        _report_claims(df, args.alpha, None, None)
        print()
        _report_prediction(df, None, None, 0.95)
        print()
        _report_preferences(df, args.alpha)


if __name__ == "__main__":
    raise SystemExit(main())
