"""Interactive Streamlit dashboard for the smartphone statistics system.

Launch with:
    C:\\Users\\omkar\\diabetes_venv\\Scripts\\python.exe -m streamlit run app.py

Works on old and new Streamlit releases: modern builds (>= 1.50) get the
polished UI (accessible chart descriptions, alert titles, material icons,
bordered cards); older releases get the same analyses and charts through
long-stable APIs. Set SMARTPHONE_STATS_LEGACY_UI=1 to force the legacy path.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from smartphone_stats import (
    DEFAULT_CLAIMS,
    battery_life_anova,
    brand_summary,
    estimate_by_brand,
    fit_battery_model,
    load_dataset,
    mean_battery_life,
    test_claims_by_brand,
    test_manufacturer_claim,
    test_preference_association,
    validate_dataset,
)


# ------------------------------------------------- streamlit version compat


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for piece in version.split(".")[:2]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


_LEGACY = os.environ.get("SMARTPHONE_STATS_LEGACY_UI") == "1"
_MODERN = (not _LEGACY) and _version_tuple(st.__version__) >= (1, 50)


def _show_df(data, *, alt: str = "", hide_index: bool = False) -> None:
    if _MODERN:
        st.dataframe(data, hide_index=hide_index, width="stretch", alt=alt)
    else:
        st.dataframe(data, hide_index=hide_index)


def _alert(kind: str, body: str, *, headline: str = "", icon: str = "") -> None:
    fn = {"success": st.success, "info": st.info, "error": st.error, "warning": st.warning}[kind]
    if _MODERN:
        fn(body, title=headline, icon=icon)
    else:
        fn(f"**{headline}** — {body}" if headline else body)


def _bar(data, *, alt: str = "", x_label: str = "", y_label: str = "") -> None:
    if _MODERN:
        st.bar_chart(data, x_label=x_label, y_label=y_label, alt=alt)
    else:
        st.bar_chart(data)


def _scatter(data, *, alt: str = "", x_label: str = "", y_label: str = "") -> None:
    if hasattr(st, "scatter_chart") and _MODERN:
        st.scatter_chart(
            data,
            x="processing_score",
            y="battery_life_hours",
            color="brand",
            x_label=x_label,
            y_label=y_label,
            alt=alt,
        )
    elif hasattr(st, "scatter_chart"):
        st.scatter_chart(data, x="processing_score", y="battery_life_hours", color="brand")
    else:  # very old Streamlit without scatter charts: fall back to a table
        st.dataframe(data[["processing_score", "battery_life_hours", "brand"]].head(20))


def _card():
    """Bordered container on modern Streamlit, plain container elsewhere."""
    return st.container(border=True) if _MODERN else st.container()


def _metric_row(items: list[tuple[str, str]]) -> None:
    """Responsive row of bordered metric cards (columns on older Streamlit)."""
    if _MODERN:
        with st.container(horizontal=True):
            for label, value in items:
                st.metric(label, value, border=True)
    else:
        columns = st.columns(len(items))
        for column, (label, value) in zip(columns, items):
            column.metric(label, value)


st.set_page_config(
    page_title="Smartphone statistics dashboard",
    page_icon="📱",
    layout="wide",
)

# ---------------------------------------------------------------- data loading


@st.cache_data(show_spinner=False, max_entries=10)
def _load_default() -> pd.DataFrame:
    return load_dataset()


@st.cache_data(show_spinner=False, max_entries=10)
def _fit(df: pd.DataFrame):
    return fit_battery_model(df)


def get_dataset() -> pd.DataFrame:
    """Bundled sample data, or a CSV the presenter uploaded in the sidebar."""
    uploaded = st.session_state.get("uploaded_df")
    if uploaded is not None:
        return uploaded
    return _load_default()


# ---------------------------------------------------------------- sidebar

if _MODERN:
    st.sidebar.title("Smartphone stats", icon=":material/smartphone:")
else:
    st.sidebar.title("📱 Smartphone stats")
st.sidebar.caption("Statistical comparison of smartphone brands")

source = st.sidebar.radio(
    "Data source",
    ["Bundled sample (240 rows)", "Upload CSV"],
)
if source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Survey CSV", type="csv")
    if uploaded_file is not None:
        try:
            st.session_state["uploaded_df"] = validate_dataset(pd.read_csv(uploaded_file))
        except ValueError as exc:
            st.sidebar.error(f"Invalid CSV: {exc}")
            st.session_state.pop("uploaded_df", None)
else:
    st.session_state.pop("uploaded_df", None)

confidence = st.sidebar.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)
alpha = st.sidebar.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)

df = get_dataset()
brands = sorted(df["brand"].unique())
groups = sorted(df["customer_group"].unique())

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**{len(df)}** surveyed units · **{len(brands)}** brands · **{len(groups)}** customer groups"
)

# ---------------------------------------------------------------- tabs

_TABS = [
    (":material/dashboard: Overview", "📊 Overview"),
    (":material/battery_charging_full: Battery life", "🔋 Battery life"),
    (":material/fact_check: Claim tests", "✅ Claim tests"),
    (":material/insights: Prediction", "🤖 Prediction"),
    (":material/groups: Preferences", "👥 Preferences"),
]
tab_overview, tab_battery, tab_claims, tab_predict, tab_prefs = st.tabs(
    [modern if _MODERN else plain for modern, plain in _TABS]
)

# ---------------------------------------------------------------- overview tab

with tab_overview:
    st.subheader("Dataset overview")
    _metric_row(
        [
            ("Surveyed units", str(len(df))),
            ("Brands", str(len(brands))),
            ("Customer groups", str(len(groups))),
            ("Avg battery life", f"{df['battery_life_hours'].mean():.2f} h"),
        ]
    )

    st.markdown("### Performance measurements by brand")
    _show_df(brand_summary(df), alt="Per-brand descriptive statistics")

    anova = battery_life_anova(df)
    anova_line = (
        f"F = {anova['f_statistic']:.2f}, p = {anova['p_value']:.2e} "
        f"(k = {anova['n_groups']} brands, n = {anova['n_total']})"
    )
    if anova["p_value"] < 0.05:
        _alert(
            "success",
            "Brand mean battery lives differ significantly.",
            headline=f"One-way ANOVA — {anova_line}",
            icon=":material/check_circle:",
        )
    else:
        _alert(
            "info",
            "No significant difference between brand means.",
            headline=f"One-way ANOVA — {anova_line}",
            icon=":material/info:",
        )

    with st.expander("Raw data (first 20 rows)"):
        _show_df(df.head(20), alt="First 20 survey rows", hide_index=True)

# ---------------------------------------------------------------- battery life tab

with tab_battery:
    st.subheader("Estimating the average battery life")

    estimates = estimate_by_brand(df, confidence)
    est_table = pd.DataFrame(
        {
            "n": [e.n for e in estimates.values()],
            "Mean (h)": [e.mean for e in estimates.values()],
            "Std dev": [e.std for e in estimates.values()],
            "Lower": [e.lower for e in estimates.values()],
            "Upper": [e.upper for e in estimates.values()],
        },
        index=list(estimates),
    )
    est_table["CI width"] = est_table["Upper"] - est_table["Lower"]
    _show_df(est_table.round(3), alt="Battery life estimates by brand")

    with _card():
        st.markdown(f"### Mean battery life with {confidence:.0%} confidence interval")
        chart_df = est_table.loc[brands, ["Lower", "Mean (h)", "Upper"]]
        _bar(
            chart_df,
            x_label="Brand",
            y_label="Battery life (hours)",
            alt="Mean battery life per brand with confidence bounds",
        )

    st.markdown("### Drill down: single sample")
    pick = st.selectbox("Brand", ["All brands", *brands])
    sample = (
        df["battery_life_hours"]
        if pick == "All brands"
        else df.loc[df["brand"] == pick, "battery_life_hours"]
    )
    est = mean_battery_life(sample, confidence)
    st.write(
        f"**{pick}** — point estimate **{est.mean:.2f} h** "
        f"({confidence:.0%} CI [{est.lower:.2f}, {est.upper:.2f}], "
        f"n = {est.n}, sd = {est.std:.2f})"
    )

# ---------------------------------------------------------------- claims tab

with tab_claims:
    st.subheader("Testing manufacturers' claims")
    st.caption(
        "One-sample t-test per brand: H₀ — the true mean battery life equals the "
        "advertised claim. Claims are rejected when p < α."
    )

    tests = test_claims_by_brand(df, claims=DEFAULT_CLAIMS, alpha=alpha)
    for brand, test in tests.items():
        details = (
            f"Claim {test.claimed_hours:.1f} h vs measured {test.sample_mean:.2f} h — "
            f"t = {test.t_statistic:.2f}, p = {test.p_value:.4g}"
        )
        if test.rejected:
            direction = "below" if test.sample_mean < test.claimed_hours else "above"
            _alert(
                "error",
                f"{details}. Measured average is significantly **{direction}** the claim.",
                headline=f"{brand}: claim rejected",
                icon=":material/cancel:",
            )
        else:
            _alert(
                "success",
                f"{details}. Insufficient evidence to reject the claim.",
                headline=f"{brand}: claim consistent with the data",
                icon=":material/check_circle:",
            )

    with _card():
        st.markdown("### Claimed vs measured")
        compare = pd.DataFrame(
            {
                "Measured mean": [t.sample_mean for t in tests.values()],
                "Claimed": [t.claimed_hours for t in tests.values()],
            },
            index=list(tests),
        )
        _bar(
            compare,
            x_label="Brand",
            y_label="Battery life (hours)",
            alt="Measured mean battery life versus the advertised claim, by brand",
        )

    with st.expander("Test a custom claim"):
        custom_brand = st.selectbox("Brand", brands, key="custom_claim_brand")
        custom_claim = st.number_input(
            "Claimed battery life (hours)",
            value=float(DEFAULT_CLAIMS[custom_brand]),
            step=0.1,
        )
        custom_alpha = st.slider("α", 0.01, 0.10, alpha, 0.01, key="custom_claim_alpha")
        custom = test_manufacturer_claim(
            df.loc[df["brand"] == custom_brand, "battery_life_hours"],
            claim=custom_claim,
            alpha=custom_alpha,
            brand=custom_brand,
        )
        st.write(
            f"t = {custom.t_statistic:.3f}, p = {custom.p_value:.4g}, "
            f"{custom_alpha:.0%} CI [{custom.ci_lower:.2f}, {custom.ci_upper:.2f}] — "
            f"{custom.verdict}"
        )

# ---------------------------------------------------------------- prediction tab

with tab_predict:
    st.subheader("Predicting battery performance")
    model = _fit(df)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("### Adjust the inputs")
        charging = st.slider(
            "Charging time (minutes)",
            float(df["charging_time_minutes"].min()),
            float(df["charging_time_minutes"].max()),
            float(df["charging_time_minutes"].mean()),
            step=1.0,
        )
        score = st.slider(
            "Processing score",
            float(df["processing_score"].min()),
            float(df["processing_score"].max()),
            float(df["processing_score"].mean()),
            step=10.0,
        )
        interval = st.slider("Prediction interval", 0.80, 0.99, 0.95, 0.01)
        brand = st.selectbox("Brand effect", ["None (reference)", *model.brand_levels])
        brand_arg = None if brand.startswith("None") else brand

        result = model.predict(charging, score, brand=brand_arg, interval=interval)
        st.metric(
            "Predicted battery life",
            f"{result['predicted_hours']:.2f} h",
            f"{interval:.0%} CI [{result['interval_low']:.2f}, {result['interval_high']:.2f}]",
        )

    with right:
        st.markdown("### Model fit")
        _metric_row(
            [
                ("R²", f"{model.r_squared:.3f}"),
                ("Adj. R²", f"{model.adj_r_squared:.3f}"),
                ("RMSE", f"{model.rmse:.3f} h"),
                ("p (F-test)", f"{model.f_p_value:.2e}"),
            ]
        )

        coef_table = pd.DataFrame(
            {
                "Estimate": [c.estimate for c in model.coefficients.values()],
                "Std error": [c.std_error for c in model.coefficients.values()],
                "t": [c.t_statistic for c in model.coefficients.values()],
                "p": [c.p_value for c in model.coefficients.values()],
            },
            index=list(model.coefficients),
        )
        _show_df(coef_table.round(4), alt="Regression coefficients")

    with _card():
        st.markdown("### Observed battery life vs processing score")
        st.caption(
            "Each dot is a surveyed unit. The regression separates within-brand "
            "slopes from between-brand differences (brand fixed effects)."
        )
        _scatter(
            df,
            x_label="Processing score",
            y_label="Battery life (hours)",
            alt="Battery life versus processing score, colored by brand",
        )

# ---------------------------------------------------------------- preferences tab

with tab_prefs:
    st.subheader("Do preferences differ across customer groups?")
    pref = test_preference_association(df, alpha=alpha)

    st.markdown("### Observed preference counts")
    _show_df(pref.contingency, alt="Preference counts by customer group")

    _metric_row(
        [
            ("χ² statistic", f"{pref.chi2_statistic:.2f}"),
            ("p-value", f"{pref.p_value:.2e}"),
            ("Cramér's V", f"{pref.cramers_v:.3f}"),
            ("dof", str(pref.dof)),
        ]
    )

    if pref.independent:
        _alert(
            "info",
            "Preferences look alike across groups.",
            headline=f"No significant association at α = {alpha}",
            icon=":material/info:",
        )
    else:
        _alert(
            "error",
            "Preferences differ significantly between customer groups.",
            headline=f"Significant association at α = {alpha}",
            icon=":material/cancel:",
        )

    with _card():
        st.markdown("### Preference share by customer group")
        share = pref.contingency.div(pref.contingency.sum(axis=1), axis=0)
        _bar(
            share,
            x_label="Customer group",
            y_label="Share of preferences",
            alt="Preferred brand share within each customer group",
        )

    st.markdown("### Standardised residuals")
    st.caption("Cells with |z| > 2 drive the association.")
    _show_df(pref.residuals, alt="Standardised chi-square residuals")

    if pref.cells_below_5:
        _alert(
            "warning",
            "The chi-square approximation may be unreliable.",
            headline=f"{pref.cells_below_5} expected cell count(s) below 5",
            icon=":material/warning:",
        )
