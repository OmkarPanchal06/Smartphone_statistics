"""Loading, validation and reproducible generation of the survey dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .models import MEASUREMENT_COLUMNS, REQUIRED_COLUMNS

SAMPLE_CSV = Path(__file__).parent / "data" / "sample_smartphones.csv"

BRANDS = ("Apex", "Bolt", "Cosmos", "Delta")
CUSTOMER_GROUPS = ("Student", "Professional", "Senior")

# Mean battery life / charging time / processing score per brand (population
# parameters used to synthesise the bundled sample data).
_BRAND_PARAMS = {
    "Apex": dict(battery=12.4, charging=62.0, score=1120),
    "Bolt": dict(battery=11.1, charging=48.0, score=1010),
    "Cosmos": dict(battery=13.2, charging=78.0, score=1240),
    "Delta": dict(battery=10.3, charging=41.0, score=940),
}

# P(preferred brand | customer group) - preferences genuinely vary by group so
# the chi-square test of independence has a real effect to detect.
_GROUP_PREFERENCE_PROBS = {
    "Student": (0.20, 0.45, 0.15, 0.20),
    "Professional": (0.42, 0.18, 0.28, 0.12),
    "Senior": (0.30, 0.14, 0.36, 0.20),
}


def generate_sample_dataset(n_per_brand: int = 60, seed: int = 42) -> pd.DataFrame:
    """Synthesise a reproducible survey dataset (the bundled CSV is built from this)."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for brand in BRANDS:
        params = _BRAND_PARAMS[brand]
        groups = np.array(CUSTOMER_GROUPS)[
            rng.integers(0, len(CUSTOMER_GROUPS), size=n_per_brand)
        ]
        prefs = np.empty(n_per_brand, dtype=object)
        for group, prob in _GROUP_PREFERENCE_PROBS.items():
            mask = groups == group
            count = int(mask.sum())
            if count:
                prefs[mask] = rng.choice(BRANDS, size=count, p=prob)
        score = rng.normal(params["score"], 85.0, n_per_brand)
        charging = rng.normal(params["charging"], 7.0, n_per_brand)
        # Battery life responds to charging speed and processing performance.
        battery = (
            params["battery"]
            + 0.008 * (score - 1080)
            - 0.050 * (charging - 57.0)
            + rng.normal(0.0, 0.55, n_per_brand)
        )
        for k in range(n_per_brand):
            rows.append(
                {
                    "brand": brand,
                    "battery_life_hours": round(float(battery[k]), 2),
                    "charging_time_minutes": round(float(charging[k]), 1),
                    "processing_score": round(float(score[k]), 1),
                    "customer_group": str(groups[k]),
                    "preference": str(prefs[k]),
                }
            )

    return pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and value ranges; returns the frame unchanged if valid."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    if len(df) == 0:
        raise ValueError("dataset is empty")
    for col in MEASUREMENT_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            raise ValueError(f"column {col!r} contains non-numeric or missing values")
        if (values < 0).any():
            raise ValueError(f"column {col!r} contains negative values")
    for col in ("brand", "customer_group", "preference"):
        if df[col].isna().any() or (df[col].astype(str).str.strip() == "").any():
            raise ValueError(f"column {col!r} contains empty values")
    return df


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate a survey CSV; defaults to the bundled sample data."""
    csv_path = Path(path) if path is not None else SAMPLE_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return validate_dataset(df)
