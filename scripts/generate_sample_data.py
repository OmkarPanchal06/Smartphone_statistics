"""Regenerate the bundled sample dataset (deterministic, seed=42)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smartphone_stats.data import SAMPLE_CSV, generate_sample_dataset, validate_dataset


def main() -> None:
    df = validate_dataset(generate_sample_dataset(n_per_brand=60, seed=42))
    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAMPLE_CSV, index=False)
    print(f"wrote {len(df)} rows to {SAMPLE_CSV}")


if __name__ == "__main__":
    main()
