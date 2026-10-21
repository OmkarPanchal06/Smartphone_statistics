import pytest

from smartphone_stats.models import Measurement
from smartphone_stats.data import validate_dataset
import pandas as pd


def test_measurement_accepts_valid_record():
    m = Measurement(
        brand="Apex",
        battery_life_hours=12.4,
        charging_time_minutes=62.0,
        processing_score=1120.0,
        customer_group="Student",
        preference="Apex",
    )
    assert m.brand == "Apex"


@pytest.mark.parametrize("field,value", [
    ("battery_life_hours", -1.0),
    ("charging_time_minutes", -5.0),
    ("processing_score", float("nan")),
])
def test_measurement_rejects_bad_values(field, value):
    kwargs = dict(
        brand="Apex",
        battery_life_hours=12.0,
        charging_time_minutes=60.0,
        processing_score=1000.0,
    )
    kwargs[field] = value
    with pytest.raises(ValueError):
        Measurement(**kwargs)


def test_validate_dataset_rejects_missing_column():
    df = pd.DataFrame({"brand": ["Apex"], "battery_life_hours": [10.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(df)


def test_validate_dataset_rejects_negative_values():
    df = pd.DataFrame(
        {
            "brand": ["Apex"],
            "battery_life_hours": [-3.0],
            "charging_time_minutes": [60.0],
            "processing_score": [1000.0],
            "customer_group": ["Student"],
            "preference": ["Apex"],
        }
    )
    with pytest.raises(ValueError, match="negative"):
        validate_dataset(df)
