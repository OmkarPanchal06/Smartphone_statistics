"""Data model for smartphone performance measurements.

Each record represents one surveyed unit: a tested smartphone whose battery
life, charging time and processing performance were measured, together with
the customer group of the respondent and the brand that respondent prefers.
"""

from __future__ import annotations

from dataclasses import dataclass

REQUIRED_COLUMNS = (
    "brand",
    "battery_life_hours",
    "charging_time_minutes",
    "processing_score",
    "customer_group",
    "preference",
)

MEASUREMENT_COLUMNS = (
    "battery_life_hours",
    "charging_time_minutes",
    "processing_score",
)


@dataclass(frozen=True)
class Measurement:
    """A single smartphone performance measurement."""

    brand: str
    battery_life_hours: float
    charging_time_minutes: float
    processing_score: float
    customer_group: str = "unknown"
    preference: str = "unknown"

    def __post_init__(self) -> None:
        for field in MEASUREMENT_COLUMNS:
            value = getattr(self, field)
            if not isinstance(value, (int, float)) or value != value:  # NaN check
                raise ValueError(f"{field} must be a real number, got {value!r}")
            if value < 0:
                raise ValueError(f"{field} must be non-negative, got {value!r}")
        if not self.brand:
            raise ValueError("brand must be a non-empty string")
