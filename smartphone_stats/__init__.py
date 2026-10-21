"""Statistical analysis of smartphone battery life, performance and preferences."""

from .claims import DEFAULT_CLAIMS, ClaimTest, test_claims_by_brand, test_manufacturer_claim
from .data import generate_sample_dataset, load_dataset, validate_dataset
from .descriptives import battery_life_anova, brand_summary
from .estimation import BatteryEstimate, estimate_by_brand, mean_battery_life
from .models import Measurement
from .preferences import PreferenceTest, test_preference_association
from .prediction import BatteryModel, fit_battery_model

__all__ = [
    "DEFAULT_CLAIMS",
    "BatteryEstimate",
    "BatteryModel",
    "ClaimTest",
    "Measurement",
    "PreferenceTest",
    "battery_life_anova",
    "estimate_by_brand",
    "fit_battery_model",
    "generate_sample_dataset",
    "load_dataset",
    "mean_battery_life",
    "brand_summary",
    "test_claims_by_brand",
    "test_manufacturer_claim",
    "test_preference_association",
    "validate_dataset",
]

__version__ = "0.1.0"
