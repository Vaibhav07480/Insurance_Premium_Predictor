import pickle
from pathlib import Path

import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
_model = None

TIER_1_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"
]
TIER_2_CITIES = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam",
    "Coimbatore", "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur",
    "Raipur", "Amritsar", "Varanasi", "Agra", "Dehradun", "Mysore", "Jabalpur",
    "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik", "Allahabad", "Udaipur",
    "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode",
    "Warangal", "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol",
    "Siliguri",
]


def load_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def bmi(weight: float, height: float) -> float:
    return weight / (height ** 2)


def lifestyle_risk(smoker: bool, bmi_value: float) -> str:
    if smoker and bmi_value > 30:
        return "high"
    if smoker or bmi_value > 27:
        return "medium"
    return "low"


def age_group(age: int) -> str:
    if age < 25:
        return "young"
    if age < 45:
        return "adult"
    if age < 60:
        return "middle_aged"
    return "senior"


def city_tier(city: str) -> int:
    if city in TIER_1_CITIES:
        return 1
    if city in TIER_2_CITIES:
        return 2
    return 3


def predict_premium_category(
    age: int,
    weight: float,
    height: float,
    income_lpa: float,
    smoker: bool,
    city: str,
    occupation: str,
) -> str:
    bmi_value = bmi(weight, height)
    input_df = pd.DataFrame(
        [
            {
                "bmi": bmi_value,
                "age_group": age_group(age),
                "lifestyle_risk": lifestyle_risk(smoker, bmi_value),
                "city_tier": city_tier(city),
                "income_lpa": income_lpa,
                "occupation": occupation,
            }
        ]
    )
    return load_model().predict(input_df)[0]
