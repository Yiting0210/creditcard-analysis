"""Validation and cleaning shared by training and scoring.

Missing numeric values are left as NaN here on purpose:
- the scorecard (WOE binning) treats missing values as their own bin;
- the tree / clustering features fill them with medians learned on the training set (see features.py).
"""
import numpy as np
import pandas as pd

from .config import ID_COL, TARGET_COL

DAYS_EMPLOYED_SENTINEL = 365243  # Home Credit's code for "not employed / unknown"

REQUIRED_COLUMNS = [ID_COL, "DAYS_BIRTH", "DAYS_EMPLOYED", "ORGANIZATION_TYPE"]

ORG_MAPPING = {
    "Business Entity Type 3": "Business", "Business Entity Type 2": "Business", "Business Entity Type 1": "Business",
    "Government": "Public", "School": "Public", "Medicine": "Public", "Kindergarten": "Public",
    "University": "Public", "Police": "Public", "Military": "Public", "Security Ministries": "Public", "Emergency": "Public",
    "Self-employed": "Self-employed",
    "Trade: type 7": "Trade_Services", "Trade: type 2": "Trade_Services", "Trade: type 3": "Trade_Services",
    "Trade: type 6": "Trade_Services", "Trade: type 1": "Trade_Services", "Trade: type 5": "Trade_Services",
    "Trade: type 4": "Trade_Services", "Services": "Trade_Services", "Hotel": "Trade_Services",
    "Restaurant": "Trade_Services", "Bank": "Trade_Services", "Insurance": "Trade_Services",
    "Culture": "Trade_Services", "Legal Services": "Trade_Services", "Advertising": "Trade_Services",
    "Postal": "Trade_Services", "Mobile": "Trade_Services", "Telecom": "Trade_Services",
    "Cleaning": "Trade_Services", "Realtor": "Trade_Services",
    "Industry: type 11": "Industry_Construction", "Industry: type 1": "Industry_Construction",
    "Industry: type 4": "Industry_Construction", "Industry: type 7": "Industry_Construction",
    "Industry: type 3": "Industry_Construction", "Industry: type 9": "Industry_Construction",
    "Industry: type 2": "Industry_Construction", "Industry: type 12": "Industry_Construction",
    "Industry: type 5": "Industry_Construction", "Industry: type 10": "Industry_Construction",
    "Industry: type 13": "Industry_Construction", "Industry: type 8": "Industry_Construction",
    "Industry: type 6": "Industry_Construction", "Construction": "Industry_Construction",
    "Housing": "Industry_Construction", "Agriculture": "Industry_Construction",
    "Electricity": "Industry_Construction",
    "Transport: type 2": "Transport", "Transport: type 4": "Transport",
    "Transport: type 3": "Transport", "Transport: type 1": "Transport",
    "Security": "Other", "Other": "Other", "XNA": "Other", "Religion": "Other",
}


def validate(df: pd.DataFrame, require_target: bool = False) -> None:
    """Fail fast on input that would silently produce wrong scores."""
    required = REQUIRED_COLUMNS + ([TARGET_COL] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if df[ID_COL].isna().any():
        raise ValueError(f"{ID_COL} has null values")
    dupes = df[ID_COL].duplicated().sum()
    if dupes:
        raise ValueError(f"{ID_COL} has {dupes} duplicated values")
    if require_target and not df[TARGET_COL].isin([0, 1]).all():
        raise ValueError(f"{TARGET_COL} must be 0/1")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise cleaning only, so the result never depends on how the data is batched."""
    out = df.copy()
    out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_SENTINEL, np.nan)
    out["YEARS_BIRTH"] = out["DAYS_BIRTH"].abs() / 365
    out["YEARS_EMPLOYED"] = out["DAYS_EMPLOYED"].abs() / 365
    out["ORGANIZATION_TYPE"] = out["ORGANIZATION_TYPE"].map(ORG_MAPPING).fillna("Other")
    cat_cols = out.select_dtypes(include=["object"]).columns
    out[cat_cols] = out[cat_cols].fillna("Unknown")
    return out
