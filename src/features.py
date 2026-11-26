"""Feature matrix for the tree model and clustering (the "Pipeline B" of the analysis notebook)."""
import pandas as pd

from .config import ID_COL, TARGET_COL

EXCLUDE = [ID_COL, TARGET_COL]  # the ID must never be a feature


def fit_medians(df: pd.DataFrame) -> dict:
    """Medians of every numeric feature, learned once on the training set."""
    num = df.drop(columns=[c for c in EXCLUDE if c in df.columns]).select_dtypes(include=["number"])
    return {c: float(v) for c, v in num.median().items()}


def build_matrix(df: pd.DataFrame, medians: dict, columns: list | None = None) -> pd.DataFrame:
    """Impute with training medians, one-hot encode, and (at scoring time) align to the training columns.

    Training: columns=None, uses drop_first=True and returns the column list to save.
    Scoring:  pass the saved columns; unseen categories are dropped and missing ones become 0,
              so the matrix always has exactly the training layout.
    """
    X = df.drop(columns=[c for c in EXCLUDE if c in df.columns]).copy()
    num_cols = [c for c in medians if c in X.columns]
    X[num_cols] = X[num_cols].fillna(pd.Series(medians)[num_cols])
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if columns is None:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)
        return X
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=float)
    X = X.reindex(columns=columns)
    # a numeric column absent from this batch gets the training median; a missing dummy column means "not this category"
    for c in columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(medians.get(c, 0.0))
    return X.astype(float)
