import numpy as np
import pandas as pd
import pytest


def make_applications(n=3000, seed=0, with_target=True):
    """Small synthetic table with the columns the pipeline relies on; default risk falls with EXT_SOURCE."""
    rng = np.random.default_rng(seed)
    ext = rng.uniform(0, 1, n)
    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(100000, 100000 + n),
        "EXT_SOURCE_1": np.where(rng.uniform(size=n) < 0.3, np.nan, ext + rng.normal(0, 0.1, n)),
        "EXT_SOURCE_2": ext + rng.normal(0, 0.15, n),
        "AMT_CREDIT": rng.lognormal(12, 0.5, n),
        "AMT_ANNUITY": rng.lognormal(10, 0.4, n),
        "DAYS_BIRTH": -rng.integers(20 * 365, 65 * 365, n),
        "DAYS_EMPLOYED": np.where(rng.uniform(size=n) < 0.1, 365243, -rng.integers(0, 30 * 365, n)),
        "ORGANIZATION_TYPE": rng.choice(["Self-employed", "School", "Trade: type 7", "XNA", "Unlisted Org"], n),
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
        "CODE_GENDER": rng.choice(["M", "F"], n),
    })
    if with_target:
        p = 1 / (1 + np.exp(-(-0.5 - 4 * (ext - 0.5))))
        df["TARGET"] = (rng.uniform(size=n) < p * 0.3).astype(int)
    return df


@pytest.fixture
def applications():
    return make_applications()


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Train once on synthetic data for the whole test session."""
    from src.train import train
    root = tmp_path_factory.mktemp("bundle")
    result = train(make_applications(), n_estimators=20, top_k=4,
                   out_dir=root / "models", report_dir=root / "reports", version="test")
    return root / "models", result
