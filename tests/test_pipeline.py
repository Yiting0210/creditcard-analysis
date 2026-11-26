import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from src import clean as cleaning
from src import features
from src.score import Bundle, load_applications, score_applications, score_frame
from tests.conftest import make_applications


def test_clean_fixes_sentinel_and_maps_industries(applications):
    out = cleaning.clean(applications)
    sentinel = applications["DAYS_EMPLOYED"] == 365243
    assert out.loc[sentinel, "DAYS_EMPLOYED"].isna().all()
    assert out.loc[sentinel, "YEARS_EMPLOYED"].isna().all()
    assert set(out["ORGANIZATION_TYPE"]) <= {"Self-employed", "Public", "Trade_Services", "Other"}
    assert (out["YEARS_BIRTH"] > 0).all()


def test_clean_is_batch_independent(applications):
    """Cleaning row by row in chunks must equal cleaning everything at once (the old script did not)."""
    whole = cleaning.clean(applications)
    chunks = pd.concat([cleaning.clean(c) for c in np.array_split(applications, 7)])
    pd.testing.assert_frame_equal(whole, chunks)


@pytest.mark.parametrize("breakage, message", [
    (lambda d: d.drop(columns=["DAYS_BIRTH"]), "missing required"),
    (lambda d: pd.concat([d, d.head(1)]), "duplicated"),
    (lambda d: d.assign(SK_ID_CURR=np.nan), "null"),
])
def test_validate_rejects_bad_input(applications, breakage, message):
    with pytest.raises(ValueError, match=message):
        cleaning.validate(breakage(applications))


def test_id_and_target_are_never_features(applications):
    df = cleaning.clean(applications)
    X = features.build_matrix(df, features.fit_medians(df))
    assert "SK_ID_CURR" not in X.columns and "TARGET" not in X.columns


def test_scoring_matrix_matches_training_layout(applications):
    df = cleaning.clean(applications)
    medians = features.fit_medians(df)
    columns = features.build_matrix(df, medians).columns.tolist()
    new = df.head(50).copy()
    new["NAME_CONTRACT_TYPE"] = "Brand-new product"          # unseen category
    new = new.drop(columns=["AMT_ANNUITY"])                     # column missing from the batch
    X = features.build_matrix(new, medians, columns)
    assert X.columns.tolist() == columns
    assert not X.isna().any().any()
    assert (X["AMT_ANNUITY"] == medians["AMT_ANNUITY"]).all()


def test_training_reports_metrics_and_versions_bundle(trained):
    model_dir, result = trained
    assert (model_dir / "LATEST").read_text() == "test"
    names = [m["model"] for m in result["metrics"]["models"]]
    assert names == ["logistic_scorecard", "decision_tree", "random_forest"]
    for m in result["metrics"]["models"]:
        assert 0.5 < m["auc"] <= 1.0
    ranks = sorted(c["risk_rank"] for c in result["metrics"]["segments"]["clusters"])
    assert ranks == [1, 2, 3, 4, 5]


def test_score_frame_one_row_per_applicant(trained):
    model_dir, _ = trained
    bundle = Bundle(model_dir)
    new = cleaning.clean(make_applications(n=400, seed=7, with_target=False))
    out = score_frame(new, bundle, "2026-11-25")
    assert len(out) == 400 and out["sk_id_curr"].is_unique
    assert out[["scorecard_pd", "rf_pd"]].apply(lambda s: s.between(0, 1)).all().all()
    assert not out.isna().any().any()
    assert out["model_version"].eq("test").all()


def test_load_and_score_are_idempotent(trained, tmp_path):
    model_dir, _ = trained
    csv = tmp_path / "new.csv"
    make_applications(n=300, seed=3, with_target=False).to_csv(csv, index=False)
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    bundle = Bundle(model_dir)
    for _ in range(2):  # a retry of the same day
        load_applications("2026-11-25", input_path=csv, engine=engine)
        score_applications("2026-11-25", engine=engine, bundle=bundle)
    load_applications("2026-11-26", input_path=csv, engine=engine)
    score_applications("2026-11-26", engine=engine, bundle=bundle)
    with engine.connect() as conn:
        per_day = dict(conn.execute(text("SELECT run_date, COUNT(*) FROM scores GROUP BY run_date")).all())
        staged = conn.execute(text("SELECT COUNT(*) FROM applications_clean")).scalar()
    assert list(per_day.values()) == [300, 300]
    assert staged == 600
