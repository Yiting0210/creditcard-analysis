"""Daily batch job: load new applications, then score them with the current model bundle.

    python -m src.score load  --run-date 2026-11-25    # validate + clean $DATA_DIR/$INPUT_FILE -> applications_clean
    python -m src.score score --run-date 2026-11-25    # score that day's applications      -> scores

Both steps are idempotent: re-running a date first deletes that date's rows, so a retry never duplicates data.
Without DATABASE_URL, `score` reads the CSV directly and writes reports/scores_<date>.csv instead.
"""
import argparse
import json
import pickle
from datetime import date, datetime, timezone

import joblib
import pandas as pd
import scorecardpy as sc
from sqlalchemy import create_engine, inspect, text

from . import clean as cleaning
from . import features
from .config import DATA_DIR, DATABASE_URL, ID_COL, INPUT_FILE, MODEL_DIR, REPORT_DIR, TARGET_COL

STAGING_TABLE = "applications_clean"
SCORES_TABLE = "scores"


class Bundle:
    """A saved model version (see train.py)."""

    def __init__(self, model_dir=MODEL_DIR, version: str | None = None):
        self.version = version or (model_dir / "LATEST").read_text().strip()
        d = model_dir / self.version
        self.meta = json.loads((d / "metadata.json").read_text())
        self.lr = joblib.load(d / "scorecard_lr.joblib")
        self.rf = joblib.load(d / "random_forest.joblib")
        self.scaler = joblib.load(d / "segment_scaler.joblib")
        self.km = joblib.load(d / "segment_kmeans.joblib")
        with open(d / "woe_bins.pkl", "rb") as f:
            self.bins = pickle.load(f)
        with open(d / "scorecard.pkl", "rb") as f:
            self.card = pickle.load(f)


def score_frame(df: pd.DataFrame, bundle: Bundle, run_date: str) -> pd.DataFrame:
    """Score cleaned applications; one output row per applicant."""
    selected = bundle.meta["scorecard_variables"]
    woe = sc.woebin_ply(df[selected], bundle.bins, print_step=0)
    lr_pd = bundle.lr.predict_proba(woe[[c + "_woe" for c in selected]])[:, 1]
    points = sc.scorecard_ply(df[selected], bundle.card, print_step=0)["score"].values

    X = features.build_matrix(df, bundle.meta["medians"], bundle.meta["tree_columns"])
    rf_pd = bundle.rf.predict_proba(X)[:, 1]
    top = bundle.meta["segment_features"]
    cluster = bundle.km.predict(bundle.scaler.transform(X[top]))
    rank = bundle.meta["segment_risk_rank"]

    return pd.DataFrame({
        "sk_id_curr": df[ID_COL].astype(int).values,
        "run_date": pd.to_datetime(run_date).date(),
        "model_version": bundle.version,
        "scorecard_points": points.round(1),
        "scorecard_pd": lr_pd.round(6),
        "rf_pd": rf_pd.round(6),
        "segment": cluster.astype(int),
        "segment_risk_rank": [rank[str(int(c))] for c in cluster],
        "scored_at": datetime.now(timezone.utc).replace(tzinfo=None),
    })


def replace_partition(engine, table: str, frame: pd.DataFrame, run_date: str) -> int:
    """Delete the run date's rows, then append the new ones, in one transaction."""
    with engine.begin() as conn:
        if inspect(conn).has_table(table):
            conn.execute(text(f"DELETE FROM {table} WHERE run_date = :d"), {"d": pd.to_datetime(run_date).date()})
        frame.to_sql(table, conn, if_exists="append", index=False, chunksize=1000)
    return len(frame)


def load_applications(run_date: str, input_path=None, engine=None) -> int:
    path = input_path or DATA_DIR / INPUT_FILE
    df = pd.read_csv(path, low_memory=False)
    cleaning.validate(df)
    df = cleaning.clean(df).drop(columns=[TARGET_COL], errors="ignore")
    df.insert(0, "run_date", pd.to_datetime(run_date).date())
    engine = engine or create_engine(DATABASE_URL)
    return replace_partition(engine, STAGING_TABLE, df, run_date)


def score_applications(run_date: str, engine=None, bundle: Bundle | None = None, input_path=None) -> int:
    bundle = bundle or Bundle()
    if engine is None and not DATABASE_URL:
        df = cleaning.clean(pd.read_csv(input_path or DATA_DIR / INPUT_FILE, low_memory=False))
        out = score_frame(df, bundle, run_date)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out.to_csv(REPORT_DIR / f"scores_{run_date}.csv", index=False)
        return len(out)
    engine = engine or create_engine(DATABASE_URL)
    df = pd.read_sql(text(f"SELECT * FROM {STAGING_TABLE} WHERE run_date = :d"), engine,
                     params={"d": pd.to_datetime(run_date).date()})
    if df.empty:
        raise RuntimeError(f"no applications loaded for {run_date}; run the load step first")
    out = score_frame(df.drop(columns=["run_date"]), bundle, run_date)
    return replace_partition(engine, SCORES_TABLE, out, run_date)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["load", "score"])
    ap.add_argument("--run-date", default=date.today().isoformat())
    args = ap.parse_args()
    n = load_applications(args.run_date) if args.step == "load" else score_applications(args.run_date)
    print(f"{args.step}: {n} rows for {args.run_date}")


if __name__ == "__main__":
    main()
