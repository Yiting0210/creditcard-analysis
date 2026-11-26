"""Train the scorecard, random forest and risk-segment models, evaluate them on a held-out split, and save a versioned bundle.

    python -m src.train                      # uses $DATA_DIR/application_train.csv
    python -m src.train --sample 20000       # quick run on a subset

The saved models are exactly the ones evaluated on the 20% hold-out, so the reported metrics describe the artifacts that score new applicants.
"""
import argparse
import json
import pickle
import subprocess
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import scorecardpy as sc
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score, roc_curve, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import clean as cleaning
from . import features
from .config import DATA_DIR, ID_COL, MODEL_DIR, REPORT_DIR, TARGET_COL

SEED = 42
IV_THRESHOLD = 0.1
N_CLUSTERS = 5
TOP_K_FEATURES = 15


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def psi(expected, actual, buckets=10):
    """Population stability of a score between two samples (quantile buckets of the first)."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def evaluate(name, y_tr, p_tr, y_te, p_te):
    pred = (p_te >= 0.5).astype(int)
    auc = roc_auc_score(y_te, p_te)
    return {
        "model": name,
        "auc": round(float(auc), 4),
        "gini": round(float(2 * auc - 1), 4),
        "ks": round(ks_stat(y_te, p_te), 4),
        "recall": round(float(recall_score(y_te, pred)), 4),
        "f1": round(float(f1_score(y_te, pred)), 4),
        "psi_train_vs_test": round(psi(p_tr, p_te), 4),
    }


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def train(df: pd.DataFrame, n_estimators: int = 100, top_k: int = TOP_K_FEATURES, out_dir=MODEL_DIR,
          report_dir=REPORT_DIR, version: str | None = None) -> dict:
    cleaning.validate(df, require_target=True)
    df = cleaning.clean(df)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df[TARGET_COL])
    y_tr, y_te = train_df[TARGET_COL].values, test_df[TARGET_COL].values

    # ---- Pipeline A: WOE binning + IV selection + logistic-regression scorecard ----
    bin_cols = [c for c in train_df.columns if c not in (ID_COL, TARGET_COL)]
    bins = sc.woebin(train_df[bin_cols + [TARGET_COL]], y=TARGET_COL, print_step=0)
    iv = sc.iv(train_df[bin_cols + [TARGET_COL]], y=TARGET_COL)
    selected = iv.loc[iv["info_value"] > IV_THRESHOLD, "variable"].tolist()
    if not selected:
        raise RuntimeError(f"no variable has IV > {IV_THRESHOLD}")
    woe_tr = sc.woebin_ply(train_df[selected], bins, print_step=0)
    woe_te = sc.woebin_ply(test_df[selected], bins, print_step=0)
    woe_cols = [c + "_woe" for c in selected]
    lr = LogisticRegression(C=1, class_weight="balanced", random_state=SEED, max_iter=1000)
    lr.fit(woe_tr[woe_cols], y_tr)
    card = sc.scorecard(bins, lr, woe_cols, points0=600, odds0=1 / 11, pdo=50)
    p_lr_tr, p_lr_te = lr.predict_proba(woe_tr[woe_cols])[:, 1], lr.predict_proba(woe_te[woe_cols])[:, 1]

    # ---- Pipeline B: median imputation + one-hot, tree models ----
    medians = features.fit_medians(train_df)
    X_tr = features.build_matrix(train_df, medians)
    columns = X_tr.columns.tolist()
    X_te = features.build_matrix(test_df, medians, columns)
    assert ID_COL not in columns and TARGET_COL not in columns

    dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=100, class_weight="balanced", random_state=SEED)
    dt.fit(X_tr, y_tr)
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=10, min_samples_leaf=50,
                                class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    p_dt_tr, p_dt_te = dt.predict_proba(X_tr)[:, 1], dt.predict_proba(X_te)[:, 1]
    p_rf_tr, p_rf_te = rf.predict_proba(X_tr)[:, 1], rf.predict_proba(X_te)[:, 1]

    # ---- Risk segments: K-Means on the random forest's top features ----
    importance = pd.Series(rf.feature_importances_, index=columns).sort_values(ascending=False)
    top = importance.head(top_k).index.tolist()
    scaler = StandardScaler().fit(X_tr[top])
    km = KMeans(n_clusters=N_CLUSTERS, init="k-means++", random_state=SEED, n_init=10)
    c_tr = km.fit_predict(scaler.transform(X_tr[top]))
    c_te = km.predict(scaler.transform(X_te[top]))
    seg = pd.DataFrame({"cluster": c_tr, "y": y_tr}).groupby("cluster")["y"].agg(["mean", "size"])
    # rank 1 = highest default rate on the training set
    rank = seg["mean"].rank(ascending=False, method="first").astype(int)
    seg_te = pd.DataFrame({"cluster": c_te, "y": y_te}).groupby("cluster")["y"].agg(["mean", "size"])
    sample = np.random.default_rng(SEED).choice(len(X_tr), size=min(20000, len(X_tr)), replace=False)
    sil = silhouette_score(scaler.transform(X_tr[top].iloc[sample]), c_tr[sample])
    base_rate = float(y_tr.mean())
    segments = [
        {"cluster": int(c), "risk_rank": int(rank[c]), "train_default_rate": round(float(seg.loc[c, "mean"]), 4),
         "train_size": int(seg.loc[c, "size"]),
         "test_default_rate": round(float(seg_te["mean"].get(c, np.nan)), 4),
         "lift_vs_average": round(float(seg.loc[c, "mean"]) / base_rate, 2)}
        for c in seg.index
    ]
    segments.sort(key=lambda s: s["risk_rank"])

    metrics = {
        "rows": int(len(df)), "train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
        "default_rate": round(float(df[TARGET_COL].mean()), 4),
        "scorecard_variables": len(selected), "tree_features": len(columns),
        "models": [evaluate("logistic_scorecard", y_tr, p_lr_tr, y_te, p_lr_te),
                   evaluate("decision_tree", y_tr, p_dt_tr, y_te, p_dt_te),
                   evaluate("random_forest", y_tr, p_rf_tr, y_te, p_rf_te)],
        "segments": {"features": top, "silhouette_sample_20k": round(float(sil), 4), "clusters": segments},
    }

    version = version or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    bundle = out_dir / version
    bundle.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr, bundle / "scorecard_lr.joblib")
    joblib.dump(rf, bundle / "random_forest.joblib")
    joblib.dump(scaler, bundle / "segment_scaler.joblib")
    joblib.dump(km, bundle / "segment_kmeans.joblib")
    with open(bundle / "woe_bins.pkl", "wb") as f:
        pickle.dump(bins, f)
    with open(bundle / "scorecard.pkl", "wb") as f:
        pickle.dump(card, f)
    meta = {
        "version": version, "trained_at": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(),
        "scorecard_variables": selected, "tree_columns": columns, "medians": medians,
        "segment_features": top, "segment_risk_rank": {str(int(c)): int(rank[c]) for c in seg.index},
    }
    (bundle / "metadata.json").write_text(json.dumps(meta, indent=1))
    (bundle / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out_dir / "LATEST").write_text(version)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps({"model_version": version, **metrics}, indent=1))
    return {"version": version, "metrics": metrics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DATA_DIR / "application_train.csv"))
    ap.add_argument("--sample", type=int, default=0, help="train on a random subset (for quick runs)")
    ap.add_argument("--n-estimators", type=int, default=100)
    args = ap.parse_args()
    df = pd.read_csv(args.input, low_memory=False)
    if args.sample:
        df = df.sample(n=args.sample, random_state=SEED)
    result = train(df, n_estimators=args.n_estimators)
    m = result["metrics"]
    print(f"model version {result['version']}: {m['rows']} rows, {m['scorecard_variables']} scorecard variables, "
          f"{m['tree_features']} tree features")
    for row in m["models"]:
        print(f"  {row['model']:<20} AUC {row['auc']:.4f}  KS {row['ks']:.4f}  F1 {row['f1']:.4f}")
    top_seg = m["segments"]["clusters"][0]
    print(f"  highest-risk segment: {top_seg['train_default_rate']:.2%} default "
          f"({top_seg['lift_vs_average']}x average), silhouette {m['segments']['silhouette_sample_20k']}")


if __name__ == "__main__":
    main()
