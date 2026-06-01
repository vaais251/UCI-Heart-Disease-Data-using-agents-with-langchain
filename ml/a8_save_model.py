"""
CardioTriage AI — Phase A, Step A8: Persist the best model (closes Phase A).

1. Evaluate the chosen model (regularized Gradient Boosting) on the held-out
   test split at threshold 0.35 -> honest metrics for the record.
2. Refit the risk pipeline on ALL data (deployment artifact).
3. Fit the K-means phenotype pipeline on ALL data and build a
   cluster -> phenotype-name + priority map.
4. Save risk_model.joblib, phenotype_model.joblib, metadata.json to models/.

Run with:  uv run python ml/a8_save_model.py
Outputs:   models/risk_model.joblib, models/phenotype_model.joblib,
           models/metadata.json
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import compute_metrics
from preprocessing import (
    CLUSTER_FEATURES,
    FEATURE_COLUMNS,
    build_preprocessor,
    load_and_split,
    load_raw,
    prepare_frame,
)

MODELS_DIR = Path(__file__).parent.parent / "models"
THRESHOLD = 0.35  # chosen in A6c (recall-favoring)


def build_risk_pipeline() -> Pipeline:
    """Preprocessor + regularized Gradient Boosting (the A6b winner)."""
    return Pipeline([
        ("pre", build_preprocessor()),
        ("clf", GradientBoostingClassifier(learning_rate=0.05, max_depth=2,
                                           n_estimators=200, subsample=0.8,
                                           random_state=42)),
    ])


def build_phenotype_pipeline() -> Pipeline:
    """Impute + scale + K-means (k=4) for phenotype assignment."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("kmeans", KMeans(n_clusters=4, n_init=10, random_state=42)),
    ])


def name_phenotype(row: pd.Series) -> str:
    """Heuristic label from a cluster's average profile (from A7 analysis)."""
    if row["ca"] >= 1.5:
        return "Advanced multi-vessel disease"
    if row["trestbps"] >= 145 and row["chol"] >= 270:
        return "Hypertensive-hypercholesterolemic"
    if row["thalch"] <= 125:
        return "Reduced-exertion / low max-HR"
    return "Younger lower-risk"


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    # --- 1. Honest held-out evaluation at the chosen threshold ------------
    X_train, X_test, y_train, y_test = load_and_split()
    eval_model = build_risk_pipeline().fit(X_train, y_train)
    proba = eval_model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)
    test_metrics = compute_metrics(y_test, y_pred, proba)
    print("Held-out test metrics (GB @ threshold 0.35):")
    for k, v in test_metrics.items():
        print(f"  {k:12s}: {v:.3f}")

    # --- 2. Refit risk model on ALL data ----------------------------------
    full = prepare_frame(load_raw())
    X_all, y_all = full[FEATURE_COLUMNS], full["target"]
    risk_model = build_risk_pipeline().fit(X_all, y_all)

    # --- 3. Fit phenotype model + build phenotype map ---------------------
    phenotype_model = build_phenotype_pipeline().fit(full[CLUSTER_FEATURES])
    full["cluster"] = phenotype_model.predict(full[CLUSTER_FEATURES])

    profile = full.groupby("cluster").agg(
        size=("cluster", "size"),
        disease_rate=("target", "mean"),
        age=("age", "mean"), trestbps=("trestbps", "mean"),
        chol=("chol", "mean"), thalch=("thalch", "mean"),
        oldpeak=("oldpeak", "mean"), ca=("ca", "mean"),
    )
    # Priority 1 = most urgent (highest disease rate).
    rank = profile["disease_rate"].rank(ascending=False).astype(int)
    phenotype_map = {}
    for cluster_id, row in profile.iterrows():
        phenotype_map[int(cluster_id)] = {
            "name": name_phenotype(row),
            "priority": int(rank[cluster_id]),
            "disease_rate": round(float(row["disease_rate"]), 3),
            "size": int(row["size"]),
        }

    print("\nPhenotype map:")
    for cid, info in sorted(phenotype_map.items(), key=lambda kv: kv[1]["priority"]):
        print(f"  cluster {cid} -> priority {info['priority']}: {info['name']} "
              f"(disease_rate={info['disease_rate']}, n={info['size']})")

    # --- 4. Save everything ------------------------------------------------
    joblib.dump(risk_model, MODELS_DIR / "risk_model.joblib")
    joblib.dump(phenotype_model, MODELS_DIR / "phenotype_model.joblib")
    metadata = {
        "model": "GradientBoostingClassifier (regularized)",
        "threshold": THRESHOLD,
        "feature_columns": FEATURE_COLUMNS,
        "cluster_features": CLUSTER_FEATURES,
        "phenotype_map": phenotype_map,
        "test_metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved to {MODELS_DIR}:")
    print("  - risk_model.joblib")
    print("  - phenotype_model.joblib")
    print("  - metadata.json")


if __name__ == "__main__":
    main()
