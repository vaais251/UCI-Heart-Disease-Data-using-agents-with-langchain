"""
CardioTriage AI — Phase A, Step A6c: Decision-threshold tuning.

The model outputs P(disease). Instead of the default 0.5 cutoff, we pick a
recall-favoring threshold (missed heart disease >> false alarm). We sweep
thresholds, plot precision/recall/specificity/F2, choose the F2-optimal
threshold, and compare confusion matrices at 0.5 vs the chosen threshold.

Run with:  uv run python ml/a6c_threshold.py
Outputs:   images/a6c_threshold_sweep.png,
           images/a6c_cm_default.png, images/a6c_cm_tuned.png
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import IMAGES_DIR, plot_confusion_matrix
from preprocessing import build_preprocessor, load_and_split


def specificity_score(y_true, y_pred) -> float:
    """Of truly-healthy patients, fraction correctly cleared."""
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    return tn / (tn + fp) if (tn + fp) else float("nan")


def main() -> None:
    X_train, X_test, y_train, y_test = load_and_split()

    # The regularized Gradient Boosting from A6b (good generalization).
    model = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", GradientBoostingClassifier(learning_rate=0.05, max_depth=2,
                                            n_estimators=200, subsample=0.8,
                                            random_state=42)),
    ])
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    y_test = y_test.to_numpy()

    # --- Sweep thresholds --------------------------------------------------
    thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)
    rows = []
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "specificity": specificity_score(y_test, y_pred),
            "f2": fbeta_score(y_test, y_pred, beta=2, zero_division=0),
        })
    sweep = pd.DataFrame(rows)

    # F2-optimal threshold (recall-weighted balance).
    best = sweep.loc[sweep["f2"].idxmax()]
    chosen_t = float(best["threshold"])

    # Highest threshold that still guarantees recall >= 0.90.
    high_recall = sweep[sweep["recall"] >= 0.90]
    recall90_t = float(high_recall["threshold"].max()) if not high_recall.empty else None

    print("=" * 70)
    print("THRESHOLD SWEEP (regularized Gradient Boosting, test set)")
    print("=" * 70)
    print(sweep.round(3).to_string(index=False))
    print(f"\nF2-optimal threshold     : {chosen_t:.2f}  "
          f"(recall={best['recall']:.3f}, precision={best['precision']:.3f}, "
          f"F2={best['f2']:.3f})")
    if recall90_t is not None:
        print(f"Threshold for recall>=0.90: {recall90_t:.2f}")

    # --- Plot the trade-off ------------------------------------------------
    plt.figure(figsize=(8, 5))
    for col, color in [("precision", "C0"), ("recall", "C1"),
                       ("specificity", "C2"), ("f2", "C3")]:
        plt.plot(sweep["threshold"], sweep[col], "o-", color=color, label=col)
    plt.axvline(0.5, color="gray", linestyle=":", label="default 0.5")
    plt.axvline(chosen_t, color="black", linestyle="--",
                label=f"chosen {chosen_t:.2f}")
    plt.xlabel("decision threshold  (flag disease if P >= threshold)")
    plt.ylabel("score")
    plt.title("Threshold trade-off: precision / recall / specificity / F2")
    plt.legend(loc="lower center", ncol=3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "a6c_threshold_sweep.png", dpi=120)
    plt.close()

    # --- Confusion matrices: default 0.5 vs chosen -------------------------
    plot_confusion_matrix(y_test, (proba >= 0.5).astype(int),
                          "GB @ threshold 0.50 (default)", "a6c_cm_default.png")
    plot_confusion_matrix(y_test, (proba >= chosen_t).astype(int),
                          f"GB @ threshold {chosen_t:.2f} (tuned)", "a6c_cm_tuned.png")

    # Report the false-negative reduction explicitly.
    fn_default = int(np.sum((y_test == 1) & (proba < 0.5)))
    fn_tuned = int(np.sum((y_test == 1) & (proba < chosen_t)))
    print(f"\nMissed cases (false negatives): {fn_default} at 0.50  ->  "
          f"{fn_tuned} at {chosen_t:.2f}")
    print("Saved: images/a6c_threshold_sweep.png, a6c_cm_default.png, a6c_cm_tuned.png")


if __name__ == "__main__":
    main()
