"""
CardioTriage AI — Phase A, Step A5: Compare three models.

Train Logistic Regression (baseline), Random Forest (intermediate), and
Gradient Boosting (advanced) through the SAME leakage-safe Pipeline, then
compare them in one table + a combined ROC plot.

Run with:  uv run python ml/a5_compare.py
Outputs:   images/a5_roc_comparison.png, images/a5_cm_*.png
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from evaluate import compute_metrics, plot_confusion_matrix, plot_roc
from preprocessing import build_preprocessor, load_and_split

# The three competitors. Each is paired with our shared preprocessor.
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}


def main() -> None:
    X_train, X_test, y_train, y_test = load_and_split()

    rows = []
    roc_curves = []
    for name, classifier in MODELS.items():
        model = Pipeline(
            steps=[("pre", build_preprocessor()), ("clf", classifier)]
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        # train_acc lets us see the overfitting gap (train minus test).
        metrics["train_acc"] = model.score(X_train, y_train)
        rows.append({"model": name, **metrics})
        roc_curves.append((name, y_test, y_proba))

        # Save each model's confusion matrix for the report.
        slug = name.lower().replace(" ", "_")
        plot_confusion_matrix(
            y_test, y_pred,
            title=f"{name} — confusion matrix",
            filename=f"a5_cm_{slug}.png",
        )

    # --- Comparison table --------------------------------------------------
    table = pd.DataFrame(rows).set_index("model")
    column_order = ["accuracy", "precision", "recall", "specificity",
                    "f1", "roc_auc", "train_acc"]
    table = table[column_order]

    pd.set_option("display.width", 120)
    print("=" * 78)
    print("MODEL COMPARISON  (metrics on held-out test set)")
    print("=" * 78)
    print(table.round(3).to_string())
    print("\nNote: 'train_acc' vs 'accuracy' (test) reveals overfitting "
          "(big gap = high variance).")

    # --- Combined ROC ------------------------------------------------------
    plot_roc(roc_curves, title="Model comparison — ROC curves",
             filename="a5_roc_comparison.png")
    print("\nSaved: images/a5_roc_comparison.png and three a5_cm_*.png files")


if __name__ == "__main__":
    main()
