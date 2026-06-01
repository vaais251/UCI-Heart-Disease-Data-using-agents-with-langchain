"""
CardioTriage AI — Phase A, Step A4: Baseline model (Logistic Regression).

We wrap our A2 preprocessor and a LogisticRegression in a single Pipeline.
The Pipeline guarantees the preprocessor is fit on TRAIN only, then reused on
TEST automatically -> no leakage. We then evaluate on the held-out test set.

Run with:  uv run python ml/a4_logreg.py
Outputs:   images/a4_confusion_matrix.png, images/a4_roc_curve.png
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from evaluate import compute_metrics, plot_confusion_matrix, plot_roc, print_metrics
from preprocessing import build_preprocessor, load_and_split


def main() -> None:
    X_train, X_test, y_train, y_test = load_and_split()

    # Pipeline = preprocessing + classifier as one object.
    # max_iter is raised so the optimizer fully converges on our scaled data.
    model = Pipeline(
        steps=[
            ("pre", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)

    # Predictions and probabilities on the held-out test set.
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # P(disease) for each patient

    # Train vs test accuracy: an early peek at over/under-fitting (full
    # bias-variance analysis comes in A6).
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print("=" * 70)
    print("LOGISTIC REGRESSION — baseline")
    print("=" * 70)
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test  accuracy: {test_acc:.3f}")

    metrics = compute_metrics(y_test, y_pred, y_proba)
    print_metrics("Logistic Regression (test set)", metrics)

    plot_confusion_matrix(
        y_test, y_pred,
        title="Logistic Regression — confusion matrix",
        filename="a4_confusion_matrix.png",
    )
    plot_roc(
        [("Logistic Regression", y_test, y_proba)],
        title="Logistic Regression — ROC curve",
        filename="a4_roc_curve.png",
    )
    print("\nSaved: images/a4_confusion_matrix.png, images/a4_roc_curve.png")


if __name__ == "__main__":
    main()
