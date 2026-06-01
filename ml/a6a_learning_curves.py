"""
CardioTriage AI — Phase A, Step A6a: Bias-variance via learning curves.

For each model we plot training-score vs cross-validation-score as the
training set grows. The shape diagnoses bias (underfit) vs variance (overfit).
All work happens on the TRAINING data only (learning_curve does its own
internal CV); the held-out test set is never touched here.

Run with:  uv run python ml/a6a_learning_curves.py
Outputs:   images/a6a_learning_curves.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor, load_and_split

IMAGES_DIR = Path(__file__).parent.parent / "images"

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}


def main() -> None:
    IMAGES_DIR.mkdir(exist_ok=True)
    X_train, _, y_train, _ = load_and_split()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, (name, classifier) in zip(axes, MODELS.items()):
        model = Pipeline([("pre", build_preprocessor()), ("clf", classifier)])

        # learning_curve retrains on growing slices and reports train + CV
        # accuracy for each slice (5-fold CV internally).
        sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=np.linspace(0.1, 1.0, 6),
            cv=5, scoring="accuracy", shuffle=True, random_state=42, n_jobs=-1,
        )

        train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
        val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

        ax.plot(sizes, train_mean, "o-", color="C0", label="training score")
        ax.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.15, color="C0")
        ax.plot(sizes, val_mean, "o-", color="C1", label="cross-val score")
        ax.fill_between(sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.15, color="C1")

        final_gap = train_mean[-1] - val_mean[-1]
        ax.set_title(f"{name}\n(final train-CV gap = {final_gap:.3f})")
        ax.set_xlabel("training examples")
        ax.set_ylim(0.6, 1.02)
        ax.legend(loc="lower right")
        axes[0].set_ylabel("accuracy")

        print(f"{name:22s} final train={train_mean[-1]:.3f}  "
              f"CV={val_mean[-1]:.3f}  gap={final_gap:.3f}")

    fig.suptitle("Learning curves: bias-variance diagnosis", fontsize=14)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "a6a_learning_curves.png", dpi=120)
    plt.close(fig)
    print("\nSaved: images/a6a_learning_curves.png")


if __name__ == "__main__":
    main()
