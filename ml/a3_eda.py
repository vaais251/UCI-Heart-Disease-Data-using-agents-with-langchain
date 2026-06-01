"""
CardioTriage AI — Phase A, Step A3: Exploratory Data Analysis.

We explore the TRAINING SET ONLY (test set stays untouched), and save four
figures to images/ plus print a short numeric summary. The question driving
every plot: which features separate disease from no-disease?

Run with:  uv run python ml/a3_eda.py
Outputs:   images/a3_*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, do not open GUI windows
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import NUMERIC_FEATURES, load_and_split

IMAGES_DIR = Path(__file__).parent.parent / "images"
CATEGORICAL_TO_PLOT = ["cp", "sex", "exang", "slope"]


def main() -> None:
    IMAGES_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Recombine training features + label into one frame for plotting.
    X_train, _, y_train, _ = load_and_split()
    df = X_train.copy()
    df["target"] = y_train.values
    # A readable label for legends/axes.
    df["outcome"] = df["target"].map({0: "No disease", 1: "Disease"})
    overall_rate = df["target"].mean()

    # --- Figure 1: target balance -----------------------------------------
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(data=df, x="outcome", hue="outcome", palette="Set2", legend=False)
    ax.set_title("Outcome balance (training set)")
    ax.set_xlabel("")
    ax.set_ylabel("patients")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "a3_target_balance.png", dpi=120)
    plt.close()

    # --- Figure 2: numeric feature distributions by outcome ---------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, col in zip(axes.flat, NUMERIC_FEATURES):
        sns.histplot(
            data=df, x=col, hue="outcome", kde=True,
            stat="density", common_norm=False, palette="Set2", ax=ax,
        )
        ax.set_title(f"{col} by outcome")
    fig.suptitle("Numeric features: disease vs no-disease", fontsize=14)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "a3_numeric_distributions.png", dpi=120)
    plt.close(fig)

    # --- Figure 3: disease RATE by categorical feature --------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, col in zip(axes.flat, CATEGORICAL_TO_PLOT):
        rate = df.groupby(col)["target"].mean().sort_values(ascending=False)
        rate.index = rate.index.astype(str)  # bool/categorical -> text labels
        sns.barplot(x=rate.values, y=rate.index, hue=rate.index,
                    palette="flare", legend=False, ax=ax)
        ax.axvline(overall_rate, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Disease rate by {col}")
        ax.set_xlabel("fraction with disease")
        ax.set_ylabel("")
        ax.set_xlim(0, 1)
    fig.suptitle("Categorical features: disease rate (dashed = overall avg)", fontsize=14)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "a3_categorical_rates.png", dpi=120)
    plt.close(fig)

    # --- Figure 4: correlation heatmap (numeric + target) -----------------
    plt.figure(figsize=(8, 6))
    corr = df[NUMERIC_FEATURES + ["target"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation: numeric features + target")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "a3_correlation_heatmap.png", dpi=120)
    plt.close()

    # --- Console summary ---------------------------------------------------
    print("=" * 70)
    print("MEAN of each numeric feature, split by outcome")
    print("=" * 70)
    print(df.groupby("outcome")[NUMERIC_FEATURES].mean().round(1).T)

    print("\n" + "=" * 70)
    print(f"OVERALL disease rate in training set: {overall_rate:.3f}")
    print("=" * 70)
    print("\nDisease rate by chest-pain type (cp):")
    print(df.groupby("cp")["target"].mean().round(3).sort_values(ascending=False))
    print("\nDisease rate by sex:")
    print(df.groupby("sex")["target"].mean().round(3))

    print(f"\nSaved 4 figures to: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
