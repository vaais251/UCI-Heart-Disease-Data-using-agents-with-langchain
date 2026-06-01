"""
CardioTriage AI — Phase A, Step A7: K-means risk phenotypes.

Unsupervised clustering of patients on core numeric clinical features.
We pick k via elbow + silhouette, fit K-means, then PROFILE each cluster
(size, disease rate, average measurements) to name a risk phenotype.
The target label is used ONLY for post-hoc interpretation, never for fitting.

Run with:  uv run python ml/a7_clustering.py
Outputs:   images/a7_k_selection.png, images/a7_clusters_pca.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import load_raw, prepare_frame

IMAGES_DIR = Path(__file__).parent.parent / "images"
CLUSTER_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
K_FINAL = 4  # chosen after inspecting the k-selection plot (see checkpoint)


def main() -> None:
    IMAGES_DIR.mkdir(exist_ok=True)

    # Full dataset (clustering is descriptive of the whole population).
    df = prepare_frame(load_raw())
    X = df[CLUSTER_FEATURES]

    # Scale features (K-means is distance-based) after median imputation.
    scaler = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    X_scaled = scaler.fit_transform(X)

    # --- Choose k: elbow (inertia) + silhouette ---------------------------
    ks = range(2, 8)
    inertias, silhouettes = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    print("=" * 60)
    print("K-SELECTION")
    print("=" * 60)
    for k, inertia, sil in zip(ks, inertias, silhouettes):
        print(f"  k={k}:  inertia={inertia:8.1f}   silhouette={sil:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(list(ks), inertias, "o-")
    axes[0].set_title("Elbow: inertia vs k")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("inertia (within-cluster spread)")
    axes[1].plot(list(ks), silhouettes, "o-", color="C1")
    axes[1].set_title("Silhouette vs k")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("silhouette score")
    axes[1].axvline(K_FINAL, color="black", linestyle="--", label=f"chosen k={K_FINAL}")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "a7_k_selection.png", dpi=120)
    plt.close(fig)

    # --- Fit final model & profile clusters -------------------------------
    km = KMeans(n_clusters=K_FINAL, n_init=10, random_state=42)
    df["cluster"] = km.fit_predict(X_scaled)

    profile = df.groupby("cluster").agg(
        size=("cluster", "size"),
        disease_rate=("target", "mean"),
        age=("age", "mean"),
        trestbps=("trestbps", "mean"),
        chol=("chol", "mean"),
        thalch=("thalch", "mean"),
        oldpeak=("oldpeak", "mean"),
        ca=("ca", "mean"),
    ).round(1)
    profile["disease_rate"] = (df.groupby("cluster")["target"].mean()).round(3)

    print("\n" + "=" * 60)
    print(f"CLUSTER PROFILES (k={K_FINAL})  — sorted by disease rate")
    print("=" * 60)
    print(profile.sort_values("disease_rate", ascending=False).to_string())

    # --- 2D PCA visualization of the clusters -----------------------------
    coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    plt.figure(figsize=(7, 5.5))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=df["cluster"],
                          cmap="tab10", alpha=0.6, s=18)
    plt.legend(*scatter.legend_elements(), title="cluster")
    plt.xlabel("PCA component 1"); plt.ylabel("PCA component 2")
    plt.title(f"Patient clusters (K-means, k={K_FINAL}) in 2D PCA space")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "a7_clusters_pca.png", dpi=120)
    plt.close()

    print("\nSaved: images/a7_k_selection.png, images/a7_clusters_pca.png")


if __name__ == "__main__":
    main()
