import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# ===================== BASIC VISUALS =====================

def plot_correlation_heatmap(df, save_path):
    """Plots correlation matrix for numeric columns only."""
    numeric_df = df.select_dtypes(include=['number'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix (Numeric Columns Only)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pca_projection(X_pca, df, color_col, save_path, title=None):
    """Scatterplot of PCA projection colored by cluster or target."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1],
        hue=df[color_col], palette="husl",
        s=60, edgecolor='k', alpha=0.8
    )
    plt.title(title or f"PCA Projection colored by {color_col}", fontsize=13, fontweight='bold')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title=color_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ===================== GMM-SPECIFIC VISUALS =====================

def plot_aic_bic_vs_k(k_values, aic_scores, bic_scores, save_path, method_name="gmm"):
    """Plots AIC and BIC across different cluster counts."""
    plt.figure(figsize=(7, 5))
    plt.plot(k_values, aic_scores, marker='o', label="AIC", color="steelblue")
    plt.plot(k_values, bic_scores, marker='s', label="BIC", color="indianred")
    plt.title(f"AIC / BIC vs K ({method_name.upper()})", fontsize=13, fontweight='bold')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Score (Lower is Better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_gmm_uncertainty(X_pca, responsibilities, save_path):
    """
    Visualizes GMM cluster uncertainty.
    High uncertainty = less confident cluster membership.
    """
    uncertainty = 1 - np.max(responsibilities, axis=1)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=uncertainty, cmap="magma", s=50, edgecolor='k', alpha=0.8
    )
    plt.colorbar(scatter, label="Cluster Uncertainty (1 - max responsibility)")
    plt.title("GMM Cluster Uncertainty", fontsize=13, fontweight='bold')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ===================== FEATURE IMPORTANCE VISUALS =====================

def plot_pca_feature_loadings(pca, features, save_path, method_name="gmm"):
    """Visualizes PCA component loadings (feature influence)."""
    plt.figure(figsize=(8, 5))
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=features
    )
    sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0)
    plt.title(f"PCA Feature Contributions ({method_name.upper()})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_cluster_feature_means(df, features, save_path, method_name="gmm"):
    """Visualizes mean feature values per cluster (centroids)."""
    cluster_means = df.groupby("cluster")[features].mean()
    plt.figure(figsize=(8, 5))
    sns.heatmap(cluster_means.T, annot=True, cmap="coolwarm")
    plt.title(f"Cluster Centroids by Feature ({method_name.upper()})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_variance(df, features, save_path, method_name="gmm"):
    """Plots variance of feature means across clusters."""
    feature_variance = df.groupby("cluster")[features].mean().var()
    plt.figure(figsize=(8, 4))
    feature_variance.sort_values(ascending=False).plot(kind='bar', color='teal')
    plt.title(f"Feature Variance Across Clusters ({method_name.upper()})", fontsize=13, fontweight='bold')
    plt.ylabel("Variance of Cluster Means")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
