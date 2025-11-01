import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_correlation_heatmap(df, save_path):
    """
    Plots a correlation matrix only for numeric columns.
    Skips text or categorical features.
    """
    numeric_df = df.select_dtypes(include=['number'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix (Numeric Columns Only)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



def plot_pca_projection(X_pca, df, color_col, save_path, title=None):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1],
        hue=df[color_col],
        palette="husl", s=60, edgecolor='k', alpha=0.8
    )
    # Use the custom title if provided
    plt.title(title or f"PCA Projection colored by {color_col}", fontsize=13, fontweight='bold')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title=color_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_silhouette_plot(k_values, scores, save_path, title="Silhouette Score vs K"):
    plt.figure(figsize=(6, 4))
    plt.plot(k_values, scores, marker='o', color='teal')
    plt.title(title)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_db_index_vs_k(k_values, scores, save_path, title="Davies–Bouldin Index vs K"):
    plt.figure(figsize=(6, 4))
    plt.plot(k_values, scores, marker='o', color='indianred')
    plt.title(title)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Davies–Bouldin Index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



def plot_ari_nmi_bar(k_values, ari_scores, nmi_scores, save_path):
    x = np.arange(len(k_values))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, ari_scores, width, label='ARI', color='steelblue')
    plt.bar(x + width/2, nmi_scores, width, label='NMI', color='salmon')
    plt.xticks(x, k_values)
    plt.title("ARI vs NMI Across Cluster Sizes")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_pca_feature_loadings(pca, features, save_path, method_name="kmeans"):
    """
    Visualizes feature contributions to PCA components.
    High absolute values indicate strong influence on separation.
    """
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

def plot_cluster_feature_means(df, features, save_path, method_name="kmeans"):
    """
    Visualizes mean feature values per cluster.
    Columns = clusters, Rows = features.
    Features with strong color variation discriminate clusters most.
    """
    cluster_means = df.groupby("cluster")[features].mean()
    plt.figure(figsize=(8, 5))
    sns.heatmap(cluster_means.T, annot=True, cmap="coolwarm")
    plt.title(f"Cluster Centroids by Feature ({method_name.upper()})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_feature_variance(df, features, save_path, method_name="kmeans"):
    """
    Plots the variance of feature means across clusters.
    High variance means the feature strongly separates clusters.
    """
    feature_variance = df.groupby("cluster")[features].mean().var()
    plt.figure(figsize=(8, 4))
    feature_variance.sort_values(ascending=False).plot(kind='bar', color='teal')
    plt.title(f"Feature Variance Across Clusters ({method_name.upper()})", fontsize=13, fontweight='bold')
    plt.ylabel("Variance of Cluster Means")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
