import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path


# ===================== HELPER =====================

def save_dual_format(save_path):
    """Saves the current figure in both .png and .pdf formats."""
    for ext in ["png", "pdf"]:
        plt.savefig(save_path.with_suffix(f".{ext}"), dpi=300, bbox_inches='tight')
    plt.close()


# ===================== BASIC VISUALS =====================

def plot_correlation_heatmap(df, save_path):
    """Plots correlation matrix for numeric columns only."""
    numeric_df = df.select_dtypes(include=['number'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix (Numeric Columns Only)")
    plt.tight_layout()
    save_dual_format(save_path)


def plot_pca_projection(X_pca, df, color_col, save_path, title=None):
    """2D PCA projection for participants or KMeans clusters with consistent colors."""
    plt.figure(figsize=(8, 6))

    # === Choose color and title logic ===
    if color_col == "td_or_asd":
        palette = {0: "#9FE2BF", 1: "#FF9999"}  # TD=green, ASD=red (soft)
        unique_vals = set(df[color_col].unique())
        if unique_vals == {0, 1}:
            plot_title = "2D Representation of TD/ASD Participants"
        elif unique_vals == {1}:
            plot_title = "2D Representation of ASD Participants"
        elif unique_vals == {0}:
            plot_title = "2D Representation of TD Participants"
        else:
            plot_title = "2D Representation of Participants"
        final_title = plot_title
    elif color_col == "cluster":
        n_clusters = df[color_col].nunique()
        palette = sns.color_palette("tab10", n_clusters)  # Distinct colors for clusters
        plot_title = "2D Representation of KMeans Clusters"
        final_title = title or plot_title
    else:
        palette = "husl"
        plot_title = f"2D Representation of {color_col}"
        final_title = title or plot_title

    # === Plot ===
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1],
        hue=df[color_col],
        palette=palette, s=70, alpha=0.9,
        edgecolor="black", linewidth=0.4
    )

    plt.title(final_title, fontsize=13, fontweight='bold')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(
        title=color_col,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=True, facecolor='white', framealpha=0.6
    )
    plt.tight_layout(pad=1.2)
    save_dual_format(save_path)


# ===================== EVALUATION METRICS =====================

def plot_silhouette_plot(k_values, scores, save_path, title="Silhouette Score vs K"):
    """Silhouette score vs number of clusters."""
    plt.figure(figsize=(6, 4))
    plt.plot(k_values, scores, marker='o', color='teal')
    plt.title(title)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    save_dual_format(save_path)


def plot_db_index_vs_k(k_values, scores, save_path, title="Davies–Bouldin Index vs K"):
    """Davies–Bouldin Index vs number of clusters."""
    plt.figure(figsize=(6, 4))
    plt.plot(k_values, scores, marker='o', color='indianred')
    plt.title(title)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Davies–Bouldin Index")
    plt.tight_layout()
    save_dual_format(save_path)


def plot_ari_nmi_bar(k_values, ari_scores, nmi_scores, save_path):
    """Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) comparison."""
    x = np.arange(len(k_values))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, ari_scores, width, label='ARI', color='steelblue')
    plt.bar(x + width / 2, nmi_scores, width, label='NMI', color='salmon')
    plt.xticks(x, k_values)
    plt.title("ARI vs NMI Across Cluster Sizes")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    save_dual_format(save_path)


# ===================== FEATURE IMPORTANCE VISUALS =====================

def plot_pca_feature_loadings(pca, features, save_path, method_name="kmeans"):
    """Visualizes PCA feature contributions (component loadings)."""
    plt.figure(figsize=(8, 5))
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=features
    )
    sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0)
    plt.title(f"PCA Feature Contributions ({method_name.upper()})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_dual_format(save_path)


def plot_cluster_feature_means(df, features, save_path, method_name="kmeans"):
    """Mean feature values per cluster."""
    cluster_means = df.groupby("cluster")[features].mean()
    plt.figure(figsize=(8, 5))
    sns.heatmap(cluster_means.T, annot=True, cmap="coolwarm")
    plt.title(f"Cluster Centroids by Feature ({method_name.upper()})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_dual_format(save_path)


def plot_feature_variance(df, features, save_path, method_name="kmeans"):
    """Feature variance across clusters."""
    feature_variance = df.groupby("cluster")[features].mean().var()
    plt.figure(figsize=(8, 4))
    feature_variance.sort_values(ascending=False).plot(kind='bar', color='teal')
    plt.title(f"Feature Variance Across Clusters ({method_name.upper()})", fontsize=13, fontweight='bold')
    plt.ylabel("Variance of Cluster Means")
    plt.tight_layout()
    save_dual_format(save_path)


# ===================== STREAMLIT VISUALS =====================

def plot_pca_projection_streamlit(X_pca, df, color_col, save_path, title=None):
    """
    Generates interactive PCA projection for Streamlit and saves as .html.
    Hover shows only: td_or_asd, cluster, PC1, PC2.
    """
    df_plot = df.copy()
    df_plot["PC1"] = X_pca[:, 0]
    df_plot["PC2"] = X_pca[:, 1]

    # === Color and title logic ===
    if color_col == "td_or_asd":
        color_map = {0: "#9FE2BF", 1: "#FF9999"}  # TD = soft green, ASD = soft red
        if df_plot[color_col].nunique() == 1:
            plot_title = "2D Representation of ASD Participants"
        else:
            plot_title = "2D Representation of TD/ASD Participants"
    elif color_col == "cluster":
        color_map = None
        plot_title = "2D Representation of KMeans Clusters"
    else:
        color_map = None
        plot_title = title or f"2D Representation of {color_col}"

    # === Interactive scatter ===
    fig = px.scatter(
        df_plot,
        x="PC1", y="PC2",
        color=color_col,
        color_discrete_map=color_map,
        hover_data={
            "td_or_asd": True if "td_or_asd" in df_plot.columns else False,
            "cluster": True if "cluster" in df_plot.columns else False,
            "PC1": ':.3f',
            "PC2": ':.3f'
        },
        title=plot_title,
        template="plotly_white"
    )

    fig.update_layout(
        height=600,
        title_font=dict(size=16),
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        legend_title=color_col,
    )

    # === Save as .html for Streamlit ===
    html_path = Path(save_path).with_suffix(".html")
    fig.write_html(html_path)
    print(f"✅ Streamlit-ready interactive PCA saved: {html_path}")
