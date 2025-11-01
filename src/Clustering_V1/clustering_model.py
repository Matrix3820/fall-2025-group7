"""
Main Clustering_V1 pipeline:
Runs preprocessing, PCA + KMeans clustering, metric evaluation, and all visualizations.
Just run this script — everything else is imported and executed automatically.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
import json
import matplotlib.pyplot as plt
from data_preprocessor import preprocess_clustering_data
from visualization import (
    plot_correlation_heatmap,
    plot_pca_projection,
    plot_silhouette_plot,
    plot_db_index_vs_k,
    plot_ari_nmi_bar,
    plot_pca_feature_loadings,
    plot_cluster_feature_means,
    plot_feature_variance
)


data_version = "Data_Clustering_v1"
model_version = "Clustering_V1"


class ClusterAnalyzer:
    """Complete unsupervised clustering pipeline (numeric features only)."""

    def __init__(self, project_root):
        self.project_root = project_root
        self.results_dir = self.project_root / "Results" / model_version / "visualizations"
        self.td_asd_dir = self.results_dir / "td_asd_clusters"
        self.asd_dir = self.results_dir / "asd_subclusters"
        self.td_asd_dir.mkdir(parents=True, exist_ok=True)
        self.asd_dir.mkdir(parents=True, exist_ok=True)

    def run_pipeline(self, df):
        print("\n[1] Correlation Heatmap...")
        plot_correlation_heatmap(df, self.td_asd_dir / "correlation_matrix.png")

        print("[2] PCA + KMeans (TD + ASD)...")
        self.run_pca_kmeans(df, save_dir=self.td_asd_dir, cluster_range=(2, 6))

        print("[3] ASD-only Subclustering...")
        asd_df = df[df["td_or_asd"] == 1]
        self.run_pca_kmeans(asd_df, save_dir=self.asd_dir, cluster_range=(2, 6))
        print("\n✅ Clustering pipeline completed successfully.\n")

    def run_pca_kmeans(self, df, save_dir, cluster_range=(2, 6),method_name="kmeans"):
        features = [
            'FSR', 'BIS', 'SRS.Raw', 'TDNorm_avg_PE',
            'overall_avg_PE', 'TDnorm_concept_learning', 'overall_concept_learning'
        ]
        target_col = "td_or_asd"
        X = df[features].values

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # PCA variance plot
        plt.figure(figsize=(6, 4))
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
        plt.title("Explained Variance by PCA Components")
        plt.xlabel("Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.tight_layout()
        plt.savefig(save_dir / "pca_variance.png", dpi=300)
        plt.close()

        silhouette_scores, db_scores, ari_scores, nmi_scores = [], [], [], []

        for k in range(cluster_range[0], cluster_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_pca)
            silhouette_scores.append(silhouette_score(X_pca, labels))
            db_scores.append(davies_bouldin_score(X_pca, labels))
            if target_col in df.columns:
                ari_scores.append(adjusted_rand_score(df[target_col], labels))
                nmi_scores.append(normalized_mutual_info_score(df[target_col], labels))

        # Metric plots
        k_values = list(range(cluster_range[0], cluster_range[1] + 1))

        plot_silhouette_plot(k_values, silhouette_scores,
                             save_dir / f"silhouette_plot_{method_name}.png",
                             title=f"Silhouette Score vs K ({method_name.upper()})")

        plot_db_index_vs_k(k_values, db_scores,
                           save_dir / f"db_index_vs_k_{method_name}.png",
                           title=f"Davies–Bouldin Index vs K ({method_name.upper()})")

        if ari_scores and nmi_scores:
            plot_ari_nmi_bar(k_values, ari_scores, nmi_scores, save_dir / "ari_nmi_bar.png")

        best_k = range(cluster_range[0], cluster_range[1] + 1)[np.argmax(silhouette_scores)]
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        df["cluster"] = kmeans.fit_predict(X_pca)

        # === Feature Importance Visualizations ===


        # PCA component loadings
        plot_pca_feature_loadings(pca, features, save_dir / f"pca_feature_loadings_{method_name}.png", method_name)

        # Mean feature values per cluster
        plot_cluster_feature_means(df, features, save_dir / f"cluster_feature_means_{method_name}.png", method_name)

        # Variance of features across clusters
        plot_feature_variance(df, features, save_dir / f"feature_variance_{method_name}.png", method_name)

        plot_pca_projection(X_pca, df, target_col,
                            save_dir / f"pca_target_projection_{method_name}.png",
                            title=f"PCA Projection colored by Target ({method_name.upper()})")

        plot_pca_projection(X_pca, df, "cluster",
                            save_dir / f"pca_{method_name}_cluster_projection.png",
                            title=f"PCA Projection colored by {method_name.upper()} Clusters")

        metrics = {
            "k_range": list(range(cluster_range[0], cluster_range[1] + 1)),
            "silhouette_scores": silhouette_scores,
            "db_index_scores": db_scores,
            "ari_scores": ari_scores,
            "nmi_scores": nmi_scores,
            "best_k": int(best_k)
        }
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Metrics + visuals saved in {save_dir}")


# ---------------------- ENTRY POINT ----------------------
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent

    print("\n=== Running Clustering_V1 Pipeline ===\n")
    df_clean = preprocess_clustering_data(project_root)
    analyzer = ClusterAnalyzer(project_root)
    analyzer.run_pipeline(df_clean)
