import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import json
import matplotlib.pyplot as plt
from data_preprocessor import preprocess_clustering_data
from visualization import (
    plot_correlation_heatmap,
    plot_pca_projection,
    plot_aic_bic_vs_k,
    plot_gmm_uncertainty,
    plot_pca_feature_loadings,
    plot_cluster_feature_means,
    plot_feature_variance
)

# ------------------------------------------------------------
data_version = "Data_Clustering_v2"
model_version = "Clustering_V2"
# ------------------------------------------------------------


class ClusterAnalyzerV2:
    """Full PCA + GMM pipeline with automatic cluster selection (via BIC)."""

    def __init__(self, project_root):
        self.project_root = project_root
        self.results_dir = self.project_root / "Results" / model_version / "visualizations"
        self.td_asd_dir = self.results_dir / "td_asd_clusters"
        self.asd_dir = self.results_dir / "asd_subclusters"
        self.td_asd_dir.mkdir(parents=True, exist_ok=True)
        self.asd_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    def run_pipeline(self, df):
        """Executes full clustering analysis."""
        print("\n[1] Correlation Heatmap...")
        plot_correlation_heatmap(df, self.td_asd_dir / "correlation_matrix.png")

        print("[2] PCA + GMM (TD + ASD)...")
        self.run_pca_gmm(df, save_dir=self.td_asd_dir)

        print("[3] ASD-only Subclustering (GMM)...")
        asd_df = df[df["td_or_asd"] == 1].reset_index(drop=True)
        self.run_pca_gmm(asd_df, save_dir=self.asd_dir)

        print("\n✅ Clustering_V2 pipeline completed successfully.\n")

    # --------------------------------------------------------
    def run_pca_gmm(self, df, save_dir, method_name="gmm"):
        """Runs PCA + GMM with automatic selection of k using BIC."""
        features = [col for col in df.columns if col not in ['td_or_asd', 'cluster']]

        target_col = "td_or_asd"
        X = df[features].values

        # === PCA Transformation ===
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # === PCA Variance Plot ===
        plt.figure(figsize=(6, 4))
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
        plt.title("Explained Variance by PCA Components")
        plt.xlabel("Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.tight_layout()
        plt.savefig(save_dir / "pca_variance.png", dpi=300)
        plt.close()

        # === Automatic GMM Cluster Selection (via BIC) ===
        n_components_range = range(2, 9)
        bic_scores, aic_scores, gmms = [], [], []

        print("   → Searching for optimal GMM cluster count using BIC...")
        for n in n_components_range:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type="full",
                random_state=42,
                init_params="kmeans",
                n_init=5
            )
            gmm.fit(X_pca)
            bic_scores.append(gmm.bic(X_pca))
            aic_scores.append(gmm.aic(X_pca))
            gmms.append(gmm)

        best_idx = np.argmin(bic_scores)
        best_k = n_components_range[best_idx]
        best_gmm = gmms[best_idx]
        labels = best_gmm.predict(X_pca)
        responsibilities = best_gmm.predict_proba(X_pca)

        df["cluster"] = labels

        print(f"   ✓ Best number of clusters (BIC): {best_k}")

        # === Save AIC/BIC Plot ===
        plot_aic_bic_vs_k(
            list(n_components_range), aic_scores, bic_scores,
            save_dir / f"aic_bic_vs_k_{method_name}.png",
            method_name
        )

        # === PCA Projections ===
        plot_pca_projection(
            X_pca, df, "cluster",
            save_dir / f"pca_{method_name}_cluster_projection.png",
            title=f"PCA Projection colored by {method_name.upper()} Clusters"
        )

        if target_col in df.columns:
            plot_pca_projection(
                X_pca, df, target_col,
                save_dir / f"pca_target_projection_{method_name}.png",
                title=f"PCA Projection colored by Target ({method_name.upper()})"
            )

        # === GMM Uncertainty Visualization ===
        plot_gmm_uncertainty(X_pca, responsibilities, save_dir / f"pca_{method_name}_uncertainty.png")

        # === Feature Importance Visualizations ===
        plot_pca_feature_loadings(pca, features, save_dir / f"pca_feature_loadings_{method_name}.png", method_name)
        plot_cluster_feature_means(df, features, save_dir / f"cluster_feature_means_{method_name}.png", method_name)
        plot_feature_variance(df, features, save_dir / f"feature_variance_{method_name}.png", method_name)

        # === Save Metrics Summary ===
        metrics = {
            "method": method_name,
            "tested_k_range": list(n_components_range),
            "aic_scores": aic_scores,
            "bic_scores": bic_scores,
            "best_k": int(best_k),
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist()
        }
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"✓ Metrics + visuals saved in {save_dir}")


# ---------------------- ENTRY POINT ----------------------
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent

    print("\n=== Running Clustering_V2 Pipeline ===\n")
    df_clean = preprocess_clustering_data(project_root)
    analyzer = ClusterAnalyzerV2(project_root)
    analyzer.run_pipeline(df_clean)
