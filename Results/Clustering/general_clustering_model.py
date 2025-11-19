# general_clustering_model.py
import json
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from .data_preprocessor import preprocess_clustering_data
from .visualization import (
    # shared
    plot_correlation_heatmap,
    plot_cluster_feature_means,
    plot_feature_variance,
    # PCA / KMeans / GMM
    plot_pca_projection,
    plot_pca_projection_streamlit,
    plot_pca_projection_3d,
    plot_pca_projection_streamlit_3d,
    plot_silhouette_plot,
    plot_db_index_vs_k,
    plot_ari_nmi_bar,
    plot_pca_feature_loadings,
    plot_pca_cluster_explorer,
    plot_aic_bic_vs_k,
    plot_gmm_uncertainty,
    # t-SNE / HDBSCAN
    plot_tsne_projection,
    plot_tsne_projection_streamlit,
    plot_hdbscan_probability,
    plot_tsne_cluster_explorer,
)

# -------------------------------------------------------------------
# GLOBAL CONFIG (used only when running this file directly)
# -------------------------------------------------------------------
DATA_VERSION = "Data_Clustering"        # folder under data/
MODEL_TYPE = "tsne_hdbscan"             # "pca_kmeans", "pca_gmm", "tsne_hdbscan"
FEATURE_MODE = "5_numeric_nlp"          # "all_numeric", "5_numeric", "5_numeric_nlp",
                                        # "4_numeric_nlp", "custom"
RUN_ID = "run01"
# -------------------------------------------------------------------

# -------------------- FEATURE DEFINITIONS ---------------------------

# 7 scaled numeric features
ALL_SCALED_NUMERIC = [
    "FSR_scaled",
    "BIS_scaled",
    "SRS.Raw_scaled",
    "TDNorm_avg_PE_scaled",
    "overall_avg_PE_scaled",
    "TDnorm_concept_learning_scaled",
    "overall_concept_learning_scaled",
]

# canonical “5 numeric” (same as in V1.2 / V2.2 / V3.2)
CORE_5_SCALED = [
    "FSR_scaled",
    "TDNorm_avg_PE_scaled",
    "overall_avg_PE_scaled",
    "TDnorm_concept_learning_scaled",
    "overall_concept_learning_scaled",
]

# “4 numeric” = above without FSR (no FSR mode)
CORE_4_SCALED = [
    "TDNorm_avg_PE_scaled",
    "overall_avg_PE_scaled",
    "TDnorm_concept_learning_scaled",
    "overall_concept_learning_scaled",
]

# NLP features (same list as V1.2 / V2.2 / V3.2)
NLP_FEATURES = [
    "word_count",
    "sentence_count",
    "char_count",
    "avg_word_length",
    "avg_sentence_length",
    "shortness_score",
    "lexical_diversity",
    "sentiment_polarity",
    "sentiment_subjectivity",
    "positive_word_count",
    "negative_word_count",
    "positive_word_ratio",
    "negative_word_ratio",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
]

# map from scaled numeric to raw numeric for explorer
UNSCALED_MAP = {
    "FSR_scaled": "FSR",
    "BIS_scaled": "BIS",
    "SRS.Raw_scaled": "SRS.Raw",
    "TDNorm_avg_PE_scaled": "TDNorm_avg_PE",
    "overall_avg_PE_scaled": "overall_avg_PE",
    "TDnorm_concept_learning_scaled": "TDnorm_concept_learning",
    "overall_concept_learning_scaled": "overall_concept_learning",
}

# Default custom feature set (overridden by UI when feature_mode="custom")
CUSTOM_FEATURES = CORE_5_SCALED + NLP_FEATURES


# -------------------------------------------------------------------
# FEATURE HELPERS
# -------------------------------------------------------------------

def get_feature_list(mode: str, custom_features: list[str] | None = None) -> list[str]:
    if mode == "all_numeric":
        return ALL_SCALED_NUMERIC
    if mode == "5_numeric":
        return CORE_5_SCALED
    if mode == "5_numeric_nlp":
        return CORE_5_SCALED + NLP_FEATURES
    if mode == "4_numeric_nlp":
        return CORE_4_SCALED + NLP_FEATURES
    if mode == "custom":
        return custom_features or CUSTOM_FEATURES
    raise ValueError(f"Unknown FEATURE_MODE: {mode}")


def get_explorer_numeric_cols(
    mode: str,
    custom_features: list[str] | None = None,
) -> list[str]:
    def map_scaled_to_raw(scaled_list: list[str]) -> list[str]:
        return [UNSCALED_MAP[c] for c in scaled_list if c in UNSCALED_MAP]

    if mode == "all_numeric":
        return map_scaled_to_raw(ALL_SCALED_NUMERIC)
    if mode == "5_numeric":
        return map_scaled_to_raw(CORE_5_SCALED)
    if mode == "5_numeric_nlp":
        return map_scaled_to_raw(CORE_5_SCALED) + NLP_FEATURES
    if mode == "4_numeric_nlp":
        return map_scaled_to_raw(CORE_4_SCALED) + NLP_FEATURES
    if mode == "custom":
        feats = custom_features or CUSTOM_FEATURES
        scaled = [c for c in feats if c.endswith("_scaled")]
        raw_numeric = map_scaled_to_raw(scaled)
        has_nlp = any(f in feats for f in NLP_FEATURES)
        nlp = NLP_FEATURES if has_nlp else []
        return raw_numeric + nlp
    raise ValueError(f"Unknown FEATURE_MODE: {mode}")


# -------------------------------------------------------------------
# MAIN ANALYZER
# -------------------------------------------------------------------

class GeneralClusterAnalyzer:
    """
    Unified clustering pipeline supporting:
      - PCA + KMeans
      - PCA + GMM
      - t-SNE + HDBSCAN

    pca_n_components:
      - None → auto-select via variance explained (>=95%, min 2 PCs)
      - int  → force that many PCs (clipped to [2, max_components])
    """

    def __init__(
        self,
        project_root: Path,
        model_type: str = MODEL_TYPE,
        feature_mode: str = FEATURE_MODE,
        run_id: str = RUN_ID,
        custom_features: list[str] | None = None,
        pca_n_components: int | None = None,
        tsne_n_components: int = 2,
    ):
        self.project_root = project_root
        self.model_type = model_type
        self.feature_mode = feature_mode
        self.run_id = run_id
        self.custom_features = custom_features
        self.pca_n_components = pca_n_components  # None → auto by variance
        # t-SNE dims: allow only 2 or 3, default 2
        if tsne_n_components not in (2, 3):
            print(f"⚠️ Invalid tsne_n_components={tsne_n_components}, falling back to 2.")
            tsne_n_components = 2
        self.tsne_n_components = tsne_n_components

        base_name = f"{self.model_type}_{self.feature_mode}_{self.run_id}"
        self.results_root = (
            self.project_root
            / "Results"
            / "Clustering"
            / "general"
            / base_name
        )

        self.td_asd_dir = self.results_root / "td_asd_clusters"
        self.asd_dir = self.results_root / "asd_subclusters"

        self.td_asd_dir.mkdir(parents=True, exist_ok=True)
        self.asd_dir.mkdir(parents=True, exist_ok=True)

    # ------------------ PUBLIC PIPELINE -------------------

    def run_pipeline(self, df: pd.DataFrame):
        print("\n[1] Correlation Heatmap (full TD+ASD)...")
        plot_correlation_heatmap(df, self.td_asd_dir / "correlation_matrix.png")

        print(f"[2] {self.model_type} on full TD+ASD...")
        self._run_backend(df, save_dir=self.td_asd_dir)

        print("[3] ASD-only subclustering...")
        asd_df = df[df["td_or_asd"] == 1].reset_index(drop=True)
        if asd_df.empty:
            print("   ⚠️ No ASD rows found; skipping ASD-only clustering.")
        else:
            self._run_backend(asd_df, save_dir=self.asd_dir)

        print(
            f"\n✅ Clustering pipeline completed for "
            f"{self.model_type} / {self.feature_mode} / {self.run_id}\n"
        )

    # ------------------ BACKEND DISPATCH -------------------

    def _run_backend(self, df: pd.DataFrame, save_dir: Path):
        if self.model_type == "pca_kmeans":
            self._run_pca_kmeans(df, save_dir=save_dir)
        elif self.model_type == "pca_gmm":
            self._run_pca_gmm(df, save_dir=save_dir)
        elif self.model_type == "tsne_hdbscan":
            self._run_tsne_hdbscan(df, save_dir=save_dir)
        else:
            raise ValueError(f"Unknown MODEL_TYPE: {self.model_type}")

    # ----------------------------------------------------------------
    # HELPER: full PCA + auto component selection
    # ----------------------------------------------------------------
    def _run_full_pca(self, X: np.ndarray, proj_dir: Path):
        """
        Fit PCA with maximum rank and:
          - save variance curve
          - auto-pick #PCs if self.pca_n_components is None
        Returns:
          pca        : fitted PCA object (full)
          X_full     : full PC scores (all comps)
          n_selected : #PCs chosen for clustering
        """
        max_components = min(X.shape[0], X.shape[1])
        print(f"   → Running full PCA with n_components={max_components}...")
        pca = PCA(n_components=max_components, random_state=42)
        X_full = pca.fit_transform(X)

        # variance curve
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        plt.figure(figsize=(6, 4))
        plt.plot(cum_var, marker="o")
        plt.xticks(
            ticks=range(len(cum_var)),
            labels=[str(i + 1) for i in range(len(cum_var))]
        )
        plt.axhline(0.99, color="gray", linestyle="--", linewidth=1)
        plt.title("Cumulative Explained Variance by PCA Components")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.tight_layout()
        plt.savefig(proj_dir / "pca_variance.png", dpi=300)
        plt.savefig(proj_dir / "pca_variance.pdf", dpi=300)
        plt.close()

        # choose n_components
        if self.pca_n_components is not None:
            n_selected = int(max(2, min(self.pca_n_components, max_components)))
            reason = f"manual override (user-selected: {self.pca_n_components})"
        else:
            # smallest k s.t. cum_var[k-1] >= 0.98, but at least 2 PCs
            auto_k = int(np.searchsorted(cum_var, 0.99) + 1)
            n_selected = max(2, min(auto_k, max_components))
            reason = f"auto-selected to reach ≥95% variance (k={auto_k})"

        print(
            f"   → PCA components selected for clustering: {n_selected} "
            f"({reason})"
        )

        return pca, X_full, n_selected, cum_var

    # ----------------------------------------------------------------
    # PCA + KMEANS
    # ----------------------------------------------------------------
    def _run_pca_kmeans(self, df: pd.DataFrame, save_dir: Path, cluster_range=(2, 6)):
        features = get_feature_list(self.feature_mode, self.custom_features)
        explorer_numeric_cols = get_explorer_numeric_cols(
            self.feature_mode, self.custom_features
        )

        print("\n📊 [PCA + KMeans] Selected features:")
        for f in features:
            print(f"   • {f}")
        print(f"Total features: {len(features)}")

        target_col = "td_or_asd"
        X = df[features].values

        proj_dir = save_dir / "projections"
        metric_dir = save_dir / "metrics_visuals"
        feat_dir = save_dir / "feature_visuals"
        for d in [proj_dir, metric_dir, feat_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # === full PCA + variance-based selection ===
        pca, X_full, n_pcs_selected, cum_var = self._run_full_pca(X, proj_dir)

        # representation for clustering (k PCs)
        X_clust = X_full[:, :n_pcs_selected]

        # 2D & 3D coords for visualization
        X_2d = X_full[:, :2]
        X_3d = X_full[:, :3] if n_pcs_selected >= 3 else None


        # === Evaluate K range on chosen PC space ===
        silhouette_scores, db_scores, ari_scores, nmi_scores = [], [], [], []
        print("   → Evaluating KMeans over k range...")
        k_values = list(range(cluster_range[0], cluster_range[1] + 1))
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(X_clust)
            silhouette_scores.append(silhouette_score(X_clust, labels))
            db_scores.append(davies_bouldin_score(X_clust, labels))
            if target_col in df.columns:
                ari_scores.append(adjusted_rand_score(df[target_col], labels))
                nmi_scores.append(normalized_mutual_info_score(df[target_col], labels))

        plot_silhouette_plot(
            k_values,
            silhouette_scores,
            metric_dir / "silhouette_plot_kmeans.png",
            title="Silhouette Score vs K (KMEANS)",
        )
        plot_db_index_vs_k(
            k_values,
            db_scores,
            metric_dir / "db_index_vs_k_kmeans.png",
            title="Davies–Bouldin Index vs K (KMEANS)",
        )
        if ari_scores and nmi_scores:
            plot_ari_nmi_bar(
                k_values,
                ari_scores,
                nmi_scores,
                metric_dir / "ari_nmi_bar_kmeans.png",
            )

        best_k = k_values[int(np.argmax(silhouette_scores))]
        print(f"   ✓ Best k (Silhouette): {best_k}")
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X_clust)
        df["cluster"] = labels

        # === Cluster distribution ===
        print("   → Plotting cluster distribution...")
        plt.figure(figsize=(7, 4))
        cluster_counts = df["cluster"].value_counts().sort_index()
        colors = [sns.color_palette("tab10")[i % 10] for i, _ in enumerate(cluster_counts.index)]
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette=colors)
        plt.title("KMeans Cluster Distribution", fontsize=13, fontweight="bold")
        plt.xlabel("Cluster Label")
        plt.ylabel("Number of Points")
        plt.tight_layout()
        plt.savefig(proj_dir / "kmeans_cluster_distribution.png", dpi=300)
        plt.savefig(proj_dir / "kmeans_cluster_distribution.pdf", dpi=300)
        plt.close()

        # === Feature importance visuals ===
        plot_pca_feature_loadings(
            pca,
            features,
            feat_dir / "pca_feature_loadings_kmeans.png",
            "kmeans",
        )
        plot_cluster_feature_means(
            df,
            features,
            feat_dir / "cluster_feature_means_kmeans.png",
            "kmeans",
        )
        plot_feature_variance(
            df,
            features,
            feat_dir / "feature_variance_kmeans.png",
            "kmeans",
        )

        # === Static 2D PCA projections ===
        if target_col in df.columns:
            plot_pca_projection(
                X_2d,
                df,
                target_col,
                proj_dir / "pca_target_projection_kmeans.png",
                method_name="pca_kmeans",
            )
        plot_pca_projection(
            X_2d,
            df,
            "cluster",
            proj_dir / "pca_cluster_projection_kmeans.png",
            method_name="pca_kmeans",
        )

        # === Static 3D PCA projections (if ≥3 PCs) ===
        if X_3d is not None:
            plot_pca_projection_3d(
                X_3d,
                df,
                "cluster",
                proj_dir / "pca_cluster_projection_kmeans_3d.png",
                method_name="pca_kmeans",
            )
            if target_col in df.columns:
                plot_pca_projection_3d(
                    X_3d,
                    df,
                    target_col,
                    proj_dir / "pca_target_projection_kmeans_3d.png",
                    method_name="pca_kmeans",
                )

        # === Interactive 2D PCA projections (for Streamlit) ===
        plot_pca_projection_streamlit(
            X_2d,
            df,
            color_col="cluster",
            save_path=proj_dir / "pca_cluster_projection_kmeans_2d.html",
            method_name="pca_kmeans",
        )
        if target_col in df.columns:
            plot_pca_projection_streamlit(
                X_2d,
                df,
                color_col=target_col,
                save_path=proj_dir / "pca_target_projection_kmeans_2d.html",
                method_name="pca_kmeans",
            )

        # === Interactive 3D PCA (for Streamlit, if ≥3 PCs) ===
        if X_3d is not None:
            plot_pca_projection_streamlit_3d(
                X_3d,
                df,
                color_col="cluster",
                save_path=proj_dir / "pca_cluster_projection_kmeans_3d.html",
                method_name="pca_kmeans",
            )
            if target_col in df.columns:
                plot_pca_projection_streamlit_3d(
                    X_3d,
                    df,
                    color_col=target_col,
                    save_path=proj_dir / "pca_target_projection_kmeans_3d.html",
                    method_name="pca_kmeans",
                )

        # === Save processed CSV with PCs used for clustering ===
        print("   → Saving processed dataset with PCA + KMeans clusters...")
        df_final = df.copy()
        for i in range(n_pcs_selected):
            df_final[f"PC{i + 1}"] = X_full[:, i]
        df_final.to_csv(save_dir / "processed_with_clusters.csv", index=False)
        print(f"      Saved full processed dataset → {save_dir / 'processed_with_clusters.csv'}")

        # === Interactive PCA cluster explorer  ===
        print("   → Building interactive PCA cluster explorer (KMeans)...")
        plot_pca_cluster_explorer(
            X_clust,
            df,
            numeric_cols=explorer_numeric_cols,
            save_path=proj_dir / "pca_cluster_explorer_kmeans.html",
            method_name="kmeans",
        )
        print("      Interactive cluster explorer saved.")

        # === Metrics summary ===
        metrics = {
            "data_version": DATA_VERSION,
            "model_type": self.model_type,
            "feature_mode": self.feature_mode,
            "method": "kmeans",
            "k_range": k_values,
            "silhouette_scores": silhouette_scores,
            "db_index_scores": db_scores,
            "ari_scores": ari_scores,
            "nmi_scores": nmi_scores,
            "best_k": int(best_k),
            "pca_params": {
                "max_components": int(X_full.shape[1]),
                "chosen_n_components": int(n_pcs_selected),
                "variance_threshold": 0.95,
                "cum_explained_at_chosen": float(cum_var[n_pcs_selected - 1]),
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            },
            "kmeans_params": {
                "n_clusters": int(best_k),
                "random_state": 42,
            },
            "cluster_counts": cluster_counts.to_dict(),
        }
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"✓ Metrics + visuals saved in {save_dir}")

    # ----------------------------------------------------------------
    # PCA + GMM
    # ----------------------------------------------------------------
    def _run_pca_gmm(self, df: pd.DataFrame, save_dir: Path):
        features = get_feature_list(self.feature_mode, self.custom_features)
        explorer_numeric_cols = get_explorer_numeric_cols(
            self.feature_mode, self.custom_features
        )

        print("\n📊 [PCA + GMM] Selected features:")
        for f in features:
            print(f"   • {f}")
        print(f"Total features: {len(features)}")

        target_col = "td_or_asd"
        X = df[features].values

        proj_dir = save_dir / "projections"
        metric_dir = save_dir / "metrics_visuals"
        feat_dir = save_dir / "feature_visuals"
        for d in [proj_dir, metric_dir, feat_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # === full PCA + variance-based selection ===
        pca, X_full, n_pcs_selected, cum_var = self._run_full_pca(X, proj_dir)

        X_clust = X_full[:, :n_pcs_selected]
        X_2d = X_full[:, :2]
        X_3d = X_full[:, :3] if n_pcs_selected >= 3 else None


        # === GMM model selection via BIC on chosen PC space ===
        n_components_range = range(2, 9)
        bic_scores, aic_scores, gmms = [], [], []
        print("   → Searching for optimal GMM cluster count using BIC...")

        for n in n_components_range:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type="full",
                random_state=42,
                init_params="kmeans",
                n_init=5,
            )
            gmm.fit(X_clust)
            bic_scores.append(gmm.bic(X_clust))
            aic_scores.append(gmm.aic(X_clust))
            gmms.append(gmm)

        best_idx = int(np.argmin(bic_scores))
        best_k = list(n_components_range)[best_idx]
        best_gmm = gmms[best_idx]

        labels = best_gmm.predict(X_clust)
        responsibilities = best_gmm.predict_proba(X_clust)
        df["cluster"] = labels

        print(f"   ✓ Best number of clusters (BIC): {best_k}")

        # === AIC/BIC plot ===
        plot_aic_bic_vs_k(
            list(n_components_range),
            aic_scores,
            bic_scores,
            metric_dir / "aic_bic_vs_k_gmm.png",
            method_name="gmm",
        )

        # === cluster distribution ===
        print("   → Plotting cluster distribution...")
        plt.figure(figsize=(7, 4))
        cluster_counts = df["cluster"].value_counts().sort_index()
        colors = [sns.color_palette("tab10")[i % 10] for i, _ in enumerate(cluster_counts.index)]
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette=colors)
        plt.title("GMM Cluster Distribution", fontsize=13, fontweight="bold")
        plt.xlabel("Cluster Label")
        plt.ylabel("Number of Points")
        plt.tight_layout()
        plt.savefig(proj_dir / "gmm_cluster_distribution.png", dpi=300)
        plt.savefig(proj_dir / "gmm_cluster_distribution.pdf", dpi=300)
        plt.close()

        # === PCA projections (2D static) ===
        plot_pca_projection(
            X_2d,
            df,
            "cluster",
            proj_dir / "pca_cluster_projection_gmm.png",
            title="2D Representation of GMM Clusters",
            method_name="pca_gmm",
        )
        if target_col in df.columns:
            title = (
                "2D Representation of ASD Participants"
                if df[target_col].nunique() == 1
                else "2D Representation of TD/ASD Participants"
            )
            plot_pca_projection(
                X_2d,
                df,
                target_col,
                proj_dir / "pca_target_projection_gmm.png",
                title=title,
                method_name="pca_gmm",
            )

        # === PCA projections (3D static, if ≥3 PCs) ===
        if X_3d is not None:
            plot_pca_projection_3d(
                X_3d,
                df,
                "cluster",
                proj_dir / "pca_cluster_projection_gmm_3d.png",
                method_name="pca_gmm",
            )
            if target_col in df.columns:
                three_d_title = (
                    "3D Representation of ASD Participants"
                    if df[target_col].nunique() == 1
                    else "3D Representation of TD/ASD Participants"
                )
                plot_pca_projection_3d(
                    X_3d,
                    df,
                    target_col,
                    proj_dir / "pca_target_projection_gmm_3d.png",
                    title=three_d_title,
                    method_name="pca_gmm",
                )

        # === 2D interactive (Streamlit) ===
        plot_pca_projection_streamlit(
            X_2d,
            df,
            color_col="cluster",
            save_path=proj_dir / "pca_cluster_projection_gmm_2d.html",
            method_name="pca_gmm",
        )
        if target_col in df.columns:
            plot_pca_projection_streamlit(
                X_2d,
                df,
                color_col=target_col,
                save_path=proj_dir / "pca_target_projection_gmm_2d.html",
                method_name="pca_gmm",
            )

        # === 3D interactive (Streamlit, if ≥3 PCs) ===
        if X_3d is not None:
            plot_pca_projection_streamlit_3d(
                X_3d,
                df,
                color_col="cluster",
                save_path=proj_dir / "pca_cluster_projection_gmm_3d.html",
                method_name="pca_gmm",
            )
            if target_col in df.columns:
                plot_pca_projection_streamlit_3d(
                    X_3d,
                    df,
                    color_col=target_col,
                    save_path=proj_dir / "pca_target_projection_gmm_3d.html",
                    method_name="pca_gmm",
                )

        # uncertainty map (2D view)
        plot_gmm_uncertainty(
            X_2d,
            responsibilities,
            proj_dir / "pca_gmm_uncertainty.png",
        )

        # === feature visuals ===
        plot_pca_feature_loadings(
            pca,
            features,
            feat_dir / "pca_feature_loadings_gmm.png",
            "gmm",
        )
        plot_cluster_feature_means(
            df,
            features,
            feat_dir / "cluster_feature_means_gmm.png",
            "gmm",
        )
        plot_feature_variance(
            df,
            features,
            feat_dir / "feature_variance_gmm.png",
            "gmm",
        )

        # explorer (2D)
        print("   → Building interactive PCA GMM cluster explorer...")
        plot_pca_cluster_explorer(
            X_clust,
            df,
            numeric_cols=explorer_numeric_cols,
            save_path=proj_dir / "pca_cluster_explorer_gmm.html",
            method_name="gmm",
        )
        print("      Interactive PCA GMM cluster explorer saved.")

        # save CSV
        print("   → Saving processed dataset with PCA + GMM clusters...")
        df_final = df.copy()
        for i in range(n_pcs_selected):
            df_final[f"PC{i + 1}"] = X_full[:, i]
        df_final.to_csv(save_dir / "processed_with_clusters.csv", index=False)
        print(f"      Saved full processed dataset → {save_dir / 'processed_with_clusters.csv'}")

        # metrics
        metrics = {
            "data_version": DATA_VERSION,
            "model_type": self.model_type,
            "feature_mode": self.feature_mode,
            "method": "gmm",
            "tested_k_range": list(n_components_range),
            "aic_scores": aic_scores,
            "bic_scores": bic_scores,
            "best_k": int(best_k),
            "pca_params": {
                "max_components": int(X_full.shape[1]),
                "chosen_n_components": int(n_pcs_selected),
                "variance_threshold": 0.95,
                "cum_explained_at_chosen": float(cum_var[n_pcs_selected - 1]),
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            },
            "gmm_params": {
                "covariance_type": "full",
                "init_params": "kmeans",
                "n_init": 5,
            },
            "cluster_counts": cluster_counts.to_dict(),
        }
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"✓ Metrics + visuals saved in {save_dir}")

    # ----------------------------------------------------------------
    # t-SNE + HDBSCAN
    # ----------------------------------------------------------------
    def _run_tsne_hdbscan(self, df: pd.DataFrame, save_dir: Path):
        features = get_feature_list(self.feature_mode, self.custom_features)
        explorer_numeric_cols = get_explorer_numeric_cols(
            self.feature_mode, self.custom_features
        )

        print("\n📊 [t-SNE + HDBSCAN] Selected features:")
        for f in features:
            print(f"   • {f}")
        print(f"Total features: {len(features)}")

        target_col = "td_or_asd"
        X = df[features].values

        proj_dir = save_dir / "projections"
        feat_dir = save_dir / "feature_visuals"
        for d in [proj_dir, feat_dir]:
            d.mkdir(parents=True, exist_ok=True)

        print("   → Running t-SNE embedding (may take several minutes)...")
        n_components = getattr(self, "tsne_n_components", 2)
        tsne = TSNE(
            n_components=n_components,
            perplexity=30,
            learning_rate=200,
            max_iter=2000,
            random_state=42,
            verbose=1,
        )

        X_tsne = tsne.fit_transform(X)

        print("   → Running HDBSCAN clustering...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
        labels = clusterer.fit_predict(X_tsne)
        df["cluster"] = labels

        unique_clusters = np.unique(labels)
        n_clusters = int(len(unique_clusters[unique_clusters != -1]))
        n_noise = int(np.sum(labels == -1))
        print(f"   ✓ Found {n_clusters} clusters (+ {n_noise} noise points)")

        print("   → Plotting cluster distribution...")
        plt.figure(figsize=(7, 4))
        cluster_counts = df["cluster"].value_counts().sort_index()
        colors = [
            "#999999" if idx == -1 else sns.color_palette("tab10")[i % 10]
            for i, idx in enumerate(cluster_counts.index)
        ]
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette=colors)
        plt.title("HDBSCAN Cluster Distribution (including Noise)", fontsize=13, fontweight="bold")
        plt.xlabel("Cluster Label (-1 = Noise)")
        plt.ylabel("Number of Points")
        plt.tight_layout()
        plt.savefig(proj_dir / "hdbscan_cluster_distribution.png", dpi=300)
        plt.savefig(proj_dir / "hdbscan_cluster_distribution.pdf", dpi=300)
        plt.close()

        plot_tsne_projection(
            X_tsne,
            df,
            "cluster",
            proj_dir / "tsne_hdbscan_clusters.png",
            title="HDBSCAN Clusters (2D t-SNE Projection)",
        )
        if target_col in df.columns:
            plot_tsne_projection(
                X_tsne,
                df,
                target_col,
                proj_dir / "tsne_target_projection.png",
                title="TD/ASD Participants (t-SNE Projection)",
            )

        plot_hdbscan_probability(
            X_tsne,
            clusterer,
            proj_dir / "hdbscan_probability_map.png",
        )

        plot_tsne_projection_streamlit(
            X_tsne,
            df,
            color_col="cluster",
            save_path=proj_dir / "tsne_hdbscan_clusters.html",
        )
        if target_col in df.columns:
            plot_tsne_projection_streamlit(
                X_tsne,
                df,
                color_col=target_col,
                save_path=proj_dir / "tsne_target_projection.html",
            )

        plot_cluster_feature_means(
            df,
            features,
            feat_dir / "cluster_feature_means_hdbscan.png",
            "hdbscan",
        )
        plot_feature_variance(
            df,
            features,
            feat_dir / "feature_variance_hdbscan.png",
            "hdbscan",
        )

        df_final = df.copy()
        if X_tsne is not None:
            df_final["tSNE1"] = X_tsne[:, 0]
            df_final["tSNE2"] = X_tsne[:, 1]
            if X_tsne.shape[1] >= 3:
                df_final["tSNE3"] = X_tsne[:, 2]
        df_final.to_csv(save_dir / "processed_with_clusters.csv", index=False)
        print(f"Saved full processed dataset → {save_dir / 'processed_with_clusters.csv'}")

        print("   → Building interactive t-SNE cluster explorer...")
        plot_tsne_cluster_explorer(
            X_tsne,
            df,
            numeric_cols=explorer_numeric_cols,
            save_path=proj_dir / "tsne_cluster_explorer_hdbscan.html",
        )
        print("      Interactive t-SNE cluster explorer saved.")

        metrics = {
            "data_version": DATA_VERSION,
            "model_type": self.model_type,
            "feature_mode": self.feature_mode,
            "method": "hdbscan",
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "unique_labels": unique_clusters.tolist(),
            "tsne_params": {
                     "n_components": int(n_components),
                     "perplexity": 30,
                     "learning_rate": 200,
                     "n_iter": 2000,
            },
            "hdbscan_params": {
                "min_cluster_size": 15,
                "min_samples": 5,
            },
            "cluster_counts": cluster_counts.to_dict(),
        }
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"✓ Metrics + visuals saved in {save_dir}")


# -------------------------------------------------------------------
# ENTRY POINT (for running this file directly)
# -------------------------------------------------------------------
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent

    print(
        f"\n=== Running General Clustering Pipeline ===\n"
        f"MODEL_TYPE   : {MODEL_TYPE}\n"
        f"FEATURE_MODE : {FEATURE_MODE}\n"
        f"RUN_ID       : {RUN_ID}\n"
    )

    preprocessed_path = (
        project_root
        / "data"
        / DATA_VERSION
        / "data_preprocessed_general.csv"
    )

    if preprocessed_path.exists():
        print(f"Found preprocessed file → {preprocessed_path}")
        df_clean = pd.read_csv(preprocessed_path)
    else:
        print("Preprocessed file NOT found — running preprocessing automatically...")
        df_clean = preprocess_clustering_data(project_root)

    analyzer = GeneralClusterAnalyzer(
        project_root,
        model_type=MODEL_TYPE,
        feature_mode=FEATURE_MODE,
        run_id=RUN_ID,
        custom_features=None,
        pca_n_components=None,
        tsne_n_components=2,  # or 3 if you want to test 3D from CLI
    )

    analyzer.run_pipeline(df_clean)
