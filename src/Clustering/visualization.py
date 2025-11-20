import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D static PCA plots


# ===================== HELPER =====================

def save_dual_format(save_path):
    """
    Saves the current figure in both .png and .pdf formats.
    Example: input '.../figure.png' → saves 'figure.png' and 'figure.pdf'
    """
    save_path = Path(save_path)
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


def plot_pca_projection(
    X_pca,
    df,
    color_col,
    save_path,
    title=None,
    method_name=None,
):
    """
    2D PCA projection for participants or clusters with consistent colors.

    method_name:
        - "pca_kmeans"  → title mentions KMeans
        - "pca_gmm"     → title mentions GMM
        - None / other  → generic wording
    """
    plt.figure(figsize=(8, 6))

    # === Color & title logic ===
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
    elif color_col == "cluster":
        n_clusters = df[color_col].nunique()
        palette = sns.color_palette("tab20", n_clusters)  # consistent style
        if method_name == "pca_kmeans":
            plot_title = "2D Representation of KMeans Clusters"
        elif method_name == "pca_gmm":
            plot_title = "2D Representation of GMM Clusters"
        else:
            plot_title = "2D Representation of Clusters"
    else:
        palette = "husl"
        plot_title = f"2D Representation of {color_col}"

    final_title = title or plot_title

    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1],
        hue=df[color_col],
        palette=palette,
        s=70, alpha=0.9,
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


def plot_pca_projection_3d(
    X_pca,
    df,
    color_col,
    save_path,
    title=None,
    method_name=None,
):
    """
    3D PCA projection using PC1, PC2, PC3.
    Used when you have at least 3 components and want a 3D view.
    """
    if X_pca.shape[1] < 3:
        raise ValueError("X_pca must have at least 3 components for 3D projection.")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # === Color & title logic (same as 2D) ===
    if color_col == "td_or_asd":
        palette = {0: "#9FE2BF", 1: "#FF9999"}
        unique_vals = set(df[color_col].unique())
        if unique_vals == {0, 1}:
            plot_title = "3D Representation of TD/ASD Participants"
        elif unique_vals == {1}:
            plot_title = "3D Representation of ASD Participants"
        elif unique_vals == {0}:
            plot_title = "3D Representation of TD Participants"
        else:
            plot_title = "3D Representation of Participants"
        colors = df[color_col].map(palette)
    elif color_col == "cluster":
        n_clusters = df[color_col].nunique()
        base_palette = sns.color_palette("tab20", n_clusters)
        color_dict = {c: base_palette[i % len(base_palette)] for i, c in enumerate(sorted(df[color_col].unique()))}
        colors = df[color_col].map(color_dict)
        if method_name == "pca_kmeans":
            plot_title = "3D Representation of KMeans Clusters"
        elif method_name == "pca_gmm":
            plot_title = "3D Representation of GMM Clusters"
        else:
            plot_title = "3D Representation of Clusters"
    else:
        # fallback: use a generic palette
        unique_vals = df[color_col].unique()
        base_palette = sns.color_palette("husl", len(unique_vals))
        color_dict = {v: base_palette[i] for i, v in enumerate(unique_vals)}
        colors = df[color_col].map(color_dict)
        plot_title = f"3D Representation of {color_col}"

    final_title = title or plot_title

    ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        X_pca[:, 2],
        c=colors,
        s=40,
        edgecolors="black",
        linewidths=0.4,
        alpha=0.9,
    )

    ax.set_title(final_title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")

    # create custom legend
    handles = []
    labels = []
    for val in sorted(df[color_col].unique()):
        if color_col == "td_or_asd":
            label = "TD" if val == 0 else "ASD"
            col = palette[val]
        else:
            label = str(val)
            col = color_dict[val]
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=col, markersize=6,
                                  markeredgecolor='black', markeredgewidth=0.4))
        labels.append(label)

    ax.legend(
        handles,
        labels,
        title=color_col,
        loc="upper left",
        bbox_to_anchor=(1.05, 1.0),
        borderaxespad=0.,
        frameon=True,
        facecolor="white",
        framealpha=0.6,
    )

    plt.tight_layout()
    save_dual_format(save_path)


# ===================== KMEANS METRICS =====================

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


# ===================== GMM METRIC VISUALS =====================

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
    save_dual_format(save_path)


def plot_gmm_uncertainty(X_pca, responsibilities, save_path):
    """
    Visualizes GMM cluster uncertainty:
    uncertainty = 1 - max responsibility over components.
    """
    uncertainty = 1 - np.max(responsibilities, axis=1)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=uncertainty, cmap="magma", s=50, edgecolor='k', alpha=0.8
    )
    plt.colorbar(scatter, label="Cluster Uncertainty (1 - max responsibility)")
    plt.title("GMM Cluster Uncertainty", fontsize=13, fontweight='bold')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
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
    # For HDBSCAN, ignore noise (-1) when computing centroids
    if method_name in ("tsne_hdbscan", "hdbscan"):
        cluster_means = df[df["cluster"] != -1].groupby("cluster")[features].mean()
    else:
        cluster_means = df.groupby("cluster")[features].mean()

    plt.figure(figsize=(8, 5))
    sns.heatmap(cluster_means.T, annot=True, cmap="coolwarm")
    plt.title(f"Cluster Centroids by Feature ({method_name.upper()})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_dual_format(save_path)


def plot_feature_variance(df, features, save_path, method_name="kmeans"):
    """Feature variance across clusters."""
    if method_name in ("tsne_hdbscan", "hdbscan"):
        feature_variance = df[df["cluster"] != -1].groupby("cluster")[features].mean().var()
    else:
        feature_variance = df.groupby("cluster")[features].mean().var()

    plt.figure(figsize=(8, 4))
    feature_variance.sort_values(ascending=False).plot(kind='bar', color='teal')
    plt.title(f"Feature Variance Across Clusters ({method_name.upper()})", fontsize=13, fontweight='bold')
    plt.ylabel("Variance of Cluster Means")
    plt.tight_layout()
    save_dual_format(save_path)


# ===================== STREAMLIT PCA VISUALS =====================

def plot_pca_projection_streamlit(
    X_pca,
    df,
    color_col,
    save_path,
    title=None,
    method_name=None,
):
    """
    Interactive PCA projection for Streamlit (2D).
    - If color_col == 'td_or_asd': use light green/red and TD/ASD legend labels.
    - Otherwise behaves as before.
    """
    df_plot = df.copy()
    df_plot["PC1"] = X_pca[:, 0]
    df_plot["PC2"] = X_pca[:, 1]

    # ---------- color + title logic ----------
    color_map = None
    legend_title = color_col
    color_field = color_col  # what we actually give to px.scatter

    if color_col == "td_or_asd":
        # Map 0/1 → TD/ASD labels for the legend
        label_map = {0: "TD", 1: "ASD"}
        df_plot["td_asd_label"] = df_plot["td_or_asd"].map(label_map).fillna("Unknown")
        color_field = "td_asd_label"
        legend_title = "Group"
        color_map = {
            "TD": "#9FE2BF",   # soft green
            "ASD": "#FF9999",  # soft red
            "Unknown": "#AAAAAA",
        }

        # Title logic like the static plot
        uniq = set(df_plot["td_or_asd"].dropna().unique())
        if uniq == {0, 1}:
            plot_title = "2D Representation of TD/ASD Participants"
        elif uniq == {1}:
            plot_title = "2D Representation of ASD Participants"
        elif uniq == {0}:
            plot_title = "2D Representation of TD Participants"
        else:
            plot_title = "2D Representation of Participants"

    elif color_col == "cluster":
        color_map = None   # let Plotly pick tab20-like colors
        legend_title = "Cluster"
        if method_name == "pca_kmeans":
            plot_title = "2D Representation of KMeans Clusters"
        elif method_name == "pca_gmm":
            plot_title = "2D Representation of GMM Clusters"
        else:
            plot_title = "2D Representation of Clusters"
    else:
        color_map = None
        plot_title = title or f"2D Representation of {color_col}"

    # ---------- figure ----------
    fig = px.scatter(
        df_plot,
        x="PC1",
        y="PC2",
        color=color_field,
        color_discrete_map=color_map,
        hover_data={
            "td_or_asd": "td_or_asd" in df_plot.columns,
            "cluster": "cluster" in df_plot.columns,
            "PC1": ':.3f',
            "PC2": ':.3f',
        },
        title=plot_title,
        template="plotly_white",
    )
    fig.update_traces(
        marker=dict(
            size=7,
            line=dict(width=0.8, color="black")  # black border
        )
    )
    fig.update_layout(
        height=600,
        title_font=dict(size=16),
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        legend_title=legend_title,
    )

    html_path = Path(save_path).with_suffix(".html")
    fig.write_html(html_path, include_plotlyjs="inline")
    print(f"✅ Streamlit-ready interactive PCA saved: {html_path}")


def plot_pca_projection_streamlit_3d(
    X_pca,
    df,
    color_col,
    save_path,
    title=None,
    method_name=None,
):
    """
    Interactive 3D PCA projection for Streamlit.
    Uses PC1, PC2, PC3 (requires X_pca with at least 3 components).
    """
    if X_pca.shape[1] < 3:
        raise ValueError("X_pca must have at least 3 components for 3D projection.")

    df_plot = df.copy()
    df_plot["PC1"] = X_pca[:, 0]
    df_plot["PC2"] = X_pca[:, 1]
    df_plot["PC3"] = X_pca[:, 2]

    color_map = None
    legend_title = color_col
    color_field = color_col

    if color_col == "td_or_asd":
        label_map = {0: "TD", 1: "ASD"}
        df_plot["td_asd_label"] = df_plot["td_or_asd"].map(label_map).fillna("Unknown")
        color_field = "td_asd_label"
        legend_title = "Group"
        color_map = {
            "TD": "#9FE2BF",
            "ASD": "#FF9999",
            "Unknown": "#AAAAAA",
        }

        uniq = set(df_plot["td_or_asd"].dropna().unique())
        if uniq == {0, 1}:
            plot_title = "3D Representation of TD/ASD Participants"
        elif uniq == {1}:
            plot_title = "3D Representation of ASD Participants"
        elif uniq == {0}:
            plot_title = "3D Representation of TD Participants"
        else:
            plot_title = "3D Representation of Participants"

    elif color_col == "cluster":
        color_map = None
        legend_title = "Cluster"
        if method_name == "pca_kmeans":
            plot_title = "3D Representation of KMeans Clusters"
        elif method_name == "pca_gmm":
            plot_title = "3D Representation of GMM Clusters"
        else:
            plot_title = "3D Representation of Clusters"
    else:
        color_map = None
        plot_title = title or f"3D Representation of {color_col}"

    fig = px.scatter_3d(
        df_plot,
        x="PC1",
        y="PC2",
        z="PC3",
        color=color_field,
        color_discrete_map=color_map,
        hover_data={
            "td_or_asd": "td_or_asd" in df_plot.columns,
            "cluster": "cluster" in df_plot.columns,
            "PC1": ':.3f',
            "PC2": ':.3f',
            "PC3": ':.3f',
        },
        title=plot_title,
        template="plotly_white",
    )
    fig.update_traces(
        marker=dict(
            size=6,
            line=dict(width=0.7, color="black")
        )
    )
    fig.update_layout(
        height=650,
        title_font=dict(size=16),
        scene=dict(
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            zaxis_title="Principal Component 3",
        ),
        legend_title=legend_title,
    )

    html_path = Path(save_path).with_suffix(".html")
    fig.write_html(html_path, include_plotlyjs="inline")
    print(f"✅ Streamlit-ready 3D PCA saved: {html_path}")


# ===================== SHARED CLUSTER-STATS HELPER =====================

def _get_cluster_stats_text(df, numeric_cols, cluster_id):
    """
    Build an HTML snippet summarizing one cluster:
    - TD / ASD counts
    - min / max / mean for selected numeric columns
    """
    sub = df[df["cluster"] == cluster_id]

    if len(sub) == 0:
        return f"<b>Cluster {cluster_id}</b><br>No points in this cluster."

    td_count = int((sub.get("td_or_asd") == 0).sum()) if "td_or_asd" in sub.columns else 0
    asd_count = int((sub.get("td_or_asd") == 1).sum()) if "td_or_asd" in sub.columns else 0

    lines = []
    lines.append(f"<b>Cluster {cluster_id}</b>")
    lines.append(f"Total: {len(sub)}")
    if "td_or_asd" in sub.columns:
        lines.append(f"TD: {td_count}")
        lines.append(f"ASD: {asd_count}")
    lines.append("")

    for col in numeric_cols:
        if col not in sub.columns:
            continue
        col_min = sub[col].min()
        col_max = sub[col].max()
        col_mean = sub[col].mean()
        lines.append(f"<b>{col}</b>")
        lines.append(f"min = {col_min:.3f}")
        lines.append(f"max = {col_max:.3f}")
        lines.append(f"mean = {col_mean:.3f}")
        lines.append("")

    return "<br>".join(lines)


# ===================== PCA CLUSTER EXPLORER (KMEANS/GMM) =====================

def plot_pca_cluster_explorer(
    X_pca,
    df,
    numeric_cols,
    save_path,
    method_name="kmeans",
    dimensionality="auto",   # kept for backward-compat; we’ll ignore it
):
    """
    Interactive PCA cluster explorer:

    - Always writes a 2D explorer  →  <stem>_2d.html
    - If X_pca has >= 3 components, also writes a 3D explorer  →  <stem>_3d.html

    Example:
        save_path = ".../pca_cluster_explorer_kmeans.html"

        → writes:
            pca_cluster_explorer_kmeans_2d.html
            pca_cluster_explorer_kmeans_3d.html   (only if n_components >= 3)
    """
    save_path = Path(save_path)
    out_dir = save_path.parent
    stem = save_path.stem  # e.g. "pca_cluster_explorer_kmeans"

    path_2d = out_dir / f"{stem}_2d.html"
    path_3d = out_dir / f"{stem}_3d.html"

    has_pc3 = X_pca.shape[1] >= 3

    # ---- base dataframe with PCs ----
    df_plot = df.copy()
    df_plot["PC1"] = X_pca[:, 0]
    df_plot["PC2"] = X_pca[:, 1]
    if has_pc3:
        df_plot["PC3"] = X_pca[:, 2]

    clusters = sorted(df_plot["cluster"].unique())

    base_palette = sns.color_palette("tab20", 20)
    noise_color = "#999999"

    def _to_rgba(col):
        r, g, b = [int(255 * c) for c in col]
        return f"rgba({r},{g},{b},1)"

    # ---- stats text (shared by 2D and 3D) ----
    stats_text_map = {str(c): _get_cluster_stats_text(df, numeric_cols, c) for c in clusters}
    first_cluster = clusters[0]
    first_key = str(first_cluster)

    js_stats = json.dumps(stats_text_map)

    post_script_template = f"""
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        var statsByCluster = {js_stats};
        var plot = document.querySelectorAll('div.js-plotly-plot')[0];
        if (!plot) return;

        var panel = document.createElement('div');
        panel.id = 'cluster-stats-panel';

        panel.style.position = 'fixed';
        panel.style.top = '80px';
        panel.style.right = '30px';
        panel.style.width = '230px';
        panel.style.maxHeight = '380px';
        panel.style.overflowY = 'auto';
        panel.style.border = '1px solid #333';
        panel.style.background = 'white';
        panel.style.padding = '8px 10px';
        panel.style.boxShadow = '0 0 4px rgba(0,0,0,0.2)';
        panel.style.fontFamily = 'Arial, sans-serif';
        panel.style.fontSize = '12px';
        panel.style.boxSizing = 'border-box';
        panel.style.zIndex = 1000;

        var header = document.createElement('div');
        header.style.display = 'flex';
        header.style.justifyContent = 'space-between';
        header.style.alignItems = 'center';
        header.style.marginBottom = '6px';

        var title = document.createElement('span');
        title.textContent = 'Cluster stats';

        var btn = document.createElement('button');
        btn.textContent = '\\u2212';  // minus sign
        btn.style.marginLeft = '8px';
        btn.style.cursor = 'pointer';
        btn.style.border = '1px solid #888';
        btn.style.background = '#f7f7f7';
        btn.style.padding = '0 6px';

        header.appendChild(title);
        header.appendChild(btn);

        var content = document.createElement('div');
        content.id = 'cluster-stats-content';
        content.innerHTML = statsByCluster['{first_key}'] || 'No stats available';

        panel.appendChild(header);
        panel.appendChild(content);
        document.body.appendChild(panel);

        var collapsed = false;
        btn.addEventListener('click', function() {{
            collapsed = !collapsed;
            if (collapsed) {{
                content.style.display = 'none';
                btn.textContent = '+';
            }} else {{
                content.style.display = 'block';
                btn.textContent = '\\u2212';
            }}
        }});

        // ---- update stats when legend item is clicked ----
        plot.on('plotly_legendclick', function(event) {{
            var trace = event.data[event.curveNumber];
            var name = trace.name || "";
            var cid = name.replace("Cluster ", "");
            var text = statsByCluster[cid];
            if (text) {{
                content.innerHTML = text;
            }}
        }});
    }});
    </script>
    """

    def _build_fig(dim: str) -> go.Figure:
        fig = go.Figure()
        for i, c in enumerate(clusters):
            sub = df_plot[df_plot["cluster"] == c]
            color = noise_color if c == -1 else _to_rgba(base_palette[i % len(base_palette)])

            if dim == "3d" and has_pc3:
                trace = go.Scatter3d(
                    x=sub["PC1"],
                    y=sub["PC2"],
                    z=sub["PC3"],
                    mode="markers",
                    name=f"Cluster {c}",
                    marker=dict(
                        size=6,
                        color=color,
                        line=dict(width=0.7, color="black"),
                    ),
                    hovertemplate=(
                        "PC1: %{x:.3f}<br>"
                        "PC2: %{y:.3f}<br>"
                        "PC3: %{z:.3f}<br>"
                        f"Cluster: {c}<extra></extra>"
                    ),
                )
            else:
                trace = go.Scatter(
                    x=sub["PC1"],
                    y=sub["PC2"],
                    mode="markers",
                    name=f"Cluster {c}",
                    marker=dict(
                        size=7,
                        color=color,
                        line=dict(width=0.8, color="black"),
                    ),
                    hovertemplate=(
                        "PC1: %{x:.3f}<br>"
                        "PC2: %{y:.3f}<br>"
                        f"Cluster: {c}<extra></extra>"
                    ),
                )

            fig.add_trace(trace)

        # --- layout / titles ---
        if method_name == "gmm":
            algo_label = "GMM Clusters"
        else:
            algo_label = "KMeans Clusters"

        if dim == "3d" and has_pc3:
            fig.update_layout(
                title=f"3D Representation of {algo_label}",
                scene=dict(
                    xaxis_title="Principal Component 1",
                    yaxis_title="Principal Component 2",
                    zaxis_title="Principal Component 3",
                ),
                hovermode="closest",
                width=950,
                height=650,
                margin=dict(l=60, r=230, t=60, b=60),
                legend=dict(
                    title=dict(text="Clusters"),
                    x=0.66,
                    y=0.98,
                    xanchor="left",
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="lightgray",
                    borderwidth=1,
                    font=dict(size=11),
                ),
            )
        else:
            fig.update_layout(
                title=f"2D Representation of {algo_label}",
                xaxis=dict(title="Principal Component 1", domain=[0.0, 0.65]),
                yaxis=dict(title="Principal Component 2"),
                hovermode="closest",
                width=950,
                height=600,
                margin=dict(l=60, r=230, t=60, b=60),
                legend=dict(
                    title=dict(text="Clusters"),
                    x=0.66,
                    y=0.98,
                    xanchor="left",
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="lightgray",
                    borderwidth=1,
                    font=dict(size=11),
                ),
            )

        return fig

    def _write_html(fig: go.Figure, path: Path):
        html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")
        insert_pos = html_str.rfind("</body>")
        if insert_pos != -1:
            html_str = html_str[:insert_pos] + post_script_template + html_str[insert_pos:]
        else:
            html_str += post_script_template
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_str)
        print(f"✅ Interactive PCA cluster explorer saved → {path}")

    # ---- Always write 2D ----
    fig_2d = _build_fig("2d")
    _write_html(fig_2d, path_2d)

    # ---- If we have >= 3 components, also write 3D ----
    if has_pc3:
        fig_3d = _build_fig("3d")
        _write_html(fig_3d, path_3d)


# ===================== T-SNE / HDBSCAN VISUALS =====================

def plot_tsne_projection(X_tsne, df, color_col, save_path, title=None):
    """Scatterplot of t-SNE projection colored by cluster or target, with distinct palettes."""
    plt.figure(figsize=(8, 6))

    # === Choose palette and title dynamically ===
    if color_col == "td_or_asd":
        palette = {0: "#9FE2BF", 1: "#FF9999"}  # TD = green, ASD = red (soft)
        if df[color_col].nunique() == 1:
            plot_title = "2D Representation of ASD Participants"
        else:
            plot_title = "2D Representation of TD/ASD Participants"
    elif color_col == "cluster":
        n_clusters = df[color_col].nunique()
        palette = sns.color_palette("tab20", n_clusters)
        plot_title = "2D Representation of HDBSCAN Clusters"
    else:
        palette = "husl"
        plot_title = title or f"2D Representation of {color_col}"

    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=df[color_col],
        palette=palette, s=70, alpha=0.9,
        edgecolor="black", linewidth=0.4
    )

    plt.title(plot_title, fontsize=13, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(
        title=color_col,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=True, facecolor='white', framealpha=0.6
    )
    plt.tight_layout(pad=1.2)
    save_dual_format(save_path)


def plot_hdbscan_probability(X_tsne, clusterer, save_path):
    """
    Visualizes HDBSCAN cluster membership strength (probability per point).
    Darker = more confident cluster assignment.
    """
    if not hasattr(clusterer, "probabilities_"):
        print("⚠️ HDBSCAN clusterer has no probabilities_ attribute.")
        return

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        c=clusterer.probabilities_,
        cmap="viridis",
        s=50, edgecolor='k', alpha=0.8
    )
    plt.colorbar(scatter, label="Cluster Membership Probability")
    plt.title("HDBSCAN Cluster Membership Confidence", fontsize=13, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    save_dual_format(save_path)


def plot_tsne_projection_streamlit(X_tsne, df, color_col, save_path, title=None):
    """
    Interactive t-SNE projection for Streamlit.
    - If color_col == 'td_or_asd': use light green/red and TD/ASD legend labels.
    """
    df_plot = df.copy()
    df_plot["tSNE1"] = X_tsne[:, 0]
    df_plot["tSNE2"] = X_tsne[:, 1]

    color_map = None
    legend_title = color_col
    color_field = color_col

    if color_col == "td_or_asd":
        label_map = {0: "TD", 1: "ASD"}
        df_plot["td_asd_label"] = df_plot["td_or_asd"].map(label_map).fillna("Unknown")
        color_field = "td_asd_label"
        legend_title = "Group"
        color_map = {
            "TD": "#9FE2BF",   # soft green
            "ASD": "#FF9999",  # soft red
            "Unknown": "#AAAAAA",
        }

        uniq = set(df_plot["td_or_asd"].dropna().unique())
        if uniq == {0, 1}:
            plot_title = "2D Representation of TD/ASD Participants"
        elif uniq == {1}:
            plot_title = "2D Representation of ASD Participants"
        elif uniq == {0}:
            plot_title = "2D Representation of TD Participants"
        else:
            plot_title = "2D Representation of Participants"

    elif color_col == "cluster":
        color_map = None
        legend_title = "Cluster"
        plot_title = "2D Representation of HDBSCAN Clusters"
    else:
        color_map = None
        plot_title = title or f"2D Representation of {color_col}"

    fig = px.scatter(
        df_plot,
        x="tSNE1",
        y="tSNE2",
        color=color_field,
        color_discrete_map=color_map,
        hover_data={
            "cluster": "cluster" in df_plot.columns,
            "td_or_asd": "td_or_asd" in df_plot.columns,
            "tSNE1": ':.3f',
            "tSNE2": ':.3f',
        },
        title=plot_title,
        template="plotly_white",
    )
    fig.update_traces(
        marker=dict(
            size=7,
            line=dict(width=0.8, color="black")  # black border
        )
    )
    fig.update_layout(
        height=600,
        title_font=dict(size=16),
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        legend_title=legend_title,
    )

    html_path = Path(save_path).with_suffix(".html")
    fig.write_html(html_path, include_plotlyjs="inline")
    print(f"✅ Streamlit-ready interactive t-SNE saved: {html_path}")


def plot_tsne_cluster_explorer(X_tsne, df, numeric_cols, save_path):
    """
    Interactive t-SNE cluster explorer (HDBSCAN):
    - Scatter by t-SNE1 / t-SNE2
    - Fixed top-right stats panel (Streamlit-friendly)
    """
    save_path = Path(save_path)

    df_plot = df.copy()
    df_plot["tSNE1"] = X_tsne[:, 0]
    df_plot["tSNE2"] = X_tsne[:, 1]

    clusters = sorted(df_plot["cluster"].unique())

    # --- color palette: tab20 + grey for noise (-1) ---
    base_palette = sns.color_palette("tab20", 20)
    noise_color = "#999999"

    def _to_rgba(col):
        r, g, b = [int(255 * c) for c in col]
        return f"rgba({r},{g},{b},1)"

    fig = go.Figure()

    for i, c in enumerate(clusters):
        sub = df_plot[df_plot["cluster"] == c]

        if c == -1:
            color = noise_color
        else:
            color = _to_rgba(base_palette[i % len(base_palette)])

        fig.add_trace(
            go.Scatter(
                x=sub["tSNE1"],
                y=sub["tSNE2"],
                mode="markers",
                name=f"Cluster {c}",
                marker=dict(
                    size=7,
                    color=color,
                    line=dict(width=0.8, color="black"),  # black border around each dot
                ),
                hovertemplate=(
                    "tSNE1: %{x:.3f}<br>"
                    "tSNE2: %{y:.3f}<br>"
                    f"Cluster: {c}<extra></extra>"
                ),
            )
        )

    # --- stats text for each cluster ---
    stats_text_map = {str(c): _get_cluster_stats_text(df, numeric_cols, c) for c in clusters}
    first_cluster = clusters[0]
    first_key = str(first_cluster)

    fig.update_layout(
        title="2D Representation of HDBSCAN Clusters",
        xaxis=dict(title="t-SNE Dimension 1", domain=[0.0, 0.65]),
        yaxis=dict(title="t-SNE Dimension 2"),
        hovermode="closest",
        width=950,
        height=600,
        margin=dict(l=60, r=230, t=60, b=60),
        legend=dict(
            title=dict(text="Clusters"),
            x=0.66,   # inside middle column [0.60–0.75]
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(size=11),
        ),
    )

    # === JS: fixed top-right stats panel ===
    js_stats = json.dumps(stats_text_map)

    post_script = f"""
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        var statsByCluster = {js_stats};
        var plot = document.querySelectorAll('div.js-plotly-plot')[0];
        if (!plot) return;

        var panel = document.createElement('div');
        panel.id = 'cluster-stats-panel';

        panel.style.position = 'fixed';
        panel.style.top = '80px';
        panel.style.right = '30px';
        panel.style.width = '230px';
        panel.style.maxHeight = '380px';
        panel.style.overflowY = 'auto';
        panel.style.border = '1px solid #333';
        panel.style.background = 'white';
        panel.style.padding = '8px 10px';
        panel.style.boxShadow = '0 0 4px rgba(0,0,0,0.2)';
        panel.style.fontFamily = 'Arial, sans-serif';
        panel.style.fontSize = '12px';
        panel.style.boxSizing = 'border-box';
        panel.style.zIndex = 1000;

        var header = document.createElement('div');
        header.style.display = 'flex';
        header.style.justifyContent = 'space-between';
        header.style.alignItems = 'center';
        header.style.marginBottom = '6px';

        var title = document.createElement('span');
        title.textContent = 'Cluster stats';

        var btn = document.createElement('button');
        btn.textContent = '\\u2212';  // minus sign
        btn.style.marginLeft = '8px';
        btn.style.cursor = 'pointer';
        btn.style.border = '1px solid #888';
        btn.style.background = '#f7f7f7';
        btn.style.padding = '0 6px';

        header.appendChild(title);
        header.appendChild(btn);

        var content = document.createElement('div');
        content.id = 'cluster-stats-content';
        content.innerHTML = statsByCluster['{first_key}'] || 'No stats available';

        panel.appendChild(header);
        panel.appendChild(content);
        document.body.appendChild(panel);

        var collapsed = false;
        btn.addEventListener('click', function() {{
            collapsed = !collapsed;
            if (collapsed) {{
                content.style.display = 'none';
                btn.textContent = '+';
            }} else {{
                content.style.display = 'block';
                btn.textContent = '\\u2212';
            }}
        }});

        // ---- update stats when a legend item is clicked ----
        plot.on('plotly_legendclick', function(event) {{
            var trace = event.data[event.curveNumber];
            var name = trace.name || "";
            var cid = name.replace("Cluster ", "");
            var text = statsByCluster[cid];
            if (text) {{
                content.innerHTML = text;
            }}
        }});
    }});
    </script>
    """

    html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")
    insert_pos = html_str.rfind("</body>")
    if (insert_pos != -1):
        html_str = html_str[:insert_pos] + post_script + html_str[insert_pos:]
    else:
        html_str += post_script

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    # Write 2D explorer
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"✅ 2D t-SNE explorer saved → {save_path}")

    # If 3D available → also write 3D explorer
    if X_tsne.shape[1] >= 3:
        path_3d = save_path.with_name(save_path.stem + "_3d.html")
        plot_tsne_cluster_explorer_3d(X_tsne, df, numeric_cols, path_3d)


def plot_tsne_cluster_explorer_3d(X_tsne, df, numeric_cols, save_path):
    """
    3D t-SNE cluster explorer (HDBSCAN):
    - Uses tSNE1, tSNE2, tSNE3
    - Same stats panel as 2D version
    """
    save_path = Path(save_path)

    if X_tsne.shape[1] < 3:
        raise ValueError("3D t-SNE requested but only 2D data available.")

    df_plot = df.copy()
    df_plot["tSNE1"] = X_tsne[:, 0]
    df_plot["tSNE2"] = X_tsne[:, 1]
    df_plot["tSNE3"] = X_tsne[:, 2]

    clusters = sorted(df_plot["cluster"].unique())

    base_palette = sns.color_palette("tab20", 20)
    noise_color = "#999999"

    def _to_rgba(col):
        r, g, b = [int(255 * c) for c in col]
        return f"rgba({r},{g},{b},1)"

    fig = go.Figure()

    for i, c in enumerate(clusters):
        sub = df_plot[df_plot["cluster"] == c]

        color = noise_color if c == -1 else _to_rgba(base_palette[i % 20])

        fig.add_trace(go.Scatter3d(
            x=sub["tSNE1"],
            y=sub["tSNE2"],
            z=sub["tSNE3"],
            mode="markers",
            name=f"Cluster {c}",
            marker=dict(
                size=6,
                color=color,
                line=dict(width=0.7, color="black"),
            ),
            hovertemplate=(
                "tSNE1: %{x:.3f}<br>"
                "tSNE2: %{y:.3f}<br>"
                "tSNE3: %{z:.3f}<br>"
                f"Cluster: {c}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="3D Representation of HDBSCAN Clusters",
        scene=dict(
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            zaxis_title="t-SNE Dimension 3",
        ),
        hovermode="closest",
        width=950,
        height=650,
        margin=dict(l=60, r=230, t=60, b=60),
        legend=dict(
            title=dict(text="Clusters"),
            x=0.66, y=0.98,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(size=11),
        ),
    )

    # reuse same JS stats panel as 2D explorer
    stats_text_map = {str(c): _get_cluster_stats_text(df, numeric_cols, c) for c in clusters}
    first_key = str(clusters[0])
    js_stats = json.dumps(stats_text_map)

    post_script = f"""
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        var statsByCluster = {js_stats};
        var plot = document.querySelectorAll('div.js-plotly-plot')[0];
        if (!plot) return;

        var panel = document.createElement('div');
        panel.id = 'cluster-stats-panel';

        panel.style.position = 'fixed';
        panel.style.top = '80px';
        panel.style.right = '30px';
        panel.style.width = '230px';
        panel.style.maxHeight = '380px';
        panel.style.overflowY = 'auto';
        panel.style.border = '1px solid #333';
        panel.style.background = 'white';
        panel.style.padding = '8px 10px';
        panel.style.boxShadow = '0 0 4px rgba(0,0,0,0.2)';
        panel.style.fontFamily = 'Arial, sans-serif';
        panel.style.fontSize = '12px';
        panel.style.boxSizing = 'border-box';
        panel.style.zIndex = 1000;

        var header = document.createElement('div');
        header.style.display = 'flex';
        header.style.justifyContent = 'space-between';
        header.style.alignItems = 'center';
        header.style.marginBottom = '6px';

        var title = document.createElement('span');
        title.textContent = 'Cluster stats';

        var btn = document.createElement('button');
        btn.textContent = '\\u2212';
        btn.style.marginLeft = '8px';
        btn.style.cursor = 'pointer';
        btn.style.border = '1px solid #888';
        btn.style.background = '#f7f7f7';
        btn.style.padding = '0 6px';

        header.appendChild(title);
        header.appendChild(btn);

        var content = document.createElement('div');
        content.id = 'cluster-stats-content';
        content.innerHTML = statsByCluster['{first_key}'];

        panel.appendChild(header);
        panel.appendChild(content);
        document.body.appendChild(panel);

        var collapsed = false;
        btn.addEventListener('click', function() {{
            collapsed = !collapsed;
            content.style.display = collapsed ? 'none' : 'block';
            btn.textContent = collapsed ? '+' : '\\u2212';
        }});

        plot.on('plotly_legendclick', function(event) {{
            var trace = event.data[event.curveNumber];
            var name = trace.name || "";
            var cid = name.replace("Cluster ", "");
            if (statsByCluster[cid]) {{
                content.innerHTML = statsByCluster[cid];
            }}
        }});
    }});
    </script>
    """

    html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")
    insert_pos = html_str.rfind("</body>")
    if insert_pos != -1:
        html_str = html_str[:insert_pos] + post_script + html_str[insert_pos:]
    else:
        html_str += post_script

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"✅ Interactive 3D t-SNE cluster explorer saved → {save_path}")
