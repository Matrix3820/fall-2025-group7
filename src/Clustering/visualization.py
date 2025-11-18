import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json


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
    if method_name == "tsne_hdbscan":
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
    if method_name == "tsne_hdbscan":
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
    Interactive PCA projection for Streamlit.
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

def plot_pca_cluster_explorer(X_pca, df, numeric_cols, save_path, method_name="kmeans"):
    """
    Interactive PCA cluster explorer:
    - Scatter by PC1 / PC2
    - Each cluster is a separate trace
    - Fixed top-right stats panel (Streamlit-friendly)

    method_name controls the title wording (KMeans vs GMM).
    """
    save_path = Path(save_path)

    df_plot = df.copy()
    df_plot["PC1"] = X_pca[:, 0]
    df_plot["PC2"] = X_pca[:, 1]

    clusters = sorted(df_plot["cluster"].unique())

    # --- color palette: tab20 + grey for noise (-1), for consistency ---
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
                x=sub["PC1"],
                y=sub["PC2"],
                mode="markers",
                name=f"Cluster {c}",
                marker=dict(
                    size=7,
                    color=color,
                    line=dict(width=0.8, color="black"),  # border around each point
                ),
                hovertemplate=(
                    "PC1: %{x:.3f}<br>"
                    "PC2: %{y:.3f}<br>"
                    f"Cluster: {c}<extra></extra>"
                ),
            )
        )

    # --- stats text for each cluster ---
    stats_text_map = {str(c): _get_cluster_stats_text(df, numeric_cols, c) for c in clusters}
    first_cluster = clusters[0]
    first_key = str(first_cluster)

    if method_name == "pca_gmm":
        title = "2D Representation of GMM Clusters"
    else:
        title = "2D Representation of KMeans Clusters"

    # 60% domain for axes; legend in the middle 15%; panel is fixed overlay
    fig.update_layout(
        title=title,
        xaxis=dict(title="Principal Component 1", domain=[0.0, 0.65]),
        yaxis=dict(title="Principal Component 2"),
        hovermode="closest",
        # width=950,
        # height=600,
        autosize=True,
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

    # === JS: fixed top-right stats panel (works inside Streamlit iframe) ===
    js_stats = json.dumps(stats_text_map)

    post_script = f"""
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        var statsByCluster = {js_stats};
        var plot = document.querySelectorAll('div.js-plotly-plot')[0];
        if (!plot) return;

        // ---- create side panel container ----
        var panel = document.createElement('div');
        panel.id = 'cluster-stats-panel';

        // Fixed relative to iframe viewport
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

    html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")
    insert_pos = html_str.rfind("</body>")
    if (insert_pos != -1):
        html_str = html_str[:insert_pos] + post_script + html_str[insert_pos:]
    else:
        html_str += post_script

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"✅ Interactive PCA cluster explorer saved → {save_path}")


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

    # fig.update_layout(
    #     title="2D Representation of HDBSCAN Clusters",
    #     xaxis=dict(title="t-SNE Dimension 1", domain=[0.0, 0.65]),
    #     yaxis=dict(title="t-SNE Dimension 2"),
    #     hovermode="closest",
    #     width=950,
    #     height=600,
    #     margin=dict(l=60, r=230, t=60, b=60),
    #     legend=dict(
    #         title=dict(text="Clusters"),
    #         x=0.66,   # inside middle column [0.60–0.75]
    #         y=0.98,
    #         xanchor="left",
    #         yanchor="top",
    #         bgcolor="rgba(255,255,255,0.85)",
    #         bordercolor="lightgray",
    #         borderwidth=1,
    #         font=dict(size=11),
    #     ),
    # )

    fig.update_layout(
        title="2D Representation of HDBSCAN Clusters",
        xaxis=dict(title="t-SNE Dimension 1"),
        yaxis=dict(title="t-SNE Dimension 2"),
        hovermode="closest",
        # width=950,
        # height=600,
        autosize=True,
        margin=dict(l=60, r=220, t=60, b=60),
        legend=dict(
            title=dict(text="Clusters"),
            x=1.02,
            y=1,
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

    print(f"✅ Interactive t-SNE cluster explorer saved → {save_path}")
