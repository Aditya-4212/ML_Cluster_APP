# ============================================================
# utils/charts.py
# Every Plotly figure builder used across the pipeline.
# Edit chart styles, sizes, colors here — nowhere else needed.
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import PLOTLY_THEME, COLORS


# ── Scatter (2-D projection) ────────────────────────────────

def scatter_clusters(X2d: np.ndarray, labels, title: str = "Cluster Map", hover_series=None):
    df = pd.DataFrame({"x": X2d[:, 0], "y": X2d[:, 1],
                        "Cluster": [str(l) for l in labels]})
    if hover_series is not None:
        df["hover"] = hover_series.values
        fig = px.scatter(df, x="x", y="y", color="Cluster",
                         hover_data={"hover": True},
                         color_discrete_sequence=COLORS, title=title)
    else:
        fig = px.scatter(df, x="x", y="y", color="Cluster",
                         color_discrete_sequence=COLORS, title=title)
    fig.update_traces(marker=dict(size=5, opacity=0.75, line=dict(width=0)))
    fig.update_layout(
        **PLOTLY_THEME, height=480,
        title=dict(font=dict(family="IBM Plex Mono", size=12)),
        xaxis=dict(gridcolor="#1e2035", zeroline=False, title=""),
        yaxis=dict(gridcolor="#1e2035", zeroline=False, title=""),
        legend=dict(bgcolor="rgba(17,18,37,0.8)", bordercolor="#1e2035", borderwidth=1),
    )
    return fig


# ── Bar distribution ────────────────────────────────────────

def cluster_bar(labels, title: str = "Cluster Sizes"):
    cnt = pd.Series(labels).value_counts().sort_index().reset_index()
    cnt.columns = ["Cluster", "Count"]
    cnt["Cluster"] = cnt["Cluster"].astype(str)
    fig = px.bar(cnt, x="Cluster", y="Count", color="Cluster",
                 color_discrete_sequence=COLORS, title=title)
    fig.update_layout(**PLOTLY_THEME, height=320, showlegend=False,
                      xaxis=dict(showgrid=False),
                      yaxis=dict(gridcolor="#1e2035"), bargap=0.3)
    fig.update_traces(marker_line_width=0)
    return fig


# ── Pie distribution ────────────────────────────────────────

def cluster_pie(labels):
    cnt = pd.Series(labels).value_counts().sort_index().reset_index()
    cnt.columns = ["Cluster", "Count"]
    cnt["Cluster"] = cnt["Cluster"].astype(str)
    fig = px.pie(cnt, values="Count", names="Cluster",
                 color_discrete_sequence=COLORS, title="Cluster Share", hole=0.5)
    fig.update_layout(**PLOTLY_THEME, height=320)
    return fig


# ── Histogram ───────────────────────────────────────────────

def feature_histogram(df: pd.DataFrame, col: str, nbins: int = 40):
    fig = px.histogram(df, x=col, nbins=nbins,
                       color_discrete_sequence=["#22d3ee"],
                       title=f"Distribution · {col}")
    fig.update_layout(**PLOTLY_THEME, height=300,
                      xaxis=dict(showgrid=False),
                      yaxis=dict(gridcolor="#1e2035"))
    return fig


# ── Box plot ────────────────────────────────────────────────

def feature_boxplot(df: pd.DataFrame, col: str):
    fig = px.box(df, y=col, color_discrete_sequence=["#a78bfa"],
                 title=f"Box Plot · {col}")
    fig.update_layout(**PLOTLY_THEME, height=300)
    return fig


# ── Scatter (EDA x vs y) ────────────────────────────────────

def eda_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col=None):
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                     color_discrete_sequence=COLORS, opacity=0.6,
                     title=f"{x_col} vs {y_col}")
    fig.update_layout(**PLOTLY_THEME, height=380)
    return fig


# ── Correlation heatmap ─────────────────────────────────────

def correlation_heatmap(df: pd.DataFrame):
    corr = df.corr()
    fig = px.imshow(
        corr,
        color_continuous_scale=[[0, "#fb7185"], [0.5, "#111225"], [1, "#22d3ee"]],
        aspect="auto", title="Pearson Correlation",
    )
    fig.update_layout(**PLOTLY_THEME,
                      height=max(350, 22 * len(corr.columns)),
                      title=dict(font=dict(family="IBM Plex Mono", size=12)))
    return fig


# ── Outlier scatter ─────────────────────────────────────────

def outlier_scatter(series: pd.Series, z_scores: np.ndarray, col_name: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(series))), y=series.values, mode="markers",
        marker=dict(color=["#fb7185" if v > 3 else "#22d3ee" for v in z_scores],
                    size=4, opacity=0.7),
    ))
    fig.update_layout(
        **PLOTLY_THEME, height=300,
        title=dict(text=f"Outlier View · {col_name} (red = |Z| > 3)",
                   font=dict(family="IBM Plex Mono", size=12)),
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1e2035"),
    )
    return fig


# ── Categorical bar ─────────────────────────────────────────

def cat_bar(series: pd.Series, col_name: str):
    vc = series.value_counts().head(20)
    fig = px.bar(x=vc.index, y=vc.values,
                 color_discrete_sequence=["#a78bfa"],
                 title=f"Value Counts · {col_name}")
    fig.update_layout(**PLOTLY_THEME, height=320,
                      xaxis=dict(showgrid=False, title=""),
                      yaxis=dict(gridcolor="#1e2035", title="Count"))
    return fig


# ── PCA explained variance ──────────────────────────────────

def pca_variance_chart(individual: np.ndarray, cumulative: np.ndarray):
    labels = [f"PC{i+1}" for i in range(len(individual))]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=individual,
                         marker_color="#a78bfa", name="Individual"))
    fig.add_trace(go.Scatter(x=labels, y=cumulative, mode="lines+markers",
                             line=dict(color="#22d3ee", width=2),
                             marker=dict(size=6), name="Cumulative"))
    fig.add_hline(y=80, line_dash="dash", line_color="#fbbf24",
                  annotation_text="80% threshold")
    fig.update_layout(**PLOTLY_THEME, height=320,
                      title=dict(text="PCA — Explained Variance",
                                 font=dict(family="IBM Plex Mono", size=12)),
                      xaxis=dict(showgrid=False),
                      yaxis=dict(gridcolor="#1e2035", title="%"))
    return fig


# ── Elbow curve (cached to avoid re-fitting on slider moves) ─

@st.cache_data(show_spinner=False)
def elbow_chart(X: np.ndarray, max_k: int = 10):
    inertias = []
    ks = list(range(2, max_k + 1))
    for k in ks:
        m = KMeans(n_clusters=k, random_state=42, n_init="auto")
        m.fit(X)
        inertias.append(m.inertia_)
    fig = go.Figure(go.Scatter(
        x=ks, y=inertias, mode="lines+markers",
        line=dict(color="#22d3ee", width=2),
        marker=dict(size=7, color="#22d3ee"),
    ))
    fig.update_layout(**PLOTLY_THEME, height=300,
                      title=dict(text="Elbow Curve",
                                 font=dict(family="IBM Plex Mono", size=12)),
                      xaxis=dict(gridcolor="#1e2035", dtick=1),
                      yaxis=dict(gridcolor="#1e2035", title="Inertia"))
    return fig


# ── Silhouette sweep (cached) ───────────────────────────────

@st.cache_data(show_spinner=False)
def silhouette_sweep(X: np.ndarray, max_k: int = 10):
    scores = []
    ks = list(range(2, max_k + 1))
    for k in ks:
        try:
            lbl = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X)
            scores.append(silhouette_score(X, lbl) if len(set(lbl)) >= 2 else 0)
        except Exception:
            scores.append(0)
    fig = go.Figure(go.Scatter(
        x=ks, y=scores, mode="lines+markers",
        line=dict(color="#a78bfa", width=2), marker=dict(size=7),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.08)",
    ))
    fig.update_layout(**PLOTLY_THEME, height=300,
                      title=dict(text="Silhouette by K",
                                 font=dict(family="IBM Plex Mono", size=12)),
                      xaxis=dict(gridcolor="#1e2035", dtick=1),
                      yaxis=dict(gridcolor="#1e2035", title="Score"))
    return fig


# ── Feature importance (range across clusters) ──────────────

def feature_importance_chart(df_clustered: pd.DataFrame):
    num_cols = [c for c in df_clustered.select_dtypes(include=np.number).columns
                if c != "Cluster"]
    if not num_cols:
        return None
    summary = df_clustered.groupby("Cluster")[num_cols].mean()
    diff = (summary.max() - summary.min()).sort_values(ascending=True).tail(10)
    fig = go.Figure(go.Bar(
        x=diff.values, y=diff.index, orientation="h",
        marker=dict(color=diff.values,
                    colorscale=[[0, "#1e2035"], [0.5, "#a78bfa"], [1, "#22d3ee"]]),
    ))
    fig.update_layout(**PLOTLY_THEME, height=350,
                      title=dict(text="Top Differentiating Features",
                                 font=dict(family="IBM Plex Mono", size=12)),
                      xaxis=dict(gridcolor="#1e2035", title="Mean Range"),
                      yaxis=dict(showgrid=False))
    return fig


# ── Radar / spider profile ──────────────────────────────────

def radar_profile(df_clustered: pd.DataFrame):
    num_cols = [c for c in df_clustered.select_dtypes(include=np.number).columns
                if c != "Cluster"]
    if not num_cols:
        return None
    summary = df_clustered.groupby("Cluster")[num_cols].mean().round(3)
    fig = go.Figure()
    for i, (idx, row) in enumerate(summary.iterrows()):
        fig.add_trace(go.Scatterpolar(
            r=row.values, theta=row.index.tolist(),
            fill="toself", name=f"Cluster {idx}",
            line=dict(color=COLORS[i % len(COLORS)]), opacity=0.7,
        ))
    fig.update_layout(
        **PLOTLY_THEME, height=420,
        polar=dict(bgcolor="#0d0e1a",
                   radialaxis=dict(visible=True, gridcolor="#1e2035", showticklabels=False),
                   angularaxis=dict(gridcolor="#1e2035")),
        title=dict(text="Radar: Cluster Profiles", font=dict(family="IBM Plex Mono", size=12)),
        legend=dict(bgcolor="rgba(17,18,37,0.8)", bordercolor="#1e2035", borderwidth=1),
    )
    return fig


# ── Heatmap (normalised means) ──────────────────────────────

def cluster_heatmap(df_clustered: pd.DataFrame):
    num_cols = [c for c in df_clustered.select_dtypes(include=np.number).columns
                if c != "Cluster"]
    if len(num_cols) < 2:
        return None
    summary = df_clustered.groupby("Cluster")[num_cols].mean()
    norm = (summary - summary.min()) / (summary.max() - summary.min() + 1e-9)
    fig = px.imshow(
        norm.T,
        color_continuous_scale=[[0, "#07080f"], [0.4, "#1e2035"],
                                 [0.7, "#a78bfa"], [1, "#22d3ee"]],
        aspect="auto", title="Feature Heatmap (Normalised per feature)",
    )
    fig.update_layout(**PLOTLY_THEME,
                      height=max(300, 28 * len(num_cols)),
                      title=dict(font=dict(family="IBM Plex Mono", size=12)))
    return fig


# ── Scatter matrix ──────────────────────────────────────────

def scatter_matrix(df_clustered: pd.DataFrame, top_n: int = 4):
    num_cols = [c for c in df_clustered.select_dtypes(include=np.number).columns
                if c != "Cluster"]
    if len(num_cols) < 2:
        return None
    summary = df_clustered.groupby("Cluster")[num_cols].mean()
    diff = (summary.max() - summary.min()).sort_values(ascending=False)
    top = diff.head(top_n).index.tolist()
    fig = px.scatter_matrix(df_clustered, dimensions=top,
                            color=df_clustered["Cluster"].astype(str),
                            color_discrete_sequence=COLORS, title="Pair Scatter")
    fig.update_traces(marker=dict(size=3, opacity=0.5))
    fig.update_layout(**PLOTLY_THEME, height=520,
                      title=dict(font=dict(family="IBM Plex Mono", size=12)))
    return fig


# ── Dendrogram (matplotlib) ─────────────────────────────────

def dendrogram_chart(df: pd.DataFrame, max_rows: int = 500):
    """
    Returns a matplotlib Figure. Caller must call plt.close(fig) after st.pyplot(fig).
    """
    num_cols = [c for c in df.select_dtypes(include=np.number).columns
                if c != "Cluster"]
    if not num_cols:
        return None
    sample = df[num_cols].dropna().sample(min(max_rows, len(df)), random_state=42)
    Z = sch.linkage(StandardScaler().fit_transform(sample), method="ward")
    fig_d, ax = plt.subplots(figsize=(12, 4))
    fig_d.patch.set_facecolor("#0d0e1a")
    ax.set_facecolor("#0d0e1a")
    sch.dendrogram(Z, ax=ax, leaf_rotation=90,
                   color_threshold=0.7 * max(Z[:, 2]),
                   above_threshold_color="#22d3ee",
                   link_color_func=lambda k: "#a78bfa")
    ax.tick_params(colors="#6b7090", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#1e2035")
    ax.set_title("Hierarchical Dendrogram (Ward Linkage)",
                 color="#6b7090", fontsize=10, pad=8)
    plt.tight_layout()
    return fig_d


# ── AutoML comparison bar ───────────────────────────────────

def automl_comparison_chart(results: list) -> go.Figure:
    """results: list of dicts from _run_automl (private keys prefixed with '_')."""
    df = pd.DataFrame(results[:15]).drop(columns=["_model", "_labels"], errors="ignore")

    # Safe column access — avoids DataFrame.get() misuse
    config_col = df["Config"].astype(str) if "Config" in df.columns else pd.Series([""] * len(df))
    y_labels = df["Algorithm"] + " " + config_col

    fig = px.bar(
        df, x="Silhouette ↑", y=y_labels,
        orientation="h", color="Silhouette ↑",
        color_continuous_scale=[[0, "#1e2035"], [0.5, "#a78bfa"], [1, "#22d3ee"]],
        title="AutoML — Silhouette Score by Configuration",
    )
    fig.update_layout(
        **PLOTLY_THEME, height=max(300, 25 * len(df)),
        showlegend=False,
        title=dict(font=dict(family="IBM Plex Mono", size=12)),
        xaxis=dict(gridcolor="#1e2035"),
        yaxis=dict(showgrid=False),
    )
    return fig
