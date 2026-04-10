# ============================================================
# config/settings.py
# All constants, color palette, plotly theme, pipeline steps
# ============================================================

# ── Accent Colors ──
CYAN    = "#22d3ee"
VIOLET  = "#a78bfa"
EMERALD = "#34d399"
AMBER   = "#fbbf24"
ROSE    = "#fb7185"
SKY     = "#38bdf8"
PURPLE  = "#c084fc"
TEAL    = "#6ee7b7"

COLORS = [CYAN, VIOLET, EMERALD, AMBER, ROSE, SKY, PURPLE, TEAL]

# ── Plotly dark theme defaults ──
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color="#6b7090", size=11),
    colorway=COLORS,
)

# ── Pipeline step definitions ──
PIPELINE_STEPS = [
    ("📥", "Load Data"),
    ("🔍", "EDA"),
    ("🧹", "Clean"),
    ("⚙️",  "Features"),
    ("🤖", "Cluster"),
    ("📈", "Results"),
    ("🎓", "Learn"),
]

# ── Algorithm metadata ──
ALGO_INFO = {
    "KMeans": {
        "icon":  "🎯",
        "level": "Beginner",
        "desc":  "Partitions data into K spherical clusters by minimising within-cluster variance. Fast and scalable.",
        "best":  "Clean data, roughly equal cluster sizes, you know roughly how many clusters to expect.",
        "worst": "Outliers, non-spherical clusters, varying density.",
    },
    "DBSCAN": {
        "icon":  "🌌",
        "level": "Intermediate",
        "desc":  "Density-based — clusters are regions of high density separated by low density. Finds outliers naturally.",
        "best":  "Arbitrary-shaped clusters, data with noise/outliers, unknown number of clusters.",
        "worst": "Varying density clusters, high-dimensional data.",
    },
    "Agglomerative": {
        "icon":  "🌳",
        "level": "Intermediate",
        "desc":  "Hierarchical bottom-up: starts with each point as its own cluster, merges closest pairs.",
        "best":  "When you want a hierarchy/dendrogram, small-to-medium datasets.",
        "worst": "Large datasets (slow), need to specify K.",
    },
    "Spectral": {
        "icon":  "🌊",
        "level": "Advanced",
        "desc":  "Uses graph/eigenvalue decomposition to find clusters. Excellent for non-convex shapes.",
        "best":  "Complex non-spherical clusters, image segmentation.",
        "worst": "Very large datasets (memory intensive).",
    },
    "Birch": {
        "icon":  "🌿",
        "level": "Intermediate",
        "desc":  "Builds a tree structure for fast incremental clustering. Memory efficient.",
        "best":  "Very large datasets, streaming data.",
        "worst": "Non-spherical clusters, outlier-heavy data.",
    },
    "MeanShift": {
        "icon":  "🎱",
        "level": "Intermediate",
        "desc":  "Finds cluster centres by shifting towards high-density regions. Auto-finds K.",
        "best":  "Unknown K, smooth density distributions.",
        "worst": "Large datasets (slow), choosing bandwidth is tricky.",
    },
}

# ── Session state defaults ──
SESSION_DEFAULTS = {
    "step":               0,
    "df_raw":             None,
    "df_clean":           None,
    "df_engineered":      None,
    "X_processed":        None,
    "labels":             None,
    "model":              None,
    "model_name":         "",
    "metrics":            {},
    "preprocessing_done": False,
    "eda_done":           False,
    "engineering_done":   False,
    "clustering_done":    False,
    "outlier_method":     "none",
    "scaler":             "StandardScaler",
    "imputer":            "Mean",
    "removed_cols":       [],
    "automl_results":     [],
    "reduction":          "PCA",
    "xp":                 "🟢 Beginner",
    # Recovery / confirmation flags
    "confirm_reset":      False,
    "created_features":   [],
}

# ── Step-specific session keys (for partial reset) ──
# Restarting from step N clears this step's keys AND all later steps' keys.
STEP_RESET_KEYS: dict = {
    0: ["df_raw", "df_clean", "df_engineered",
        "X_processed", "labels", "model", "model_name", "metrics",
        "preprocessing_done", "eda_done", "engineering_done",
        "clustering_done", "automl_results", "created_features"],
    1: ["eda_done"],
    2: ["df_clean", "df_engineered",
        "X_processed", "labels", "model", "model_name", "metrics",
        "preprocessing_done", "engineering_done",
        "clustering_done", "automl_results", "created_features"],
    3: ["df_engineered",
        "X_processed", "labels", "model", "model_name", "metrics",
        "engineering_done", "clustering_done", "automl_results",
        "created_features"],
    4: ["X_processed", "labels", "model", "model_name",
        "metrics", "clustering_done", "automl_results"],
    5: [],
    6: [],
}
