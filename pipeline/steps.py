# ============================================================
# pipeline/steps.py  — v2  (improved)
# ============================================================

from __future__ import annotations

import html as _html
import io
import time
import urllib.request

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import zscore
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering,
    SpectralClustering, Birch, MeanShift,
)

from config.settings import ALGO_INFO, COLORS
from utils.data import (
    load_csv, apply_imputation, apply_outlier_removal,
    preprocess_X, reduce_2d,
    get_numeric_cols, get_cat_cols,
    get_low_variance_features, get_pca_explained,
    get_high_corr_pairs, get_auto_remove_cols,
)
from utils.metrics import compute_all_metrics, safe_silhouette, safe_davies_bouldin
from utils.charts import (
    scatter_clusters, cluster_bar, cluster_pie,
    feature_histogram, feature_boxplot, eda_scatter,
    correlation_heatmap, outlier_scatter, cat_bar,
    pca_variance_chart, elbow_chart, silhouette_sweep,
    feature_importance_chart, radar_profile,
    cluster_heatmap, scatter_matrix, dendrogram_chart,
    automl_comparison_chart,
)
from components.ui import section, explain, metric_strip, progress_tracker, recovery_panel, nav_buttons


# ── Shared helpers ───────────────────────────────────────────

def _need(key: str, msg: str = "Complete the previous step first.") -> None:
    if st.session_state.get(key) is None:
        st.warning(msg)
        st.stop()


def _safe_html(value) -> str:
    return _html.escape(str(value))


# ── Report generator ─────────────────────────────────────────

def generate_report(df_r: pd.DataFrame, metrics: dict, model_name: str) -> str:
    """Build a Markdown string summarising the complete analysis."""
    num_cols = [c for c in df_r.select_dtypes(include=np.number).columns if c != "Cluster"]

    lines = [
        "# ML Clustering Analysis Report",
        "",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Tool:** ML Clustering Studio",
        "",
        "---",
        "",
        "## 1. Dataset Overview",
        "",
        f"| Property | Value |",
        f"|---|---|",
        f"| Rows | {len(df_r)} |",
        f"| Features used | {len(num_cols)} numeric |",
        f"| Feature names | {', '.join(num_cols[:10])}{'…' if len(num_cols) > 10 else ''} |",
        "",
        "## 2. Preprocessing Choices",
        "",
        f"| Setting | Value |",
        f"|---|---|",
        f"| Imputer | {st.session_state.get('imputer', 'Mean')} |",
        f"| Scaler | {st.session_state.get('scaler', 'StandardScaler')} |",
        f"| Outlier handling | {st.session_state.get('outlier_method', 'None')} |",
        f"| Visualisation | {st.session_state.get('reduction', 'PCA')} |",
        "",
        "## 3. Model",
        "",
        f"| Property | Value |",
        f"|---|---|",
        f"| Algorithm | {model_name} |",
        f"| Clusters found | {metrics.get('Clusters', '?')} |",
    ]

    noise = metrics.get("Noise pts", 0)
    if noise > 0:
        lines.append(f"| Noise points (DBSCAN) | {noise} |")

    lines += [
        "",
        "## 4. Evaluation Metrics",
        "",
        "| Metric | Value | Interpretation |",
        "|---|---|---|",
    ]

    interp = {
        "Silhouette ↑":      "Higher is better. >0.5 = strong clusters, >0.7 = excellent.",
        "Davies-Bouldin ↓":  "Lower is better. Measures average cluster similarity.",
        "Calinski-Harabasz ↑": "Higher is better. Ratio of between-to-within cluster dispersion.",
    }
    for k, v in metrics.items():
        if k not in ("Clusters", "Noise pts"):
            lines.append(f"| {k} | {v} | {interp.get(k, '—')} |")

    lines += [
        "",
        "## 5. Cluster Sizes",
        "",
    ]
    sizes = df_r["Cluster"].value_counts().sort_index()
    lines.append("| Cluster | Count | % of Total |")
    lines.append("|---|---|---|")
    for cluster_id, count in sizes.items():
        pct = round(count / len(df_r) * 100, 1)
        lines.append(f"| Cluster {cluster_id} | {count} | {pct}% |")

    if num_cols:
        lines += [
            "",
            "## 6. Cluster Profiles (Mean Feature Values)",
            "",
        ]
        profile = df_r.groupby("Cluster")[num_cols].mean().round(3)
        lines.append(profile.to_markdown())

        lines += [
            "",
            "## 7. Cluster Descriptions",
            "",
        ]
        overall_means = df_r[num_cols].mean()
        for cluster_id in sorted(df_r["Cluster"].unique()):
            if cluster_id == -1:
                lines.append(f"**Cluster -1 (Noise):** Outlier points not assigned to any cluster by DBSCAN.")
                continue
            cluster_data = df_r[df_r["Cluster"] == cluster_id][num_cols].mean()
            n = (df_r["Cluster"] == cluster_id).sum()
            pct = round(n / len(df_r) * 100, 1)
            above = [f for f in num_cols if cluster_data[f] > overall_means[f] * 1.1]
            below = [f for f in num_cols if cluster_data[f] < overall_means[f] * 0.9]
            desc_parts = []
            if above:
                desc_parts.append(f"above-average {', '.join(above[:3])}")
            if below:
                desc_parts.append(f"below-average {', '.join(below[:3])}")
            desc = "; ".join(desc_parts) if desc_parts else "near-average across all features"
            lines.append(f"**Cluster {cluster_id}** ({n} records, {pct}%): Characterised by {desc}.")
            lines.append("")

    lines += [
        "",
        "## 8. Next Steps",
        "",
        "- **Label your clusters**: Review the profiles above and assign business-meaningful names "
        "(e.g. 'High-Value Customers', 'Occasional Buyers').",
        "- **Validate with domain experts**: The best cluster count balances statistical metrics "
        "with business interpretability.",
        "- **Test stability**: Run the same algorithm on a different random seed or sample "
        "to confirm clusters are stable.",
        "- **Consider DBSCAN**: If you used KMeans, try DBSCAN to see if noise points reveal outliers.",
        "- **Deploy**: Save the model using the `.joblib` export and wrap it in a sklearn Pipeline "
        "with the scaler for production use.",
        "",
        "---",
        "*Generated by ML Clustering Studio*",
    ]

    return "\n".join(lines)


# ============================================================
# STEP 0 — LOAD DATA
# ============================================================

def step_load(uploaded) -> None:
    section("Step 1 · Load Your Dataset")
    explain(
        "📥 What is this step?",
        "We start by <strong>loading your CSV file</strong>. "
        "Every row is a record (customer, product, event) and every column is a feature. "
        "The goal of clustering is to <strong>find hidden groups</strong> automatically — "
        "no labels or prior knowledge required. The better your data, the better your clusters.",
        kind="learn",
    )

    # ── No file uploaded ──
    if not uploaded and st.session_state.df_raw is None:
        # ── Inline upload (main area) ──
        section("Upload Your Dataset")
        inline_upload = st.file_uploader(
            "Drop a CSV file here",
            type=["csv"],
            key="inline_uploader",
            help="Max 50 MB · Supports numeric and categorical columns · Auto-detects types",
        )
        if inline_upload:
            try:
                with st.spinner("Loading and detecting column types…"):
                    df = load_csv(inline_upload)
                st.session_state.df_raw = df
                st.rerun()
            except ValueError as e:
                st.error(str(e))
                st.stop()

        st.markdown(
            """
            <div style="text-align:center;padding:1.5rem 2rem;background:#111225;
            border:2px dashed #1e2035;border-radius:14px;margin:1rem 0;">
              <div style="font-size:2.5rem;margin-bottom:0.8rem">🧬</div>
              <div style="font-family:IBM Plex Mono;font-size:0.85rem;color:#8b90b0;margin-bottom:0.3rem;">
              Or load a sample dataset below
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        section("Or load a sample dataset")
        samples = {
            "🛒 Mall Customers": "https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Mall%20Customers.csv",
            "🌸 Iris Flowers":   "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        }
        c1, c2 = st.columns(2)
        for idx, (name, url) in enumerate(samples.items()):
            col = c1 if idx == 0 else c2
            with col:
                if st.button(f"Load {name}", use_container_width=True):
                    with st.spinner(f"Fetching {name} — detecting column types…"):
                        try:
                            with urllib.request.urlopen(url, timeout=10) as r:
                                df = pd.read_csv(io.StringIO(r.read().decode("utf-8")))
                            st.session_state.df_raw = df
                            st.session_state.step = 1
                            st.rerun()
                        except Exception as e:
                            st.error(f"Could not load sample: {_safe_html(e)}")
        st.stop()

    # ── File uploaded (sidebar) ──
    if uploaded:
        try:
            with st.spinner("Loading and detecting column types…"):
                df = load_csv(uploaded)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        st.session_state.df_raw = df

    df = st.session_state.df_raw

    # ── Preview ──
    section("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True, height=300)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",    df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Numeric", len(get_numeric_cols(df)))
    c4.metric("Missing", int(df.isnull().sum().sum()))

    # ── Column info table ──
    section("Column Overview")
    info_rows = []
    for col in df.columns:
        info_rows.append({
            "Column":        col,
            "Type":          str(df[col].dtype),
            "Missing":       int(df[col].isnull().sum()),
            "Unique Values": int(df[col].nunique()),
            "Sample":        str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "—",
        })
    st.dataframe(pd.DataFrame(info_rows), use_container_width=True, hide_index=True)

    explain(
        "💡 Column Types Matter",
        "<strong>Numeric columns</strong> (int, float) are used directly for distance calculations. "
        "<strong>Object/string columns</strong> are categorical and will be one-hot encoded automatically. "
        "Columns like 'CustomerID' or 'Name' are identifiers — they should be removed in the Clean step, "
        "as they add noise without signal.",
        kind="learn",
    )

    nav_buttons("Proceed to EDA →", proceed_step=1, proceed_key="load_proceed")
    recovery_panel(0)


# ============================================================
# STEP 1 — EDA
# ============================================================

def step_eda() -> None:
    section("Step 2 · Exploratory Data Analysis")
    _need("df_raw", "Please load a dataset first.")
    df = st.session_state.df_raw

    explain(
        "🔍 What is EDA?",
        "<strong>Exploratory Data Analysis (EDA)</strong> is the practice of systematically "
        "examining your data before modelling. Professional data scientists spend 60–80% of "
        "their project time here — not training models. EDA answers: What shape is my data? "
        "Are there outliers? Which features are correlated? What scale are the values? "
        "Understanding these things guides every decision in the steps that follow.",
        kind="learn",
    )

    num_cols = get_numeric_cols(df)
    cat_cols = get_cat_cols(df)

    # Tab order reordered: Statistics first, then Distributions (natural exploration path)
    tab_stats, tab_dist, tab_corr, tab_out, tab_cat = st.tabs([
        "📋 Statistics", "📊 Distributions", "🔗 Correlations", "🎯 Outliers", "🏷️ Categorical"
    ])

    # ── Statistics (first — understand scale before visualising) ──
    with tab_stats:
        section("Descriptive Statistics")
        if num_cols:
            st.dataframe(df[num_cols].describe().round(3), use_container_width=True)
        explain(
            "📋 What do these numbers mean?",
            "<strong>count</strong> = non-missing rows · "
            "<strong>mean</strong> = average · "
            "<strong>std</strong> = standard deviation (how spread out values are) · "
            "<strong>min/max</strong> = extremes · "
            "<strong>25%/50%/75%</strong> = quartiles. "
            "A large std relative to mean indicates the feature may need scaling before clustering. "
            "Features with very different scales (Age 0–100 vs Salary 0–100,000) will skew "
            "distance-based algorithms like KMeans unless you standardise them.",
            kind="learn",
        )

    # ── Distributions ──
    with tab_dist:
        section("Feature Distributions")
        if num_cols:
            feat = st.selectbox("Select feature", num_cols, key="eda_feat")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(feature_histogram(df, feat), use_container_width=True)
            with c2:
                st.plotly_chart(feature_boxplot(df, feat), use_container_width=True)

            if len(num_cols) >= 2:
                section("Feature vs Feature Scatter")
                c1, c2, c3 = st.columns(3)
                fx = c1.selectbox("X axis", num_cols, key="sc_x")
                fy = c2.selectbox("Y axis", num_cols, index=min(1, len(num_cols) - 1), key="sc_y")
                color_by = c3.selectbox("Color by", ["None"] + cat_cols, key="sc_c")
                color_col = None if color_by == "None" else color_by
                st.plotly_chart(eda_scatter(df, fx, fy, color_col), use_container_width=True)
        else:
            st.info("No numeric features found.")

    # ── Correlations ──
    with tab_corr:
        section("Correlation Matrix")
        if len(num_cols) >= 2:
            st.plotly_chart(correlation_heatmap(df[num_cols]), use_container_width=True)
            high = get_high_corr_pairs(df)
            if high:
                st.markdown(
                    f'<div class="warn-box"><strong>⚠ High Correlation:</strong> '
                    f'{len(high)} pair(s) with |r| > 0.8. Consider removing one from each pair in the Clean step.</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    pd.DataFrame(high, columns=["Feature A", "Feature B", "r"]),
                    hide_index=True, use_container_width=True,
                )
            explain(
                "📐 What does correlation mean?",
                "<strong>Correlation (r)</strong> measures how closely two features move together. "
                "<strong>+1</strong> = perfectly in sync · <strong>-1</strong> = perfectly opposite · "
                "<strong>0</strong> = no relationship. Features with |r| > 0.8 carry redundant "
                "information — including both in clustering is like voting twice for the same thing. "
                "It's generally better to drop one of each highly-correlated pair.",
                kind="learn",
            )
        else:
            st.info("Need at least 2 numeric features.")

    # ── Outliers ──
    with tab_out:
        section("Outlier Detection")
        if num_cols:
            sel = st.selectbox("Feature", num_cols, key="out_feat")
            col_data = df[sel].dropna()
            z = np.abs(zscore(col_data, nan_policy="omit"))
            q25, q75 = col_data.quantile(0.25), col_data.quantile(0.75)
            iqr = q75 - q25
            iqr_low  = q25 - 1.5 * iqr
            iqr_high = q75 + 1.5 * iqr

            c1, c2, c3 = st.columns(3)
            c1.metric("Z-Score Outliers (|z|>3)", int((z > 3).sum()))
            c2.metric("IQR Outliers", int(((col_data < iqr_low) | (col_data > iqr_high)).sum()))
            c3.metric("Total Rows", len(col_data))

            st.plotly_chart(outlier_scatter(col_data, z, sel), use_container_width=True)

            explain(
                "🎯 What are outliers?",
                "Outliers are data points that differ dramatically from others. "
                "<strong>Z-score</strong>: flags points more than 3 standard deviations from the mean. "
                "<strong>IQR method</strong>: flags points beyond 1.5× the interquartile range (the distance between the 25th and 75th percentile). "
                "KMeans is very sensitive to outliers — a single extreme value can drag an entire cluster off-centre. "
                "DBSCAN, by contrast, naturally classifies outliers as 'noise' rather than forcing them into a cluster.",
                kind="learn",
            )

    # ── Categorical ──
    with tab_cat:
        section("Categorical Features")
        if cat_cols:
            sel_c = st.selectbox("Column", cat_cols, key="cat_sel")
            st.plotly_chart(cat_bar(df[sel_c], sel_c), use_container_width=True)
            explain(
                "🏷️ How are categorical features handled?",
                "Categorical features (text/object columns) cannot be used directly in distance-based "
                "algorithms because there's no numeric meaning to 'Male' vs 'Female' or 'Red' vs 'Blue'. "
                "The app automatically applies <strong>one-hot encoding</strong>: each unique category "
                "becomes its own binary (0/1) column. For example, 'Gender' with values Male/Female becomes "
                "two columns: Gender_Male and Gender_Female.",
                kind="learn",
            )
        else:
            st.info("No categorical features found.")

    st.session_state.eda_done = True
    nav_buttons("Proceed to Data Cleaning →", proceed_step=2, back_step=0, back_label="Load Data", proceed_key="eda_proceed")
    recovery_panel(1)


# ============================================================
# STEP 2 — CLEAN
# ============================================================

def step_clean() -> None:
    section("Step 3 · Data Cleaning")
    _need("df_raw", "Please load a dataset first.")
    df = st.session_state.df_raw

    explain(
        "🧹 Why clean data?",
        "Real-world data is almost always messy. Missing values cause most algorithms to crash or "
        "produce meaningless results. Useless columns like IDs add noise without contributing signal. "
        "Extreme outliers drag cluster centres away from where the true groups lie. "
        "Cleaning is not optional — it's the difference between a model that works and one that misleads you.",
        kind="learn",
    )

    # ── Missing values ──
    section("Missing Value Strategy")
    miss_pct = (df.isnull().sum() / len(df) * 100).round(1)
    miss_df  = miss_pct[miss_pct > 0].reset_index()
    if miss_df.empty:
        st.markdown(
            '<div class="success-box"><strong>✓ No missing values!</strong> Your dataset is complete.</div>',
            unsafe_allow_html=True,
        )
    else:
        miss_df.columns = ["Column", "Missing %"]
        st.dataframe(miss_df, hide_index=True, use_container_width=True)

    imputer_choice = st.selectbox(
        "Imputation Strategy", ["Mean", "Median", "KNN", "Drop Rows"],
        help="Mean/Median fill with average. KNN uses similar rows. Drop Rows removes them.",
    )
    st.session_state["imputer"] = imputer_choice

    # ── Drop Rows safety check ──
    if imputer_choice == "Drop Rows" and not miss_df.empty:
        rows_after = df.dropna().shape[0]
        rows_before = df.shape[0]
        pct_remaining = round(rows_after / rows_before * 100, 1)
        if rows_after < 50 or pct_remaining < 30:
            st.markdown(
                f'<div class="warn-box"><strong>⚠ Data Loss Warning:</strong> '
                f'Dropping rows with missing values would reduce your dataset from '
                f'<strong>{rows_before}</strong> to <strong>{rows_after}</strong> rows '
                f'(<strong>{pct_remaining}%</strong> remaining). '
                f'This may not leave enough data for reliable clustering. '
                f'Consider using KNN imputation instead.</div>',
                unsafe_allow_html=True,
            )

    explain(
        "🔢 Which imputer to choose?",
        "<strong>Mean</strong>: Fast and works well when data is roughly normally distributed. "
        "Can be pulled off by outliers since the mean is sensitive to extremes. "
        "<strong>Median</strong>: Better than mean when outliers are present, because the median is "
        "not affected by extreme values. "
        "<strong>KNN</strong>: The most intelligent option — it finds the K most similar rows "
        "and borrows their values. Best quality, but slowest. "
        "<strong>Drop Rows</strong>: Safest when missing values are truly random and your dataset "
        "is large enough to afford the loss.",
        kind="learn",
    )

    # ── Column selection ──
    section("Column Management")
    auto_remove = get_auto_remove_cols(df)
    if auto_remove:
        safe_names = ", ".join(_safe_html(c) for c in auto_remove)
        st.markdown(
            f'<div class="warn-box"><strong>⚠ Suggested for removal:</strong> '
            f'{safe_names} — likely IDs (unique per row) or constant values (no variation).</div>',
            unsafe_allow_html=True,
        )

    keep_cols = st.multiselect(
        "Columns to KEEP for clustering",
        df.columns.tolist(),
        default=[c for c in df.columns if c not in auto_remove],
    )

    # ── Outlier removal ──
    section("Outlier Handling")
    outlier_method = st.selectbox(
        "Outlier Removal Method",
        ["None", "Z-Score (|z| > 3)", "IQR (1.5×IQR)", "Clip to 99th Percentile"],
    )
    st.session_state["outlier_method"] = outlier_method

    explain(
        "✂️ When to remove outliers?",
        "Use <strong>Z-Score or IQR removal</strong> when outliers are clearly data errors "
        "(e.g. an age of 999 or a salary of -1). These methods remove the entire row. "
        "Use <strong>Clip</strong> to keep rows but cap extreme values at the 1st/99th percentile — "
        "useful when the data is real but extreme. "
        "Use <strong>None</strong> when using DBSCAN, which naturally identifies outliers as 'noise points' "
        "rather than forcing them into a cluster.",
        kind="learn",
    )

    # ── Scaler ──
    section("Scaling Method")
    scaler_choice = st.selectbox(
        "Feature Scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler"]
    )
    st.session_state["scaler"] = scaler_choice

    explain(
        "📏 Why scale features?",
        "Distance-based algorithms (like KMeans) measure similarity using numeric distance. "
        "Without scaling, a feature like Salary (0–100,000) will completely overpower Age (0–100) "
        "simply because of its larger numeric range — not because it's more important. "
        "<strong>StandardScaler</strong> (recommended): transforms each feature to mean=0 and std=1. "
        "<strong>MinMaxScaler</strong>: squeezes values to the range [0, 1]. "
        "<strong>RobustScaler</strong>: uses median and IQR instead of mean and std — "
        "the best choice when your data has many outliers you can't or don't want to remove.",
        kind="learn",
    )

    # ── Apply ──
    if st.button("✅ Apply Cleaning & Continue →", type="primary"):
        if not keep_cols:
            st.error("Please select at least one column to keep.")
            st.stop()

        df_c = df[keep_cols].copy()
        df_c = apply_imputation(df_c, imputer_choice)
        df_c = apply_outlier_removal(df_c, outlier_method)

        # Validate enough data remains
        if len(df_c) < 10:
            st.error(f"Only {len(df_c)} rows remain after cleaning. Need at least 10 for clustering.")
            st.stop()

        st.session_state.df_clean = df_c
        st.session_state.preprocessing_done = True
        st.markdown(
            f'<div class="success-box"><strong>✓ Cleaning applied!</strong> '
            f'{df_c.shape[0]} rows × {df_c.shape[1]} columns ready for feature engineering.</div>',
            unsafe_allow_html=True,
        )
        st.session_state.step = 3
        st.rerun()

   # nav_buttons(back_step=1, back_label="EDA", proceed_label="✅ Apply & Continue →",
   #             proceed_key="clean_nav_proceed")
    recovery_panel(2)


# ============================================================
# STEP 3 — FEATURE ENGINEERING
# ============================================================

def step_features() -> None:
    section("Step 4 · Feature Engineering & Selection")
    _need("df_clean", "Complete the cleaning step first.")
    df = st.session_state.df_clean

    explain(
        "⚙️ What is Feature Engineering?",
        "Feature engineering is the process of creating new, more informative inputs for your model "
        "by combining or transforming existing columns. For example, if you have 'Total Spend' and "
        "'Number of Visits', dividing them creates 'Spend per Visit' — a feature that may reveal "
        "customer value more clearly than either original column alone. "
        "You can also remove noisy features — counterintuitively, fewer but better features "
        "often produce <em>cleaner</em> clusters than many weak ones.",
        kind="learn",
    )

    num_cols = get_numeric_cols(df)
    df_eng   = df.copy()

    # Show previously created features
    created = st.session_state.get("created_features", [])
    if created:
        st.markdown(
            f'<div class="success-box"><strong>✓ Features created this session:</strong> '
            f'{", ".join(_safe_html(f) for f in created)}</div>',
            unsafe_allow_html=True,
        )

    # ── Ratio / Interaction features ──
    section("Create New Features")
    if len(num_cols) >= 2:
        with st.expander("➕ Add ratio feature (A ÷ B)", expanded=st.session_state.get("expand_ratio", False)):
            c1, c2, c3 = st.columns(3)
            feat_a = c1.selectbox("Numerator",   num_cols, key="ra")
            feat_b = c2.selectbox("Denominator", num_cols, index=1, key="rb")
            feat_n = c3.text_input("Name", value=f"{feat_a}_per_{feat_b}")
            if st.button("Create Ratio Feature", key="create_ratio"):
                if feat_n in df_eng.columns:
                    st.warning(f"Feature '{feat_n}' already exists. Choose a different name.")
                else:
                    df_eng[feat_n] = df_eng[feat_a] / (df_eng[feat_b].replace(0, np.nan) + 1e-9)
                    st.session_state.df_clean = df_eng
                    feats = st.session_state.get("created_features", [])
                    feats.append(feat_n)
                    st.session_state.created_features = feats
                    st.session_state.expand_ratio = True
                    st.success(f"✓ Created: {_safe_html(feat_n)}")
                    st.rerun()

        with st.expander("✖️ Add interaction feature (A × B)", expanded=st.session_state.get("expand_interact", False)):
            c1, c2, c3 = st.columns(3)
            feat_a2 = c1.selectbox("Feature A", num_cols, key="ia")
            feat_b2 = c2.selectbox("Feature B", num_cols, index=min(1, len(num_cols) - 1), key="ib")
            feat_n2 = c3.text_input("Name", value=f"{feat_a2}_x_{feat_b2}")
            if st.button("Create Interaction Feature", key="create_interact"):
                if feat_n2 in df_eng.columns:
                    st.warning(f"Feature '{feat_n2}' already exists. Choose a different name.")
                else:
                    df_eng[feat_n2] = df_eng[feat_a2] * df_eng[feat_b2]
                    st.session_state.df_clean = df_eng
                    feats = st.session_state.get("created_features", [])
                    feats.append(feat_n2)
                    st.session_state.created_features = feats
                    st.session_state.expand_interact = True
                    st.success(f"✓ Created: {_safe_html(feat_n2)}")
                    st.rerun()
    else:
        st.info("Need ≥ 2 numeric columns to create ratio or interaction features.")

    # ── Feature selection ──
    section("Feature Selection")
    num_cols2 = get_numeric_cols(df_eng)
    low_var   = get_low_variance_features(df_eng) if num_cols2 else []
    if low_var:
        safe_lv = ", ".join(_safe_html(c) for c in low_var)
        st.markdown(
            f'<div class="warn-box"><strong>⚠ Low Variance Features:</strong> '
            f'{safe_lv} — these are near-constant and may add noise without adding signal. '
            f'Consider deselecting them below.</div>',
            unsafe_allow_html=True,
        )

    selected = st.multiselect(
        "Features to include in clustering",
        df_eng.columns.tolist(),
        default=[c for c in df_eng.columns if c not in low_var],
    )

    # ── PCA chart ──
    section("PCA Variance Explained")
    indiv, cum = get_pca_explained(df_eng)
    if indiv is not None:
        st.plotly_chart(pca_variance_chart(indiv, cum), use_container_width=True)
        explain(
            "🔵 What is PCA?",
            "<strong>Principal Component Analysis (PCA)</strong> is a technique that compresses "
            "many features into fewer 'super-features' called components, while preserving as much "
            "information (variance) as possible. The chart shows how many components are needed to "
            "explain 80% of the total variance in your data. "
            "In the clustering and results steps, PCA is used purely for <em>visualisation</em> — "
            "to reduce your data to 2D so it can be plotted. The actual clustering happens in the "
            "full feature space.",
            kind="learn",
        )

    if st.button("✅ Lock Features & Continue →", type="primary", key="features_proceed"):
        if not selected:
            st.error("Please select at least one feature.")
            st.stop()
        df_final = df_eng[selected]
        st.session_state.df_engineered = df_final
        st.session_state.engineering_done = True
        st.session_state.step = 4
        st.rerun()

   # nav_buttons(back_step=2, back_label="Cleaning", proceed_label="✅ Lock & Continue →",
   #             proceed_key="features_nav_proceed") 
    recovery_panel(3)


# ============================================================
# STEP 4 — CLUSTERING
# ============================================================

def step_cluster() -> None:
    section("Step 5 · Clustering")
    df = st.session_state.df_engineered if st.session_state.df_engineered is not None else st.session_state.df_clean
    if df is None:
        st.warning("Complete previous steps first.")
        st.stop()

    # ── Warn if skipped cleaning ──
    if not st.session_state.get("preprocessing_done"):
        st.markdown(
            '<div class="warn-box"><strong>⚠ Cleaning step skipped.</strong> '
            'Results may be unreliable for datasets with text columns, missing values, '
            'or very different numeric scales. Consider going back to run Step 3 first.</div>',
            unsafe_allow_html=True,
        )

    # ── Warn if only 1 column ──
    if len(df.columns) < 2:
        st.error("Need at least 2 columns to cluster. Go back to Feature Engineering and select more features.")
        st.stop()

    explain(
        "🤖 What happens in this step?",
        "Clustering is <strong>unsupervised machine learning</strong> — unlike classification, "
        "we don't predict a known label. Instead, the algorithm discovers natural groups in the data "
        "based on similarity. Points that are close together in feature space end up in the same cluster. "
        "You choose the algorithm and its settings; the algorithm finds the groups automatically. "
        "Use <strong>Manual Mode</strong> to understand each algorithm, or <strong>AutoML Mode</strong> "
        "to let the app try dozens of configurations and pick the best one automatically.",
        kind="learn",
    )

    mode_tab, auto_tab = st.tabs(["🎓 Manual Mode", "⚡ AutoML Mode"])

    # ── Manual Mode ──
    with mode_tab:
        section("Choose Your Algorithm")
        algo = st.selectbox("Algorithm", list(ALGO_INFO.keys()))
        info = ALGO_INFO[algo]

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(
                f'<div class="card" style="height:100%">'
                f'<div style="font-size:2.5rem;margin-bottom:0.5rem">{info["icon"]}</div>'
                f'<div style="font-family:IBM Plex Mono;font-size:0.65rem;letter-spacing:0.12em;'
                f'text-transform:uppercase;color:#fbbf24;margin-bottom:0.4rem">{_safe_html(info["level"])}</div>'
                f'<div style="font-size:0.85rem;color:#8b90b0;line-height:1.6">{info["desc"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="card">'
                f'<div style="margin-bottom:0.6rem">'
                f'<div style="font-family:IBM Plex Mono;font-size:0.65rem;color:#34d399;'
                f'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.3rem">✓ Best when</div>'
                f'<div style="font-size:0.85rem;color:#8b90b0">{info["best"]}</div>'
                f'</div>'
                f'<div>'
                f'<div style="font-family:IBM Plex Mono;font-size:0.65rem;color:#fb7185;'
                f'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.3rem">✗ Avoid when</div>'
                f'<div style="font-size:0.85rem;color:#8b90b0">{info["worst"]}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        section("Hyperparameters")
        params = _algo_params_ui(algo)

        scaler_c = st.selectbox(
            "Scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler"], key="c_scaler"
        )
        reduction_c = st.selectbox(
            "Visualisation", ["PCA", "t-SNE"], key="c_red"
        )
        st.session_state["reduction"] = reduction_c

        if st.button("▶ Train Model", type="primary", key="train_manual"):
            _train_model(df, algo, params, scaler_c)

        # Results navigation after training
        if st.session_state.get("clustering_done", False):
            st.success("✅ Model trained successfully!")
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("View Results →", type="primary", key="goto_results_manual"):
                    st.session_state.step = 5
                    st.rerun()
            with col2:
                if st.button("← Back to Features", key="manual_back_to_features"):
                    st.session_state.step = 3
                    st.rerun()

    # ── AutoML Mode ──
    with auto_tab:
        section("AutoML — Automated Model Selection")
        explain(
            "⚡ What does AutoML do?",
            "AutoML tries <strong>dozens of algorithm and hyperparameter combinations</strong> "
            "automatically and selects the configuration that achieves the best Silhouette Score. "
            "This is the best starting point when you're not sure which algorithm suits your data.",
            kind="learn",
        )

        scaler_a = st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler"], key="a_scaler")
        reduction_a = st.selectbox("Visualisation", ["PCA", "t-SNE"], key="a_red")
        n_km = st.slider("KMeans: max K to try", 2, 12, 8)

        if st.button("⚡ Run AutoML", type="primary", key="run_automl"):
            _run_automl(df, scaler_a, reduction_a, n_km)

        # Navigation
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("← Features", key="automl_back"):
                st.session_state.step = 3
                st.rerun()
        with col2:
            if st.session_state.get("clustering_done", False):
                if st.button("View Results →", type="primary", key="automl_to_results"):
                    st.session_state.step = 5
                    st.rerun()

# ============================================================
# STEP 5 — RESULTS
# ============================================================

def step_results() -> None:
    section("Step 6 · Results & Visualisation")

    # Prioritize data sources
    df_src = None
    for key in ["df_engineered", "df_clean", "df_raw"]:
        if st.session_state.get(key) is not None:
            df_src = st.session_state.get(key)
            break

    if df_src is None:
        st.warning("Run clustering first.")
        st.stop()

    X = st.session_state.get("X_processed")
    labels = st.session_state.get("labels")
    metrics = st.session_state.get("metrics")
    model_name = st.session_state.get("model_name", "Unknown")

    if X is None or labels is None or metrics is None:
        st.warning("Run clustering first.")
        st.stop()

    df_r = df_src.reset_index(drop=True).copy()

    # Align rows with labels
    if len(df_r) != len(labels):
        st.warning("Row count mismatch. Trying fallback...")
        fallback = st.session_state.get("df_clean") or st.session_state.get("df_raw")
        if fallback is not None and len(fallback) == len(labels):
            df_r = fallback.reset_index(drop=True).copy()
        else:
            st.error("Cannot align data with cluster labels. Please re-run clustering.")
            st.stop()

    df_r["Cluster"] = labels

    # ── Metrics strip ──
    metric_strip(metrics, model_name)

    if metrics.get("Noise pts", 0) > 0:
        st.markdown(
            f'<div class="warn-box"><strong>⚠ {metrics["Noise pts"]} noise points</strong> '
            f'(DBSCAN label -1) — these are outliers not assigned to any cluster.</div>',
            unsafe_allow_html=True,
        )

    explain(
        "📊 How to read the scores?",
        "<strong>Silhouette Score</strong> (−1 to +1): measures how well each point fits its cluster "
        "vs neighbouring clusters. >0.5 = strong clusters. "
        "<strong>Davies-Bouldin</strong>: lower is better. "
        "<strong>Calinski-Harabasz</strong>: higher is better.",
        kind="learn",
    )

    reduction = st.session_state.get("reduction", "PCA")

    tab_sc, tab_dist, tab_prof, tab_heat, tab_elbow, tab_dend, tab_exp = st.tabs([
        "🗺️ Scatter", "📊 Distribution", "🧬 Profiles",
        "🌡️ Heatmap", "📐 Elbow / Sweep", "🌳 Dendrogram", "💾 Export"
    ])

    with tab_sc:
        with st.spinner(f"Computing {reduction} projection…"):
            X2d = reduce_2d(X, reduction)
        num_cols = [c for c in df_r.select_dtypes(include=np.number).columns if c != "Cluster"]
        hover_col = st.selectbox("Hover column", ["None"] + num_cols, key="hover_col")
        hover_s = df_r[hover_col].reset_index(drop=True) if hover_col != "None" else None

        st.plotly_chart(
            scatter_clusters(X2d, labels, f"Cluster Map · {reduction}", hover_s),
            use_container_width=True,
        )

    with tab_dist:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(cluster_bar(labels), use_container_width=True)
        with c2:
            st.plotly_chart(cluster_pie(labels), use_container_width=True)

    with tab_prof:
        if fig_radar := radar_profile(df_r):
            st.plotly_chart(fig_radar, use_container_width=True)
        if fig_imp := feature_importance_chart(df_r):
            st.plotly_chart(fig_imp, use_container_width=True)
        num_cols = [c for c in df_r.select_dtypes(include=np.number).columns if c != "Cluster"]
        if num_cols:
            section("Mean Values by Cluster")
            st.dataframe(df_r.groupby("Cluster")[num_cols].mean().round(3), use_container_width=True)

    with tab_heat:
        if fig_heat := cluster_heatmap(df_r):
            st.plotly_chart(fig_heat, use_container_width=True)
        if fig_pair := scatter_matrix(df_r):
            st.plotly_chart(fig_pair, use_container_width=True)

    with tab_elbow:
        max_k = st.slider("Max K for sweep", 3, 15, 10)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(elbow_chart(X, max_k), use_container_width=True)
        with c2:
            st.plotly_chart(silhouette_sweep(X, max_k), use_container_width=True)

    with tab_dend:
        if fig_d := dendrogram_chart(df_r):
            try:
                import matplotlib.pyplot as plt
                st.pyplot(fig_d)
            finally:
                plt.close(fig_d)

    with tab_exp:
        section("Export Your Results")
        report_md = generate_report(df_r, metrics, model_name)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.download_button("⬇ Clustered CSV", df_r.to_csv(index=False).encode(),
                             "clustered_data.csv", "text/csv", use_container_width=True)
        with c2:
            if st.session_state.get("model"):
                buf = io.BytesIO()
                joblib.dump(st.session_state.model, buf)
                st.download_button("💾 Model (.joblib)", buf.getvalue(),
                                 "model.joblib", use_container_width=True)
        with c3:
            st.download_button("📋 Full Report (.md)", report_md.encode(),
                             "analysis_report.md", "text/markdown", use_container_width=True)

        with st.expander("👁 Preview Full Report"):
            st.markdown(report_md)

    # Navigation
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back to Clustering", key="results_back_to_cluster"):
            st.session_state.step = 4
            st.rerun()
    recovery_panel(5)
# ============================================================
# STEP 6 — LEARN
# ============================================================

def step_learn() -> None:
    section("Step 7 · Learning Module — Beginner to Pro")

    st.markdown(
        """
        <div style="background:linear-gradient(135deg,rgba(34,211,238,0.06),rgba(167,139,250,0.06));
        border:1px solid rgba(34,211,238,0.15);border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem;">
          <div style="font-family:Syne;font-size:1.3rem;font-weight:800;color:#e2e4f0;margin-bottom:0.4rem;">
            🎓 The ML Clustering Roadmap
          </div>
          <div style="font-size:0.88rem;color:#8b90b0;line-height:1.7;">
            Everything you need to understand, apply, and explain clustering — from first model to production.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4 = st.tabs(["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced", "📚 Glossary"])

    with t1:
        _beginner_content()
    with t2:
        _intermediate_content()
    with t3:
        _advanced_content()
    with t4:
        _glossary()

    progress_tracker()

    # ── Restart with confirmation ──
    section("Start a New Analysis")
    if st.session_state.get("confirm_reset"):
        st.warning("⚠ This will clear all data and results. Are you sure?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✓ Yes, start fresh", type="primary", key="learn_confirm_reset"):
                from config.settings import SESSION_DEFAULTS
                for k, v in SESSION_DEFAULTS.items():
                    st.session_state[k] = v
                st.rerun()
        with c2:
            if st.button("Cancel", key="learn_cancel_reset"):
                st.session_state.confirm_reset = False
                st.rerun()
    else:
        if st.button("🔄 Start New Analysis", type="primary", key="learn_new_analysis"):
            st.session_state.confirm_reset = True
            st.rerun()


# ── Learn content helpers ────────────────────────────────────

def _beginner_content() -> None:
    topics = [
        (
            "What is Machine Learning?",
            "Machine learning is a branch of artificial intelligence where computers learn patterns "
            "from data rather than following explicit, hand-coded rules. Instead of a programmer writing "
            "instructions like 'if age &gt; 60 and income &gt; 50,000 then label as high-value', a machine "
            "learning algorithm looks at thousands of examples and discovers those rules on its own. "
            "This is powerful because the patterns in real data are often too complex or numerous for a "
            "human to specify manually. The result is a <strong>model</strong> — a mathematical function "
            "that takes inputs and produces predictions or groupings.",
            "A music streaming platform uses ML to recommend songs. It doesn't have a rule for every "
            "artist and listener. Instead, it finds patterns: 'listeners who liked X also liked Y.' "
            "The algorithm discovers this automatically from millions of play histories.",
        ),
        (
            "What is Clustering?",
            "Clustering is a type of <strong>unsupervised machine learning</strong> — 'unsupervised' "
            "means there are no labels or correct answers in the training data. The algorithm looks at "
            "the raw features and discovers natural groups by itself, based purely on which points are "
            "similar to each other. This is different from classification (where you already have labels "
            "like 'spam/not spam') or regression (where you predict a number). "
            "Clustering asks: 'Are there hidden groups here I didn't know about?'",
            "A retailer uploads 50,000 customer transaction records — no labels. "
            "Clustering automatically discovers 'high-value frequent buyers', 'occasional discount seekers', "
            "and 'one-time purchasers'. The business can then target each group differently.",
        ),
        (
            "How Does KMeans Work?",
            "KMeans is the most popular clustering algorithm. It works in four repeating steps: "
            "<strong>1. Initialise</strong>: randomly place K cluster centres (centroids) in the data space. "
            "<strong>2. Assign</strong>: assign each data point to the nearest centroid. "
            "<strong>3. Update</strong>: move each centroid to the mean (average position) of all "
            "points assigned to it. "
            "<strong>4. Repeat</strong>: repeat steps 2–3 until the centroids stop moving significantly. "
            "The algorithm converges when no point changes its cluster assignment between iterations.",
            "Think of K magnets placed randomly among iron filings. Each filing is attracted to the "
            "nearest magnet. Then each magnet moves to the centre of its filings. This repeats until "
            "the magnets settle — those stable positions are your cluster centres.",
        ),
        (
            "What is a Silhouette Score?",
            "The Silhouette Score measures how well each data point fits within its assigned cluster "
            "compared to neighbouring clusters. For each point, it calculates: "
            "<strong>a</strong> = average distance to other points in the same cluster, and "
            "<strong>b</strong> = average distance to points in the nearest other cluster. "
            "The score = (b − a) / max(a, b). "
            "<strong>+1</strong> means the point is clearly in the right cluster. "
            "<strong>0</strong> means the point is on the boundary between two clusters. "
            "<strong>−1</strong> means the point may be in the wrong cluster.",
            "Score of 0.7+ = excellent, well-separated clusters. "
            "Score of 0.5–0.7 = reasonable structure. "
            "Score below 0.3 = overlapping or poorly-defined clusters.",
        ),
        (
            "What is Scaling and Why Does It Matter?",
            "Scaling transforms your features so they operate on the same numeric range. "
            "Without scaling, features with larger values dominate distance calculations. "
            "For example: Age (0–100) and Annual Salary (0–100,000). "
            "A difference of 1 year in age = distance of 1. "
            "A difference of £1,000 in salary = distance of 1,000. "
            "The algorithm 'sees' salary differences as 1,000× more important than age differences — "
            "not because they are, but simply because of the numeric scale. "
            "<strong>StandardScaler</strong> fixes this by converting each feature to have mean=0 and std=1.",
            "After StandardScaler: a 1-unit difference in any feature means exactly '1 standard deviation "
            "away from average' — now all features compete on equal footing.",
        ),
    ]
    for title, body, example in topics:
        with st.expander(f"📖 {title}"):
            st.markdown(
                f'<div style="font-size:0.88rem;color:#8b90b0;line-height:1.85;margin-bottom:1rem">{body}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:rgba(34,211,238,0.05);border-left:3px solid #22d3ee;'
                f'border-radius:4px;padding:0.9rem 1rem;font-size:0.85rem;color:#8b90b0;line-height:1.75;">'
                f'💡 <strong>Real-world example:</strong> {example}</div>',
                unsafe_allow_html=True,
            )


def _intermediate_content() -> None:
    topics = [
        (
            "The Full ML Pipeline",
            "A real ML project is never just 'train a model'. There are 8 stages, and skipping "
            "any of them leads to results that look correct in isolation but fail in practice. "
            "Each stage feeds the next — bad data at stage 1 produces bad clusters at stage 7, "
            "no matter how sophisticated the algorithm.",
            ["1️⃣ <strong>Load Data</strong> — get the raw dataset into a structured format",
             "2️⃣ <strong>EDA</strong> — understand distributions, correlations, outliers",
             "3️⃣ <strong>Clean</strong> — handle missing values, remove noise, scale features",
             "4️⃣ <strong>Feature Engineering</strong> — create, select, and transform inputs",
             "5️⃣ <strong>Preprocessing</strong> — encode categoricals, apply final scaling",
             "6️⃣ <strong>Model Selection</strong> — choose algorithm and tune hyperparameters",
             "7️⃣ <strong>Evaluation</strong> — measure quality with multiple metrics",
             "8️⃣ <strong>Interpret & Deploy</strong> — explain clusters, save model, monitor drift"],
        ),
        (
            "When to Use Each Algorithm",
            "Choosing the right algorithm matters as much as tuning its parameters. "
            "The key factors are: do you know K? Is your data large? Do you expect noise/outliers? "
            "Are clusters roughly round or do they have complex shapes?",
            [
                "<strong>KMeans</strong>: best default. Fast, scalable, well-understood. Use when K is known, data is clean, and clusters are roughly spherical.",
                "<strong>DBSCAN</strong>: best for noisy data with outliers. Discovers K automatically. Handles arbitrary cluster shapes. Struggles with varying densities.",
                "<strong>Agglomerative</strong>: best when you want a hierarchy (dendrogram). Good for small-to-medium datasets where you want to explore multiple K values visually.",
                "<strong>Spectral</strong>: best for complex non-spherical clusters (rings, crescents). Very expensive — avoid on datasets > 10,000 rows.",
                "<strong>Birch</strong>: best for very large datasets (100k+ rows). Memory-efficient. Assumes spherical clusters like KMeans.",
                "<strong>MeanShift</strong>: discovers K automatically like DBSCAN. Very slow on large datasets. Use only on small datasets (< 5,000 rows).",
            ],
        ),
        (
            "What is Feature Engineering?",
            "Feature engineering is the practice of creating new, more informative inputs by "
            "combining or transforming existing columns. The intuition is that raw features often "
            "capture individual measurements, but the relationships between them carry more signal. "
            "A customer's 'total spend' and 'number of visits' are useful, but "
            "'spend per visit' may more clearly separate high-value from budget-conscious customers. "
            "Feature engineering is part art, part domain knowledge — the best features come from "
            "understanding your data's business context.",
            [
                "<strong>Ratio features</strong> (A ÷ B): 'Revenue per Employee', 'Spend per Visit', 'Conversions per Click'",
                "<strong>Interaction features</strong> (A × B): 'Age × Income' can capture a different 'life stage' dimension",
                "<strong>Log transforms</strong>: useful for skewed distributions like income or population — compresses large values",
                "<strong>Binning</strong>: turning continuous age into 'under 25 / 25–45 / over 45' can make clusters more interpretable",
                "<strong>Feature selection</strong>: removing low-variance or highly-correlated features often improves clustering quality",
            ],
        ),
        (
            "What is Imputation?",
            "Imputation is the process of filling in missing values so that algorithms (which cannot "
            "handle NaN values) can run on your data. The choice of imputation strategy affects your "
            "clustering results more than most people realise — different strategies can shift cluster "
            "boundaries. The best strategy depends on why values are missing: "
            "Are they missing randomly? Are they systematically missing for one group?",
            [
                "<strong>Mean imputation</strong>: replaces missing values with the column average. Fast, but sensitive to outliers and assumes data is normally distributed.",
                "<strong>Median imputation</strong>: replaces with the middle value. Better than mean for skewed distributions or when outliers exist.",
                "<strong>KNN imputation</strong>: finds the K most similar rows (by other features) and borrows their values. Most accurate, but slowest.",
                "<strong>Drop Rows</strong>: removes any row with at least one missing value. Only use when missingness is rare and your dataset is large.",
                "<strong>Rule of thumb</strong>: if < 5% of values are missing, mean/median is fine. If > 20% are missing in a column, consider dropping the entire column instead.",
            ],
        ),
        (
            "Choosing K — The Elbow and Silhouette Methods",
            "Choosing the number of clusters K is the most common challenge in KMeans clustering. "
            "There's no single correct answer — K is a tuning decision, not a discovery. "
            "Always triangulate between at least two methods, and validate the final K "
            "against business logic: 'Does splitting customers into 5 groups give us actionable "
            "segments, or are some indistinguishable?'",
            [
                "<strong>Elbow method</strong>: plot inertia (WCSS) vs K. The 'elbow' — where the curve flattens — suggests diminishing returns. Pick the K at the bend.",
                "<strong>Silhouette sweep</strong>: plot Silhouette Score vs K. Pick the K with the highest score.",
                "<strong>Domain knowledge</strong>: business logic often dictates K. A marketing team may only be able to run 3 campaigns — so K=3 regardless of the elbow.",
                "<strong>Stability test</strong>: run KMeans 5× with different random seeds. If K=4 always gives similar clusters, that K is stable.",
                "<strong>Gap statistic</strong>: a more rigorous method that compares inertia to that of random data. Computationally expensive but statistically principled.",
            ],
        ),
        (
            "The Curse of Dimensionality",
            "As the number of features increases, distance metrics become unreliable. "
            "In high-dimensional spaces, the difference between the nearest and farthest neighbour "
            "approaches zero — meaning all points seem equally far apart. This makes "
            "distance-based clustering (KMeans, DBSCAN) less effective. "
            "The curse is real: with 50 features, KMeans may produce meaningless clusters "
            "even on perfectly structured data.",
            [
                "<strong>Fix 1: PCA</strong> — reduce to the top N components that explain 80–90% of variance. Often 2–10 components replace 50+ original features.",
                "<strong>Fix 2: Feature selection</strong> — remove low-variance features and one from each highly-correlated pair.",
                "<strong>Fix 3: Cosine similarity</strong> — for text/document data, use cosine distance instead of Euclidean. It's less affected by dimensionality.",
                "<strong>Rule of thumb</strong>: for N features, aim for datasets with at least 10×N rows. Fewer rows than this and clusters will be statistically unreliable.",
            ],
        ),
    ]
    for title, intro, bullets in topics:
        with st.expander(f"⚙️ {title}"):
            st.markdown(
                f'<div style="font-size:0.88rem;color:#8b90b0;line-height:1.85;margin-bottom:0.9rem">{intro}</div>',
                unsafe_allow_html=True,
            )
            for b in bullets:
                st.markdown(
                    f'<div style="font-size:0.85rem;color:#8b90b0;line-height:1.9;padding:0.2rem 0 0.2rem 0.5rem;'
                    f'border-left:2px solid #1e2035;margin:0.3rem 0">• {b}</div>',
                    unsafe_allow_html=True,
                )


def _advanced_content() -> None:
    topics = [
        (
            "Evaluation Beyond Silhouette",
            "Relying on a single metric is dangerous — every metric has blind spots. "
            "Use a battery of internal metrics (no ground truth needed) and, where possible, "
            "external metrics (if you have partial labels). "
            "The most important validation is ultimately human: do the clusters make sense to "
            "domain experts? Can you name them? Would different business decisions follow from "
            "different clusters? If yes, you have good clusters.",
            [
                "<strong>Silhouette Score</strong>: good overall measure. Penalises overlapping clusters. Can be fooled by convex shapes.",
                "<strong>Davies-Bouldin Index</strong>: lower = better. Penalises clusters that are large relative to their inter-cluster distance. Faster to compute than Silhouette.",
                "<strong>Calinski-Harabasz (Variance Ratio)</strong>: higher = better. Works well for compact, well-separated clusters. Biased toward larger K.",
                "<strong>Dunn Index</strong>: ratio of min inter-cluster distance to max intra-cluster diameter. Higher = better. Sensitive to noise.",
                "<strong>ARI / NMI (external)</strong>: Adjusted Rand Index and Normalised Mutual Information require ground truth labels. Use for benchmarking on labelled datasets.",
            ],
        ),
        (
            "Preprocessing Choices and Their Effects",
            "The preprocessing pipeline — scaler, imputer, encoder — affects your clustering results "
            "as much as the algorithm choice. There is no universal best approach; the optimal "
            "preprocessing depends on your data's distribution, the presence of outliers, and the "
            "algorithm you're using.",
            [
                "<strong>StandardScaler</strong>: assumes Gaussian-ish distribution. Best default for KMeans and Agglomerative.",
                "<strong>RobustScaler</strong>: uses IQR instead of std. Best when outliers are present and you can't or won't remove them.",
                "<strong>MinMaxScaler</strong>: compresses to [0,1]. Good for neural network inputs; avoid with DBSCAN (epsilon becomes scale-dependent).",
                "<strong>PCA before clustering</strong>: can help by removing noise dimensions, but can hurt by collapsing meaningful variation. Test both with and without.",
                "<strong>Mixed data (numeric + categorical)</strong>: one-hot encoding inflates dimensionality. Consider Gower distance or FAMD (Factor Analysis of Mixed Data) for heavy categorical data.",
                "<strong>Order matters</strong>: always impute → scale → encode. Encoding before scaling causes the scaler to treat binary indicators the same as continuous measurements.",
            ],
        ),
        (
            "DBSCAN Parameter Tuning",
            "DBSCAN has two parameters — ε (epsilon) and min_samples — that are notoriously "
            "difficult to tune. The wrong values produce either everything as noise (ε too small) "
            "or one giant cluster (ε too large). There are principled methods for choosing both.",
            [
                "<strong>ε from k-distance graph</strong>: for each point, compute the distance to its k-th nearest neighbour (where k = min_samples). Sort these distances in ascending order and plot them. The 'knee' of this curve is a good ε.",
                "<strong>min_samples rule of thumb</strong>: set to 2 × number of features (after encoding). For low-dimensional data (2–4 features), min_samples = 4 is a safe start.",
                "<strong>HDBSCAN</strong>: a hierarchical extension that removes the need for ε entirely. It finds clusters at varying density levels. Available as a separate Python package (hdbscan). Often outperforms DBSCAN on real-world data.",
                "<strong>When DBSCAN produces one giant cluster</strong>: ε is too large. Reduce it by 30–50%.",
                "<strong>When DBSCAN produces mostly noise</strong>: ε is too small or min_samples is too high. Increase ε or reduce min_samples.",
            ],
        ),
        (
            "Production Deployment",
            "Deploying a clustering model to production introduces challenges that don't exist "
            "in a notebook: new data arrives in real-time, cluster assignments must be stable "
            "across time, and models need versioning and monitoring.",
            [
                "<strong>Sklearn Pipeline</strong>: wrap your preprocessor and model in a single sklearn Pipeline object. This ensures the exact same transformations are applied to new data as were applied to training data. Save with joblib.",
                "<strong>Cluster drift</strong>: re-run clustering monthly on fresh data. Track whether the Silhouette Score degrades — this indicates that customer behaviour or data distribution has shifted.",
                "<strong>Label assignment</strong>: for KMeans, new data points can be assigned to clusters by finding the nearest centroid (model.predict()). DBSCAN doesn't natively support this — build a KNN classifier trained on cluster labels instead.",
                "<strong>MLflow / Weights & Biases</strong>: log your preprocessing config, algorithm, K value, and all metrics for every experiment. This makes it easy to roll back to a previous version if a new model underperforms.",
                "<strong>Monitoring</strong>: track the distribution of records per cluster over time. A cluster that grows from 10% to 50% of records may indicate that cluster is no longer meaningful.",
            ],
        ),
        (
            "Interpreting and Communicating Clusters",
            "The hardest part of unsupervised learning is explaining results to stakeholders who "
            "don't know what a Silhouette Score is. Good cluster interpretation requires translating "
            "statistical structure into actionable business narrative.",
            [
                "<strong>Name your clusters</strong>: use the mean values per cluster to assign human labels. 'Cluster 2' is useless; 'High-Income Occasional Buyers' is actionable.",
                "<strong>Feature importance</strong>: identify which features differ most between clusters using ANOVA F-statistic or simple mean comparisons. Those are the features that 'define' each cluster.",
                "<strong>Radar/spider plots</strong>: visualise cluster profiles as overlapping polygons — excellent for stakeholder presentations because they show all features simultaneously.",
                "<strong>T-tests / statistical validation</strong>: confirm that the mean differences between clusters on key features are statistically significant, not just noise.",
                "<strong>Business validation</strong>: the ultimate test. Show cluster profiles to domain experts. If they say 'these segments make complete sense given what we know about our customers', you have good clusters.",
            ],
        ),
    ]
    for title, intro, bullets in topics:
        with st.expander(f"🔬 {title}"):
            st.markdown(
                f'<div style="font-size:0.88rem;color:#8b90b0;line-height:1.85;margin-bottom:0.9rem">{intro}</div>',
                unsafe_allow_html=True,
            )
            for b in bullets:
                st.markdown(
                    f'<div style="font-size:0.85rem;color:#8b90b0;line-height:1.9;padding:0.2rem 0 0.2rem 0.5rem;'
                    f'border-left:2px solid #1e2035;margin:0.3rem 0">• {b}</div>',
                    unsafe_allow_html=True,
                )


def _glossary() -> None:
    terms = {
        # Core concepts
        "Clustering":           "Unsupervised grouping of data points by similarity. No labels required — the algorithm discovers groups automatically.",
        "Unsupervised Learning": "ML where the algorithm finds patterns without any pre-labelled examples. Clustering is the primary unsupervised task.",
        "Feature":              "A single measurable property of your data (a column). Also called a variable, attribute, or dimension.",
        "Centroid":             "The centre point of a cluster — calculated as the mean of all data points assigned to that cluster. Used by KMeans.",
        # Algorithms
        "KMeans":               "A clustering algorithm that partitions data into K groups by iteratively assigning points to the nearest centroid and updating centroid positions.",
        "DBSCAN":               "Density-Based Spatial Clustering. Groups points in high-density regions and labels low-density points as noise (-1). Does not require specifying K.",
        "Agglomerative":        "Hierarchical bottom-up clustering. Starts with each point as its own cluster and merges the closest pairs step by step.",
        "Spectral Clustering":  "Uses graph theory and eigenvalue decomposition to find clusters. Excellent for non-convex cluster shapes.",
        "Birch":                "Balanced Iterative Reducing and Clustering using Hierarchies. Builds a tree structure for fast, memory-efficient clustering on large datasets.",
        "MeanShift":            "Finds cluster centres by iteratively shifting towards areas of higher density. Automatically discovers K.",
        # Metrics
        "Silhouette Score":     "Measures cluster separation quality. Range: −1 to +1. Above 0.5 is good; above 0.7 is excellent. Measures how similar each point is to its own cluster vs the nearest other cluster.",
        "Davies-Bouldin":       "Lower = better. Average ratio of within-cluster scatter to inter-cluster distance. Penalises clusters that are large relative to their separation.",
        "Calinski-Harabasz":    "Higher = better. Ratio of between-cluster to within-cluster dispersion. Fast to compute. Can be biased towards larger K.",
        "Inertia (WCSS)":       "Within-Cluster Sum of Squares. Total squared distance of each point from its centroid. Lower = tighter clusters. Used in the Elbow method.",
        "Noise Points":         "Points labelled -1 by DBSCAN. They don't belong to any cluster because they are too far from any dense region. Often genuine outliers.",
        # Preprocessing
        "StandardScaler":       "Transforms each feature to mean=0 and standard deviation=1. The recommended default scaler for most clustering algorithms.",
        "MinMaxScaler":         "Scales each feature to the range [0, 1]. Use when you need bounded values. Sensitive to outliers.",
        "RobustScaler":         "Uses median and IQR instead of mean and std. Best choice when your data has outliers you cannot or do not want to remove.",
        "Imputation":           "Filling in missing values in a dataset. Strategies include mean, median, KNN, or dropping rows with any missing value.",
        "One-Hot Encoding":     "Converting a categorical feature (e.g. Gender: Male/Female) into multiple binary (0/1) columns (Gender_Male, Gender_Female).",
        # Dimensionality reduction
        "PCA":                  "Principal Component Analysis. Compresses many features into fewer 'super-features' (components) that retain maximum variance. Used here for 2D visualisation.",
        "t-SNE":                "t-Distributed Stochastic Neighbour Embedding. Non-linear dimensionality reduction for visualisation. Preserves local structure better than PCA but is not suitable for distance calculations.",
        # Methods
        "Elbow Method":         "A technique for choosing K in KMeans. Plot inertia vs K — the 'elbow' (where improvement slows) suggests the optimal K.",
        "Silhouette Sweep":     "Plot Silhouette Score vs K. Choose the K with the highest score. Use alongside the Elbow method.",
        "Dendrogram":           "A tree diagram from hierarchical clustering. The height of each merge indicates how dissimilar those groups are. Cut the tree at a height to get clusters.",
        "Feature Engineering":  "Creating new features by combining or transforming existing columns (e.g. A ÷ B for a ratio, A × B for an interaction).",
        "Low Variance":         "A feature where almost all values are the same. Such features carry little information and may add noise. Variance Threshold is a common filter.",
        "Z-Score":              "Measures how many standard deviations a point is from the mean. |z| > 3 typically indicates an outlier. Formula: z = (x − mean) / std.",
        "IQR":                  "Interquartile Range — the range between the 25th and 75th percentile. A robust measure of spread not affected by extreme values. Used in IQR outlier detection.",
        # Advanced
        "AutoML":               "Automated Machine Learning. Tries many algorithm/hyperparameter combinations automatically and returns the best result. Used here to compare 30+ configurations.",
        "Hyperparameter":       "A setting chosen before training (e.g. K in KMeans, ε in DBSCAN). Unlike model parameters, hyperparameters are not learned from data — you must tune them.",
        "Linkage":              "In Agglomerative clustering, linkage defines how distance between clusters is measured. Ward (minimise variance), complete (max distance), average, or single (min distance).",
        "Cluster Drift":        "When the distribution of incoming data shifts over time, causing previously good cluster assignments to become stale. Monitor by recomputing metrics on fresh data periodically.",
    }

    # ── Search / filter ──
    search = st.text_input("🔍 Search glossary", placeholder="Type a term…", key="glossary_search")
    filtered = {
        k: v for k, v in terms.items()
        if not search or search.lower() in k.lower() or search.lower() in v.lower()
    }

    if not filtered:
        st.info(f"No terms matching '{search}'. Try a shorter search.")
        return

    st.caption(f"Showing {len(filtered)} of {len(terms)} terms")

    c1, c2 = st.columns(2)
    for i, (term, defn) in enumerate(filtered.items()):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(
                f'<div style="background:#111225;border:1px solid #1e2035;border-radius:8px;'
                f'padding:0.9rem 1rem;margin-bottom:0.7rem;">'
                f'<div style="font-family:IBM Plex Mono;font-size:0.75rem;color:#22d3ee;'
                f'margin-bottom:0.3rem;font-weight:500">'
                f'{_safe_html(term)}</div>'
                f'<div style="font-size:0.85rem;color:#8b90b0;line-height:1.6">'
                f'{_safe_html(defn)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
