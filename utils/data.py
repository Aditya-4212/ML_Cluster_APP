# ============================================================
# utils/data.py
# All data loading, cleaning, preprocessing, feature engineering
# ============================================================

from __future__ import annotations

import html as _html

import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import zscore  # requires scipy in requirements.txt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

_MAX_UPLOAD_MB = 50  # reject files larger than this


# ── Column helpers ──────────────────────────────────────────

def get_numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=np.number).columns.tolist()


def get_cat_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(exclude=np.number).columns.tolist()


# ── Load ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_csv(f) -> pd.DataFrame:
    """
    Parse an uploaded CSV file.
    Validates size and raises ValueError on bad content.
    """
    # Size check (Streamlit exposes .size in bytes)
    size_mb = getattr(f, "size", 0) / (1024 * 1024)
    if size_mb > _MAX_UPLOAD_MB:
        raise ValueError(f"File too large ({size_mb:.1f} MB). Limit is {_MAX_UPLOAD_MB} MB.")

    try:
        df = pd.read_csv(f)
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded CSV is empty.")
    if df.shape[1] < 2:
        raise ValueError("The CSV must have at least 2 columns.")

    return df


# ── Clean ───────────────────────────────────────────────────

def apply_imputation(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """
    Fill missing values.
    strategy: 'Mean' | 'Median' | 'KNN' | 'Drop Rows'
    """
    df = df.copy()

    if strategy == "Drop Rows":
        return df.dropna().reset_index(drop=True)

    num_cols = get_numeric_cols(df)
    cat_cols = get_cat_cols(df)

    for col in num_cols:
        if df[col].isnull().any():
            if strategy == "Mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "Median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "KNN":
                vals = df[[col]].values
                df[col] = KNNImputer(n_neighbors=5).fit_transform(vals).ravel()

    for col in cat_cols:
        df[col] = df[col].fillna("Missing")

    return df


def apply_outlier_removal(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    method: 'None' | 'Z-Score (|z| > 3)' | 'IQR (1.5×IQR)' | 'Clip to 99th Percentile'

    IQR removal builds a single combined mask (not per-column sequential),
    preventing excessive row loss on wide datasets.
    """
    df = df.copy()
    num_cols = get_numeric_cols(df)
    if not num_cols:
        return df

    if method == "Z-Score (|z| > 3)":
        z = np.abs(zscore(df[num_cols], nan_policy="omit"))
        mask = (z < 3).all(axis=1)
        df = df[mask].reset_index(drop=True)

    elif method == "IQR (1.5×IQR)":
        # Build a combined boolean mask in one pass — avoids compounding row loss
        mask = pd.Series(True, index=df.index)
        for col in num_cols:
            q25 = df[col].quantile(0.25)
            q75 = df[col].quantile(0.75)
            iqr = q75 - q25
            mask &= (df[col] >= q25 - 1.5 * iqr) & (df[col] <= q75 + 1.5 * iqr)
        df = df[mask].reset_index(drop=True)

    elif method == "Clip to 99th Percentile":
        for col in num_cols:
            lo = df[col].quantile(0.01)
            hi = df[col].quantile(0.99)
            df[col] = df[col].clip(lo, hi)

    return df


# ── Preprocess (scale + encode) ─────────────────────────────

def preprocess_X(df: pd.DataFrame, scaler_name: str, imputer_name: str) -> np.ndarray:
    """
    Returns a 2-D numpy array ready for clustering.
    Applies imputation → scaling on numeric cols, one-hot on categoricals.
    """
    num = get_numeric_cols(df)
    cat = get_cat_cols(df)

    scaler_map: dict = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler":   MinMaxScaler(),
        "RobustScaler":   RobustScaler(),
    }
    imp_map: dict = {
        "Mean":   SimpleImputer(strategy="mean"),
        "Median": SimpleImputer(strategy="median"),
        "KNN":    KNNImputer(n_neighbors=5),
    }

    num_pipe = Pipeline([
        ("imp",    imp_map.get(imputer_name, SimpleImputer(strategy="mean"))),
        ("scaler", scaler_map.get(scaler_name, StandardScaler())),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("enc", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    ])

    transformers = []
    if num:
        transformers.append(("num", num_pipe, num))
    if cat:
        transformers.append(("cat", cat_pipe, cat))

    if not transformers:
        return np.zeros((len(df), 1))

    ct = ColumnTransformer(transformers)
    return ct.fit_transform(df)


# ── Dimensionality reduction (for visualisation only) ───────

@st.cache_data(show_spinner=False)
def reduce_2d(X: np.ndarray, method: str = "PCA") -> np.ndarray:
    """
    Reduce to 2 components for scatter plot visualisation.
    Result is cached so t-SNE is not recomputed on every rerun.
    """
    if X.shape[1] == 1:
        X = np.hstack([X, np.zeros((X.shape[0], 1))])

    if method == "t-SNE":
        from sklearn.manifold import TSNE
        perp = min(30, max(5, X.shape[0] // 3))
        return TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(X)

    return PCA(n_components=2, random_state=42).fit_transform(X)


# ── Feature engineering helpers ─────────────────────────────

def get_low_variance_features(df: pd.DataFrame, threshold: float = 0.01) -> list[str]:
    num_cols = get_numeric_cols(df)
    if not num_cols:
        return []
    vt = VarianceThreshold(threshold=threshold)
    try:
        vt.fit(df[num_cols].fillna(0).values)
        variances = dict(zip(num_cols, vt.variances_))
        return [c for c, v in variances.items() if v < threshold]
    except Exception:
        return []


def get_pca_explained(df: pd.DataFrame):
    """Returns (individual%, cumulative%) arrays for PCA variance chart."""
    num_cols = get_numeric_cols(df)
    if len(num_cols) < 2:
        return None, None
    arr = StandardScaler().fit_transform(df[num_cols].fillna(0))
    pca = PCA().fit(arr)
    individual = pca.explained_variance_ratio_ * 100
    cumulative  = np.cumsum(individual)
    return individual, cumulative


def get_high_corr_pairs(df: pd.DataFrame, threshold: float = 0.8) -> list[tuple]:
    """Returns list of (colA, colB, r) tuples where |r| > threshold."""
    num_cols = get_numeric_cols(df)
    if len(num_cols) < 2:
        return []
    corr = df[num_cols].corr()
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if abs(v) > threshold:
                pairs.append((cols[i], cols[j], round(v, 3)))
    return pairs


def get_auto_remove_cols(df: pd.DataFrame) -> list[str]:
    """Suggest columns that are unique-per-row (IDs) or fully constant."""
    return [
        c for c in df.columns
        if df[c].nunique() == len(df) or df[c].nunique() <= 1
    ]
