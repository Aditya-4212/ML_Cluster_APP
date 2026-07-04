from pathlib import Path
from textwrap import dedent
import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Syne:wght@600;700;800&display=swap');

:root {
    --bg: #ffffff;
    --card: #ffffff;
    --border: #e5e7eb;
    --primary: #2563eb;
    --primary-light: #3b82f6;
    --text: #111827;
    --muted: #6b7280;
    --success: #16a34a;
    --warn: #d97706;
    --danger: #dc2626;
    --radius: 14px;
    --shadow: 0 10px 15px -3px rgb(0 0 0 / 0.05), 0 4px 6px -4px rgb(0 0 0 / 0.05);
}

html, body, [data-testid="stAppViewContainer"], .main {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
}

/* Layout */
.block-container {
    max-width: 1480px;
    padding: 2rem 2.5rem 3rem !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #fafbfc !important;
    border-right: 1px solid var(--border) !important;
}
.stSidebar label {
    font-weight: 500 !important;
    color: var(--muted) !important;
}

/* Hero */
.hero {
    padding: 1rem 0 2rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #1e3a8a;
    line-height: 1.1;
}
.hero-sub {
    color: var(--muted);
    font-size: 1.05rem;
}

/* Cards */
.card, .stExpander, [data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow) !important;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
}

/* Metrics */
div[data-testid="stMetricValue"] {
    font-family: 'Outfit', sans-serif;
    font-size: 2rem;
    font-weight: 600;
    color: var(--primary);
}
div[data-testid="stMetricLabel"] {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--muted);
}

/* Buttons */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.2s ease;
}
.stButton > button[kind="primary"] {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--primary-light) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.25) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid var(--border) !important;
    gap: 2rem !important;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 500;
    padding: 0.75rem 0 !important;
}
.stTabs [aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom: 3px solid var(--primary) !important;
}

/* File Uploader */
[data-testid="stFileUploadDropzone"] {
    background: #f8fafc !important;
    border: 2px dashed var(--primary) !important;
    border-radius: 12px !important;
    padding: 2rem !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #1e40af !important;
    background: #f0f9ff !important;
}

/* Tables */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden;
}
.stDataFrame thead th {
    background: #f8fafc !important;
    color: var(--text) !important;
    font-weight: 600;
}

/* Section Headers */
.sec {
    font-weight: 600;
    color: #374151;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.6rem;
    margin: 2rem 0 1.2rem;
    font-size: 1.05rem;
}

/* Insight / Alert Boxes */
.insight, .learn-box, .warn-box, .success-box {
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin: 1rem 0;
}
.insight { border-left: 5px solid var(--primary); }
.learn-box { border-left: 5px solid #7c3aed; }
.warn-box { border-left: 5px solid var(--warn); }
.success-box { border-left: 5px solid var(--success); }

/* Misc */
hr { border-color: var(--border) !important; }
</style>
"""

def inject_css():
    """Inject the professional light theme"""
    st.markdown(CSS, unsafe_allow_html=True)
