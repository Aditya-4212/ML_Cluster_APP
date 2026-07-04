# ============================================================
# config/theme.py
# Professional White Theme - Improved Contrast
# ============================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600&display=swap');

:root {
    --bg: #ffffff;
    --bg2: #f8fafc;
    --card: #ffffff;
    --card2: #f8fafc;
    --border: #e2e8f0;
    --border2: #cbd5e1;
    --primary: #2563eb;
    --cyan: #0284c8;
    --violet: #7c3aed;
    --emerald: #10b981;
    --amber: #d97706;
    --rose: #e11d48;
    --text: #0f172a;           /* Main text - darker */
    --muted: #475569;          /* Improved from too light gray */
    --dim: #64748b;            /* Secondary text */
    --font-head: 'Syne', sans-serif;
    --font-mono: 'IBM Plex Mono', monospace;
    --font-body: 'Outfit', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

.main .block-container { 
    padding: 1.5rem 2rem 3rem; 
    max-width: 1500px; 
}

/* ── PIPELINE STEPPER ── */
.pipeline-nav {
    display: flex; align-items: center; gap: 0;
    background: var(--bg2); 
    border: 1px solid var(--border);
    border-radius: 10px; 
    overflow: hidden; 
    margin-bottom: 2rem;
}
.step-btn {
    flex: 1; padding: 0.85rem 0.5rem; text-align: center;
    border-right: 1px solid var(--border);
    transition: all 0.2s; position: relative;
}
.step-btn:last-child { border-right: none; }
.step-btn.done .step-label { color: var(--emerald); }
.step-btn.active {
    background: linear-gradient(135deg, rgba(37,99,235,0.08), rgba(124,58,237,0.06));
    border-bottom: 2px solid var(--primary);
}
.step-label {
    font-family: var(--font-mono); font-size: 0.7rem;
    letter-spacing: 0.08em; text-transform: uppercase; 
    color: var(--muted);
}
.step-btn.active .step-label { color: var(--primary); }

/* ── CARDS ── */
.card {
    background: var(--card); 
    border: 1px solid var(--border);
    border-radius: 12px; 
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem; 
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
}

/* ── TEXT IMPROVEMENTS ── */
.card-desc, .insight, .learn-body, .step-label {
    color: var(--muted) !important;
}

/* ── ALERT BOXES ── */
.insight strong { color: var(--primary); }
.warn-box strong { color: var(--amber); }
.success-box strong { color: var(--emerald); }

/* ── HERO ── */
.hero-title {
    font-family: var(--font-head); 
    font-size: 2.4rem; 
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary) 0%, var(--violet) 70%);
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent;
}

/* ── BUTTONS ── */
.stButton > button {
    font-family: var(--font-mono) !important; 
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important; 
    text-transform: uppercase !important;
    border-radius: 8px !important; 
    padding: 0.6rem 1.5rem !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--primary), #1e40af) !important;
    color: white !important;
}

/* ── TABS & INPUTS ── */
.stTabs [aria-selected="true"] { 
    color: var(--primary) !important; 
    border-bottom: 2px solid var(--primary) !important; 
}

.stTextInput > div > div > input {
    color: var(--text) !important;
}

/* ── METRICS & MISC ── */
div[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
}

.stDataFrame, .stExpander {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploadDropzone"] {
    background: var(--bg2) !important; 
    border: 2px dashed var(--border2) !important; 
    border-radius: 12px !important; 
}
[data-testid="stFileUploadDropzone"]:hover { 
    border-color: var(--primary) !important; 
}

hr { border-color: var(--border) !important; }
</style>
"""

def inject_css() -> None:
    """Call once at app startup to apply the professional white theme."""
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)
