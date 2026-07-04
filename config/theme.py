# ============================================================
# config/theme.py
# All custom CSS injected via st.markdown(CSS, unsafe_allow_html=True)
# ============================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&family=Outfit:wght@300;400;500&display=swap');

:root {
    --bg: #ffffff;
    --bg2: #f8fafc;
    --card: #ffffff;
    --card2: #f1f5f9;
    --border: #e2e8f0;
    --border2: #cbd5e1;
    --primary: #2563eb;           /* Blue accent */
    --cyan: #0284c8;
    --violet: #7c3aed;
    --emerald: #10b981;
    --amber: #d97706;
    --rose: #e11d48;
    --text: #0f172a;
    --muted: #64748b;
    --dim: #94a3b8;
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
.step-btn.done {
    background: rgba(16,185,129,0.08);
}
.step-btn.done .step-label { color: var(--emerald); }
.step-btn.done .step-check { display: inline; }
.step-check { display: none; font-size: 0.6rem; }
.step-btn.active {
    background: linear-gradient(135deg, rgba(37,99,235,0.08), rgba(124,58,237,0.06));
    border-bottom: 2px solid var(--primary);
}
.step-icon { font-size: 1.3rem; display: block; margin-bottom: 0.2rem; }
.step-label {
    font-family: var(--font-mono); font-size: 0.7rem;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted);
}
.step-btn.active .step-label { color: var(--primary); }
.step-num { 
    position: absolute; top: 6px; left: 8px;
    font-family: var(--font-mono); font-size: 0.6rem; color: var(--dim); 
}

/* ── CARDS ── */
.card {
    background: var(--card); 
    border: 1px solid var(--border);
    border-radius: 12px; 
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem; 
    position: relative; 
    overflow: hidden;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
}
.card::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(37,99,235,0.3), transparent);
}

/* ── ALERT / INSIGHT BOXES ── */
.insight {
    background: linear-gradient(135deg, rgba(37,99,235,0.05), rgba(124,58,237,0.05));
    border: 1px solid rgba(37,99,235,0.15); 
    border-left: 3px solid var(--primary);
    border-radius: 8px; 
    padding: 0.9rem 1.1rem; 
    margin: 0.8rem 0;
    font-size: 0.85rem; 
    color: var(--muted); 
    line-height: 1.7;
}
.insight strong { color: var(--primary); }

.warn-box {
    background: rgba(245,158,11,0.06); 
    border: 1px solid rgba(245,158,11,0.2);
    border-left: 3px solid var(--amber); 
    border-radius: 8px;
    padding: 0.8rem 1rem; 
    margin: 0.6rem 0; 
    font-size: 0.85rem; 
    color: var(--muted);
}
.warn-box strong { color: var(--amber); }

.success-box {
    background: rgba(16,185,129,0.06); 
    border: 1px solid rgba(16,185,129,0.2);
    border-left: 3px solid var(--emerald); 
    border-radius: 8px;
    padding: 0.8rem 1rem; 
    margin: 0.6rem 0; 
    font-size: 0.85rem; 
    color: var(--muted);
}
.success-box strong { color: var(--emerald); }

/* Other boxes remain similar - adjusted for light theme */
.learn-box, .recovery-box {
    border-radius: 8px; 
    padding: 1rem 1.2rem; 
    margin: 1rem 0;
}

/* ── METRIC TILE ── */
.metric-tile {
    background: var(--card2); 
    border: 1px solid var(--border2);
    border-radius: 10px; 
    padding: 1rem; 
    text-align: center;
}

/* ── HERO ── */
.hero { padding: 1.5rem 0 1rem; margin-bottom: 1.5rem; }
.hero-title {
    font-family: var(--font-head); 
    font-size: 2.4rem; 
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary) 0%, var(--violet) 70%);
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent;
    background-clip: text; 
    margin: 0; 
    line-height: 1.1;
}

/* ── BUTTONS ── */
.stButton > button {
    font-family: var(--font-mono) !important; 
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important; 
    text-transform: uppercase !important;
    border-radius: 8px !important; 
    padding: 0.6rem 1.5rem !important; 
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--primary), #1e40af) !important;
    color: white !important; 
    border: none !important; 
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(37,99,235,0.3) !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] { 
    background: var(--bg2) !important; 
    border-bottom: 1px solid var(--border) !important; 
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-mono) !important; 
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important; 
    text-transform: uppercase !important; 
    color: var(--muted) !important;
}
.stTabs [aria-selected="true"] { 
    color: var(--primary) !important; 
    border-bottom: 2px solid var(--primary) !important; 
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploadDropzone"] { 
    background: var(--bg2) !important; 
    border: 2px dashed var(--border2) !important; 
    border-radius: 12px !important; 
}
[data-testid="stFileUploadDropzone"]:hover { 
    border-color: var(--primary) !important; 
}

/* ── MISC ── */
.stDataFrame { 
    border: 1px solid var(--border) !important; 
    border-radius: 10px !important; 
    overflow: hidden !important; 
}
.stExpander { 
    border: 1px solid var(--border) !important; 
    border-radius: 10px !important; 
    background: var(--card) !important; 
}
hr { border-color: var(--border) !important; }
</style>
"""

def inject_css() -> None:
    """Call once at app startup to apply the professional white theme."""
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)
