# ============================================================
# config/theme.py
# All custom CSS injected via st.markdown(CSS, unsafe_allow_html=True)
# ============================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&family=Outfit:wght@300;400;500&display=swap');

:root {
    --bg:      #ffffff;
    --bg2:     #f8fafc;
    --card:    #ffffff;
    --card2:   #f5f5f5;

    --border:  #d1d5db;
    --border2: #cbd5e1;

    --cyan:    #2563eb;
    --violet:  #7c3aed;
    --emerald: #16a34a;
    --amber:   #d97706;
    --rose:    #dc2626;

    --text:    #111827;
    --muted:   #6b7280;
    --dim:     #9ca3af;

    --font-head: 'Syne', sans-serif;
    --font-mono: 'IBM Plex Mono', monospace;
    --font-body: 'Outfit', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background: #ffffff !important;
    color: #111827 !important;
}
.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1500px; }

/* ── PIPELINE STEPPER (now clickable) ── */
.pipeline-nav {
    display: flex; align-items: center; gap: 0;
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 10px; overflow: hidden; margin-bottom: 2rem;
}
.step-btn {
    flex: 1; padding: 0.85rem 0.5rem; text-align: center;
    border-right: 1px solid var(--border);
    transition: all 0.2s; position: relative;
}
.step-btn:last-child { border-right: none; }
.step-btn.done {
    background: rgba(52,211,153,0.05);
}
.step-btn.done .step-label { color: var(--emerald); }
.step-btn.done .step-check { display: inline; }
.step-check { display: none; font-size: 0.6rem; }
.step-btn.active {
    background: linear-gradient(135deg, rgba(34,211,238,0.12), rgba(167,139,250,0.08));
    border-bottom: 2px solid var(--cyan);
}
.step-icon  { font-size: 1.3rem; display: block; margin-bottom: 0.2rem; }
.step-label {
    font-family: var(--font-mono); font-size: 0.7rem;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted);
}
.step-btn.active .step-label { color: var(--cyan); }
.step-num  { position: absolute; top: 6px; left: 8px;
    font-family: var(--font-mono); font-size: 0.6rem; color: var(--dim); }

/* ── CARDS ── */
.card {
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    transition: all 0.25s ease;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.10);
}

.card::after {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: #2563eb;
}

.card-title {
    font-family: var(--font-head);
    font-size: 1rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-desc {
    font-size: 0.9rem;
    color: #6b7280;
    line-height: 1.7;
    margin-bottom: 0.8rem;
}

/* ── ALERT / INSIGHT BOXES ── */
.insight {
    background: linear-gradient(135deg, rgba(34,211,238,0.05), rgba(167,139,250,0.05));
    border: 1px solid rgba(34,211,238,0.2); border-left: 3px solid var(--cyan);
    border-radius: 6px; padding: 0.9rem 1.1rem; margin: 0.8rem 0;
    font-size: 0.85rem; color: var(--muted); line-height: 1.7;
}
.insight strong { color: var(--cyan); }

.warn-box {
    background: rgba(251,191,36,0.05); border: 1px solid rgba(251,191,36,0.2);
    border-left: 3px solid var(--amber); border-radius: 6px;
    padding: 0.8rem 1rem; margin: 0.6rem 0; font-size: 0.85rem; color: var(--muted);
}
.warn-box strong { color: var(--amber); }

.success-box {
    background: rgba(52,211,153,0.05); border: 1px solid rgba(52,211,153,0.2);
    border-left: 3px solid var(--emerald); border-radius: 6px;
    padding: 0.8rem 1rem; margin: 0.6rem 0; font-size: 0.85rem; color: var(--muted);
}
.success-box strong { color: var(--emerald); }

/* ── LEARN BOX ── */
.learn-box {
    background: rgba(167,139,250,0.06); border: 1px solid rgba(167,139,250,0.2);
    border-radius: 8px; padding: 1rem 1.2rem; margin: 1rem 0;
}
.learn-title {
    font-family: var(--font-mono); font-size: 0.7rem;
    letter-spacing: 0.15em; text-transform: uppercase; color: var(--violet); margin-bottom: 0.5rem;
}
.learn-body { font-size: 0.85rem; color: var(--muted); line-height: 1.8; }
.learn-body strong { color: var(--text); }

/* ── RECOVERY BOX ── */
.recovery-box {
    background: rgba(251,113,133,0.04); border: 1px solid rgba(251,113,133,0.18);
    border-left: 3px solid var(--rose); border-radius: 8px;
    padding: 1rem 1.2rem; margin: 1.5rem 0;
}
.recovery-title {
    font-family: var(--font-mono); font-size: 0.65rem;
    letter-spacing: 0.15em; text-transform: uppercase; color: var(--rose); margin-bottom: 0.6rem;
}

/* ── METRIC TILE ── */
.metric-tile {
    background: var(--card2); border: 1px solid var(--border2);
    border-radius: 8px; padding: 1rem; text-align: center;
}
.metric-tile .val { font-family: var(--font-mono); font-size: 1.6rem; font-weight: 600; display: block; }
.metric-tile .lbl {
    font-family: var(--font-mono); font-size: 0.7rem;
    letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted);
    margin-top: 0.3rem; display: block;
}

/* ── SECTION DIVIDER ── */
.sec {
    font-family: var(--font-mono); font-size: 0.7rem;
    letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted);
    border-bottom: 1px solid var(--border); padding-bottom: 0.4rem;
    margin: 1.8rem 0 1.2rem; display: flex; align-items: center; gap: 0.5rem;
}
.sec::before { content: ''; width: 3px; height: 3px; border-radius: 50%; background: var(--cyan); flex-shrink: 0; }

/* ── HERO ── */
.hero { padding: 1.5rem 0 1rem; margin-bottom: 1.5rem; }
.hero-title {
    font-family: var(--font-head);
    font-size: 2.4rem;
    font-weight: 800;
    color: #1e3a8a;
    margin: 0;
    line-height: 1.1;
}
.hero-sub { font-size: 0.9rem; color: var(--muted); margin-top: 0.5rem; font-family: var(--font-mono); }

/* ── SIDEBAR ── */
.stSidebar {
    background: #f8fafc !important;
    border-right: 1px solid #d1d5db !important;
}
.stSidebar label { font-family: var(--font-mono) !important; font-size: 0.75rem !important; color: var(--muted) !important; }

/* ── BUTTONS ── */

.stButton > button {
    font-family: var(--font-body) !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;

    border-radius: 10px !important;
    padding: 0.65rem 1.4rem !important;

    transition: all 0.25s ease !important;
}

/* Primary Button */

.stButton > button[kind="primary"] {
    background: #2563eb !important;
    color: white !important;
    border: none !important;
}

.stButton > button[kind="primary"]:hover {
    background: #1d4ed8 !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(37,99,235,0.30) !important;
}

/* Secondary Button */

.stButton > button[kind="secondary"] {
    background: white !important;
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
}

.stButton > button[kind="secondary"]:hover {
    background: #f8fafc !important;
    border-color: #2563eb !important;
    color: #2563eb !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] { background: var(--bg2) !important; border-bottom: 1px solid var(--border) !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-mono) !important; font-size: 0.7rem !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important; color: var(--muted) !important;
    padding: 0.75rem 1.2rem !important;
}
.stTabs [aria-selected="true"] { color: var(--cyan) !important; border-bottom: 2px solid var(--cyan) !important; }

/* ── GLOSSARY SEARCH ── */
.stTextInput > div > div > input {
    background: var(--card) !important; border: 1px solid var(--border2) !important;
    color: var(--text) !important; font-family: var(--font-mono) !important; font-size: 0.82rem !important;
    border-radius: 6px !important;
}

/* ── MISC ── */
.stDataFrame    { border: 1px solid var(--border) !important; border-radius: 8px !important; overflow: hidden !important; }
.stExpander     { border: 1px solid var(--border) !important; border-radius: 8px !important; background: var(--card) !important; }
div[data-testid="stMetricValue"] { font-family: var(--font-mono) !important; }
div[data-testid="stMetricLabel"] { font-family: var(--font-mono) !important; font-size: 0.7rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }
[data-testid="stFileUploadDropzone"] { background: var(--card) !important; border: 1px dashed var(--border2) !important; border-radius: 10px !important; }
[data-testid="stFileUploadDropzone"]:hover { border-color: var(--cyan) !important; }
hr { border-color: var(--border) !important; }
</style>
"""


def inject_css() -> None:
    """Call once at app startup to apply the dark theme."""
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)
