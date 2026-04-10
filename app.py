# ============================================================
# app.py — v2
# Entry point — page config, CSS injection, session init, router
# Run with: streamlit run app.py
# ============================================================

import streamlit as st

from config.settings import SESSION_DEFAULTS
from config.theme import inject_css
from components.ui import hero, pipeline_stepper
from pipeline.steps import (
    step_load, step_eda, step_clean, step_features,
    step_cluster, step_results, step_learn,
)

# ── Page config (must be first Streamlit call) ──────────────
st.set_page_config(
    page_title="ML Clustering Studio",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ──────────────────────────────────────────────
inject_css()

# ── Session state init ──────────────────────────────────────
for k, v in SESSION_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🕸️ Navigation")

    step_labels = [
        ("📂 Load Data",          0),
        ("📊 EDA",                1),
        ("🧹 Cleaning",           2),
        ("⚙️ Feature Engineering", 3),
        ("🤖 Clustering",         4),
        ("📈 Results",            5),
        ("🎓 Learn ML",           6),
    ]

    done_flags = {
        0: st.session_state.get("df_raw") is not None,
        1: bool(st.session_state.get("eda_done")),
        2: bool(st.session_state.get("preprocessing_done")),
        3: bool(st.session_state.get("engineering_done")),
        4: bool(st.session_state.get("clustering_done")),
        5: bool(st.session_state.get("clustering_done")),
        6: False,
    }

    for name, idx in step_labels:
        is_active = idx == st.session_state.step
        is_done   = done_flags.get(idx, False) and not is_active
        prefix    = "👉" if is_active else ("✓" if is_done else "•")
        if st.button(
            f"{prefix} {name}",
            key=f"nav_{idx}",
            use_container_width=True,
        ):
            st.session_state.step = idx
            st.rerun()

    st.markdown("---")

    # ── Sidebar file uploader (kept for convenience) ──
    st.caption("Upload a CSV to begin:")
    uploaded_file = st.file_uploader(
        "📂 Upload CSV",
        type=["csv"],
        key="sidebar_uploader",
        help="Max 50 MB. Supports numeric and categorical columns.",
    )

    # ── Pipeline progress (sidebar mini-display) ──
    st.markdown("---")
    st.markdown("**Pipeline Status**")
    status_items = [
        ("df_raw",             "📥 Data Loaded"),
        ("eda_done",           "🔍 EDA Done"),
        ("preprocessing_done", "🧹 Cleaned"),
        ("engineering_done",   "⚙️ Features Set"),
        ("clustering_done",    "🤖 Model Trained"),
    ]
    for key, label in status_items:
        val = st.session_state.get(key)
        done = val is not None if key == "df_raw" else bool(val)
        icon = "🟢" if done else "⚪"
        st.caption(f"{icon} {label}")

# ── Header ───────────────────────────────────────────────────
hero()
pipeline_stepper()

# ── Router ───────────────────────────────────────────────────
_STEP_FUNCTIONS = {
    0: lambda: step_load(uploaded_file),
    1: step_eda,
    2: step_clean,
    3: step_features,
    4: step_cluster,
    5: step_results,
    6: step_learn,
}

step = st.session_state.step
handler = _STEP_FUNCTIONS.get(step)

try:
    if handler:
        handler()
    else:
        st.error(f"Unknown step: {step}")
        st.session_state.step = 0
        st.rerun()
except Exception as e:
    st.error("🚨 Application Error")
    st.exception(e)
    st.markdown(
        '<div class="warn-box"><strong>💡 Recovery:</strong> '
        'Use the Recovery Options at the bottom of any step to restart from where you left off, '
        'or use the sidebar to navigate to a different step.</div>',
        unsafe_allow_html=True,
    )
    if st.button("↩ Return to Step 1 (Load Data)", key="error_recovery"):
        st.session_state.step = 0
        st.rerun()

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.caption("ML Clustering Studio · End-to-End Production Pipeline")
