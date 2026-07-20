# ============================================================
# components/ui.py
# Reusable UI helper components used across pipeline steps
# ============================================================

import html as _html
import streamlit as st
from config.settings import PIPELINE_STEPS, SESSION_DEFAULTS, STEP_RESET_KEYS


# ── Section divider ─────────────────────────────────────────

def section(title: str) -> None:
    safe = _html.escape(str(title))
    st.markdown(f'<div class="sec">{safe}</div>', unsafe_allow_html=True)


# ── Learn / info / warn / success boxes ─────────────────────

def explain(title: str, body: str, kind: str = "learn") -> None:
    """
    Render a contextual info box.
    kind: 'learn' (violet) | 'warn' (amber) | 'success' (emerald) | 'insight' (cyan)
    """
    cls_map = {
        "learn":   "learn-box",
        "warn":    "warn-box",
        "success": "success-box",
        "insight": "insight",
    }
    cls = cls_map.get(kind, "insight")
    safe_title = _html.escape(str(title))

    if cls == "learn-box":
        st.markdown(
            f'<div class="{cls}">'
            f'<div class="learn-title">{safe_title}</div>'
            f'<div class="learn-body">{body}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="{cls}"><strong>{safe_title}</strong> {body}</div>',
            unsafe_allow_html=True,
        )


# ── Metric strip ─────────────────────────────────────────────

def metric_strip(metrics: dict, model_name: str) -> None:
    display = {k: v for k, v in metrics.items() if k not in ("Clusters",)}
    n_clusters = metrics.get("Clusters", "?")

    st.markdown(
        f'<div class="sec">Model: {_html.escape(str(model_name))} '
        f'· {n_clusters} cluster(s)</div>',
        unsafe_allow_html=True,
    )

    keys = list(display.keys())
    if not keys:
        return

    cols = st.columns(len(keys))
    for col, key in zip(cols, keys):
        col.metric(label=key, value=display[key])


# ── Progress tracker ──────────────────────────────────────────

def progress_tracker() -> None:
    flags = [
        ("preprocessing_done", "Data Cleaned"),
        ("eda_done",           "EDA Complete"),
        ("engineering_done",   "Features Locked"),
        ("clustering_done",    "Model Trained"),
    ]
    done = [label for key, label in flags if st.session_state.get(key)]
    total = len(flags)
    count = len(done)

    bar_pct = int(count / total * 100)
    labels_str = " · ".join(done) if done else "Not started"

    st.markdown(
        f'<div class="insight">'
        f'<strong>Pipeline Progress: {count}/{total}</strong><br>'
        f'<div style="background:#1e2035;border-radius:4px;height:4px;margin:0.5rem 0;">'
        f'<div style="background:#22d3ee;height:4px;border-radius:4px;width:{bar_pct}%;"></div>'
        f'</div>'
        f'<span style="font-size:0.8rem;">{_html.escape(labels_str)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Pipeline stepper (now clickable) ─────────────────────────
'''

def pipeline_stepper() -> None:
    """
    Render the horizontal pipeline step indicator.
    Each step is now a real Streamlit button for navigation.
    Completed steps are highlighted in emerald with a ✓ mark.
    """
    current = st.session_state.get("step", 0)

    # Determine which steps are "done"
    done_flags = {
        0: st.session_state.get("df_raw") is not None,
        1: bool(st.session_state.get("eda_done")),
        2: bool(st.session_state.get("preprocessing_done")),
        3: bool(st.session_state.get("engineering_done")),
        4: bool(st.session_state.get("clustering_done")),
        5: bool(st.session_state.get("clustering_done")),
        6: False,
    }

    cols = st.columns(len(PIPELINE_STEPS))
    for i, ((icon, label), col) in enumerate(zip(PIPELINE_STEPS, cols)):
        is_active = i == current
        is_done   = done_flags.get(i, False) and not is_active

        if is_done:
            indicator = "✓"
            border_color = "#34d399"
            label_color  = "#34d399"
        elif is_active:
            indicator = "▶"
            border_color = "#22d3ee"
            label_color  = "#22d3ee"
        else:
            indicator = str(i + 1)
            border_color = "#1e2035"
            label_color  = "#8b90b0"

        bg = "rgba(34,211,238,0.07)" if is_active else ("rgba(52,211,153,0.04)" if is_done else "transparent")

        with col:
            if st.button(
                f"{icon}\n{indicator} {label}",
                key=f"stepper_nav_{i}",
                use_container_width=True,
                help=f"Go to: {label}",
            ):
                st.session_state.step = i
                st.rerun()
'''

def pipeline_stepper():
    return

# ── Hero header ───────────────────────────────────────────────

def hero(title: str = "ML Clustering Studio", subtitle: str = "End-to-End Production") -> None:
    st.markdown(
        f'<div class="hero">'
        f'<div class="hero-title">{_html.escape(title)}</div>'
        f'<div class="hero-sub">{_html.escape(subtitle)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Recovery panel ────────────────────────────────────────────

def recovery_panel(current_step: int) -> None:
    """
    Shows a collapsible panel at the bottom of each step offering:
    - Restart from current step (clears this step + later steps only)
    - Restart everything (with two-step confirmation)
    """
    step_names = ["Load Data", "EDA", "Cleaning", "Feature Engineering",
                  "Clustering", "Results", "Learn"]
    current_name = step_names[current_step] if current_step < len(step_names) else "Current Step"

    with st.expander("⚙️ Recovery Options — stuck or want to redo a step?"):
        st.markdown(
            '<div class="recovery-title">Recovery Options</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Use these if you made a mistake in a previous step or want to "
            "try different settings without losing earlier work."
        )

        col1, col2 = st.columns(2)

        # ── Restart from current step ──
        with col1:
            st.markdown(f"**Restart from: {current_name}**")
            st.caption("Clears this step's output and everything after it. Your earlier work is preserved.")
            if st.button(
                f"↺ Restart from {current_name}",
                key=f"restart_step_{current_step}",
                use_container_width=True,
            ):
                _reset_from_step(current_step)
                st.success(f"✓ Reset from step {current_step + 1}. You can now redo from here.")
                st.rerun()

        # ── Restart everything (two-step confirm) ──
        with col2:
            st.markdown("**Restart Everything**")
            st.caption("Wipes all data, results, and choices. Use to start a completely new analysis.")

            if st.session_state.get("confirm_reset"):
                st.warning("⚠ Are you sure? All data and results will be cleared.")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Yes, clear all", key="confirm_reset_yes", use_container_width=True):
                        for k, v in SESSION_DEFAULTS.items():
                            st.session_state[k] = v
                        st.session_state.step = 0
                        st.rerun()
                with c2:
                    if st.button("Cancel", key="confirm_reset_no", use_container_width=True):
                        st.session_state.confirm_reset = False
                        st.rerun()
            else:
                if st.button("🗑 Restart Everything", key="restart_all", use_container_width=True):
                    st.session_state.confirm_reset = True
                    st.rerun()


def _reset_from_step(step: int) -> None:
    """
    Clear session state keys owned by `step` and all subsequent steps.
    Earlier steps' data is preserved.
    """
    from config.settings import SESSION_DEFAULTS
    keys_to_clear = STEP_RESET_KEYS.get(step, [])
    for key in keys_to_clear:
        default = SESSION_DEFAULTS.get(key)
        st.session_state[key] = default


# ── Navigation row (Back + Proceed buttons) ───────────────────

def nav_buttons(
    proceed_label: str = "Proceed →",
    proceed_step: int = None,
    back_step: int = None,
    back_label: str = None,
    proceed_key: str = "nav_proceed",
) -> bool:
    """
    Render a consistent Back / Proceed button row at the bottom of each step.
    Returns True if the Proceed button was clicked (for inline actions before advancing).
    """
    step_names = ["Load Data", "EDA", "Cleaning", "Feature Engineering",
                  "Clustering", "Results", "Learn"]

    cols = st.columns([1, 3, 1]) if back_step is not None else [None, st.container(), None]

    if back_step is not None:
        _back_name = back_label or (step_names[back_step] if back_step < len(step_names) else "Back")
        with cols[0]:
            if st.button(f"← {_back_name}", key=f"nav_back_{back_step}", use_container_width=True):
                st.session_state.step = back_step
                st.rerun()

    clicked = False
    with cols[2] if back_step is not None else cols[1]:
        if st.button(proceed_label, type="primary", key=proceed_key, use_container_width=True):
            if proceed_step is not None:
                st.session_state.step = proceed_step
                st.rerun()
            clicked = True

    return clicked
