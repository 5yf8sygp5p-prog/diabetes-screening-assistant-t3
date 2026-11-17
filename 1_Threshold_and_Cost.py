"""
pages/1_Threshold_and_Cost.py — Threshold and Cost Simulator (4 Languages)
==========================================================================
"""

import altair as alt
import pandas as pd
import streamlit as st
from i18n import inject_friendly_css, load_artifacts, render_sidebar_and_get_lang

st.set_page_config(
    page_title="Threshold and Cost | Diabetes Screening Assistant",
    page_icon="⚖️",
    layout="wide",
)

inject_friendly_css()
lang_code, t = render_sidebar_and_get_lang()
_, _, meta = load_artifacts()

curve_list = meta["threshold_curve"]
default_th = float(meta["cost_assumptions"]["chosen_threshold"])

if "active_threshold" not in st.session_state:
    st.session_state["active_threshold"] = default_th
if "cost_ratio" not in st.session_state:
    st.session_state["cost_ratio"] = 5.0

st.title(t["tc_title"])
st.info(t["tc_banner"])

col_left, col_right = st.columns([1.05, 1.95], gap="large")

with col_left:
    cost_ratio = st.slider(
        t["tc_slider_label"],
        min_value=1.0,
        max_value=15.0,
        value=float(st.session_state["cost_ratio"]),
        step=0.5,
    )
    st.session_state["cost_ratio"] = cost_ratio

    # Compute expected validation cost per person for each threshold on the grid
    best_pt = min(
        curve_list,
        key=lambda row: row["val_fp_rate_per_person"] * 1.0 + row["val_fn_rate_per_person"] * cost_ratio,
    )
    opt_th = float(best_pt["threshold"])

    manual_mode = st.checkbox(t["tc_manual_check"], value=False)
    if manual_mode:
        chosen_th = st.slider(
            t["tc_manual_slider"],
            min_value=0.02,
            max_value=0.80,
            value=opt_th,
            step=0.02,
        )
    else:
        chosen_th = opt_th

    st.metric(label=t["tc_opt_metric"], value=f"{chosen_th:.2f}")
    st.caption(t["tc_default_note"].format(th=f"{default_th:.2f}"))

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if st.button(t["tc_btn_use"], type="primary", use_container_width=True):
            st.session_state["active_threshold"] = chosen_th
            st.success(t["tc_applied_msg"].format(th=f"{chosen_th:.2f}"))
    with b_col2:
        if st.button(t["tc_btn_reset"], use_container_width=True):
            st.session_state["cost_ratio"] = 5.0
            st.session_state["active_threshold"] = default_th
            st.rerun()

# Find matching row in threshold_curve
current_pt = min(curve_list, key=lambda r: abs(r["threshold"] - chosen_th))
p1000 = current_pt["per_1000"]

with col_right:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t["tc_kpi_recall"], f"{current_pt['test_recall']:.1%}")
    m2.metric(t["tc_kpi_precision"], f"{current_pt['test_precision']:.1%}")
    m3.metric(t["tc_kpi_fpr"], f"{current_pt['test_fpr']:.1%}")
    m4.metric(t["tc_kpi_flagged"], f"{current_pt['test_flagged']:.1%}")

    st.markdown(f"**{t['tc_per1000_title']}**")
    table_1000_html = (
        "<table class='clean-table'>"
        "<thead>"
        f"<tr><th></th><th style='text-align:right;'>{t['tc_col_flag']}</th><th style='text-align:right;'>{t['tc_col_noflag']}</th></tr>"
        "</thead>"
        "<tbody>"
        f"<tr><td>{t['tc_row_pos']}</td><td style='text-align:right;'><b>{p1000['tp']}</b></td><td style='text-align:right;'>{p1000['fn']}</td></tr>"
        f"<tr><td>{t['tc_row_neg']}</td><td style='text-align:right;'>{p1000['fp']}</td><td style='text-align:right;'><b>{p1000['tn']}</b></td></tr>"
        "</tbody>"
        "</table>"
    )
    st.markdown(table_1000_html, unsafe_allow_html=True)
    st.caption(t["tc_per1000_caption"].format(fn=p1000["fn"], fp=p1000["fp"]))

st.markdown("---")

# Build Bottom 2 Interactive Charts (Expected Cost vs Threshold & Recall/Precision/Flagged vs Threshold)
chart_df = pd.DataFrame(curve_list)
chart_df["expected_cost"] = (
    chart_df["val_fp_rate_per_person"] * 1.0 + chart_df["val_fn_rate_per_person"] * cost_ratio
)

c_left, c_right = st.columns(2, gap="large")

with c_left:
    st.markdown(f"<h4 style='text-align:center; margin-bottom:4px;'>{t['tc_chart1_title']}</h4>", unsafe_allow_html=True)
    cost_line = (
        alt.Chart(chart_df)
        .mark_line(color="#2b6cb0", strokeWidth=2.5)
        .encode(
            x=alt.X("threshold:Q", title="Threshold", scale=alt.Scale(domain=[0.0, 0.85])),
            y=alt.Y("expected_cost:Q", title=t["tc_chart1_y"], scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("threshold:Q", format=".2f", title="Threshold"),
                alt.Tooltip("expected_cost:Q", format=".3f", title="Assumed Cost / Person"),
            ],
        )
    )
    vline_cost = (
        alt.Chart(pd.DataFrame({"threshold": [chosen_th]}))
        .mark_rule(color="#d95f02", strokeDash=[6, 4], strokeWidth=2)
        .encode(x="threshold:Q")
    )
    st.altair_chart((cost_line + vline_cost).properties(height=300), use_container_width=True)

with c_right:
    st.markdown(f"<h4 style='text-align:center; margin-bottom:4px;'>{t['tc_chart2_title']}</h4>", unsafe_allow_html=True)
    melted = chart_df.melt(
        id_vars=["threshold"],
        value_vars=["test_recall", "test_precision", "test_flagged"],
        var_name="Metric",
        value_name="Rate",
    )
    metric_label_map = {
        "test_recall": "Recall",
        "test_precision": "Precision",
        "test_flagged": "Share flagged",
    }
    melted["Metric"] = melted["Metric"].map(metric_label_map)

    rates_chart = (
        alt.Chart(melted)
        .mark_line(strokeWidth=2.4)
        .encode(
            x=alt.X("threshold:Q", title="Threshold", scale=alt.Scale(domain=[0.0, 0.85])),
            y=alt.Y("Rate:Q", title="", scale=alt.Scale(domain=[0.0, 1.0])),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(
                    domain=["Recall", "Precision", "Share flagged"],
                    range=["#d95f02", "#2b6cb0", "#64748b"],
                ),
                legend=alt.Legend(orient="right", title=None),
            ),
            strokeDash=alt.StrokeDash(
                "Metric:N",
                scale=alt.Scale(
                    domain=["Recall", "Precision", "Share flagged"],
                    range=[[1, 0], [1, 0], [3, 3]],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("threshold:Q", format=".2f"),
                alt.Tooltip("Metric:N"),
                alt.Tooltip("Rate:Q", format=".1%"),
            ],
        )
    )
    vline_rates = (
        alt.Chart(pd.DataFrame({"threshold": [chosen_th]}))
        .mark_rule(color="#1e293b", strokeDash=[6, 4], strokeWidth=2)
        .encode(x="threshold:Q")
    )
    st.altair_chart((rates_chart + vline_rates).properties(height=300), use_container_width=True)

st.markdown(t["tc_how_to_read"])
