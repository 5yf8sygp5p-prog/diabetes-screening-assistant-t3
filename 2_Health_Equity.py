"""
pages/2_Health_Equity.py — Health Equity Check Across Income & Education (4 Languages)
======================================================================================
"""

import altair as alt
import pandas as pd
import streamlit as st
from i18n import inject_friendly_css, load_artifacts, render_sidebar_and_get_lang

st.set_page_config(
    page_title="Health Equity | Diabetes Screening Assistant",
    page_icon="🤝",
    layout="wide",
)

inject_friendly_css()
lang_code, t = render_sidebar_and_get_lang()
_, _, meta = load_artifacts()

st.title(t["he_title"])
st.markdown(t["he_intro"])

r_col1, r_col2 = st.columns([1.3, 1.0])
with r_col1:
    model_opt = st.radio(
        t["he_model_radio"],
        options=["app_model", "full_model"],
        format_func=lambda k: t["he_model_app"] if k == "app_model" else t["he_model_full"],
        index=0,
    )
with r_col2:
    group_opt = st.radio(
        t["he_group_radio"],
        options=["Income", "Education"],
        format_func=lambda k: t["he_group_income"] if k == "Income" else t["he_group_edu"],
        horizontal=True,
        index=0,
    )

he_data = meta["health_equity"]
th_val = float(he_data["threshold"])
rows_list = he_data[model_opt][group_opt]
app_overall = meta["app_model_test_metrics"]

# Build formatted HTML table with 95% confidence intervals
t_rows = ""
for r in rows_list:
    rec_str = f"{r['recall']:.1%} ({r['recall_ci'][0]:.1%}–{r['recall_ci'][1]:.1%})"
    fnr_str = f"{r['fnr']:.1%} ({r['fnr_ci'][0]:.1%}–{r['fnr_ci'][1]:.1%})"
    fpr_str = f"{r['fpr']:.1%} ({r['fpr_ci'][0]:.1%}–{r['fpr_ci'][1]:.1%})"
    prec_str = f"{r['precision']:.1%} ({r['precision_ci'][0]:.1%}–{r['precision_ci'][1]:.1%})"
    t_rows += (
        f"<tr>"
        f"<td>{r['group']}</td>"
        f"<td>{r['n']:,}</td>"
        f"<td>{r['prevalence']:.1%}</td>"
        f"<td>{rec_str}</td>"
        f"<td>{fnr_str}</td>"
        f"<td>{fpr_str}</td>"
        f"<td>{prec_str}</td>"
        f"<td>{r['flagged']:.1%}</td>"
        f"</tr>"
    )

table_html = (
    "<table class='clean-table'>"
    "<thead>"
    "<tr>"
    "<th>Group</th><th>n</th><th>Share with label</th>"
    "<th>Recall (cases caught)</th><th>Miss rate (FNR)</th>"
    "<th>False-positive rate</th><th>Precision</th><th>Flagged</th>"
    "</tr>"
    "</thead>"
    f"<tbody>{t_rows}</tbody>"
    "</table>"
)
st.markdown(table_html, unsafe_allow_html=True)
st.caption(
    f"Threshold {th_val:.2f} · Overall recall {app_overall['recall']:.1%}, false-positive rate {app_overall['fpr']:.1%}."
)

# Build the two side-by-side Bar Charts with 95% CI Error Bars & Dashed Overall Line
plot_df = pd.DataFrame(rows_list)
plot_df["rec_lo"] = [r["recall_ci"][0] for r in rows_list]
plot_df["rec_hi"] = [r["recall_ci"][1] for r in rows_list]
plot_df["fpr_lo"] = [r["fpr_ci"][0] for r in rows_list]
plot_df["fpr_hi"] = [r["fpr_ci"][1] for r in rows_list]

c_left, c_right = st.columns(2, gap="large")

with c_left:
    st.markdown(f"<h4 style='text-align:center; margin-bottom:4px;'>{t['he_chart_recall']}</h4>", unsafe_allow_html=True)
    bars_rec = (
        alt.Chart(plot_df)
        .mark_bar(color="#d95f02", width=55)
        .encode(
            x=alt.X("group:N", sort=None, title=""),
            y=alt.Y("recall:Q", scale=alt.Scale(domain=[0.0, 1.0]), title=""),
            tooltip=[
                alt.Tooltip("group:N", title="Group"),
                alt.Tooltip("recall:Q", format=".1%", title="Recall"),
            ],
        )
    )
    err_rec = (
        alt.Chart(plot_df)
        .mark_errorbar(color="#0f172a", ticks=True)
        .encode(
            x=alt.X("group:N", sort=None),
            y=alt.Y("rec_lo:Q", title=""),
            y2="rec_hi:Q",
        )
    )
    rule_rec = (
        alt.Chart(pd.DataFrame({"y": [float(app_overall["recall"])]}))
        .mark_rule(color="#334155", strokeDash=[5, 4], strokeWidth=1.6)
        .encode(y="y:Q")
    )
    st.altair_chart((bars_rec + err_rec + rule_rec).properties(height=300), use_container_width=True)

with c_right:
    st.markdown(f"<h4 style='text-align:center; margin-bottom:4px;'>{t['he_chart_fpr']}</h4>", unsafe_allow_html=True)
    bars_fpr = (
        alt.Chart(plot_df)
        .mark_bar(color="#1f77b4", width=55)
        .encode(
            x=alt.X("group:N", sort=None, title=""),
            y=alt.Y("fpr:Q", scale=alt.Scale(domain=[0.0, 1.0]), title=""),
            tooltip=[
                alt.Tooltip("group:N", title="Group"),
                alt.Tooltip("fpr:Q", format=".1%", title="False-positive rate"),
            ],
        )
    )
    err_fpr = (
        alt.Chart(plot_df)
        .mark_errorbar(color="#0f172a", ticks=True)
        .encode(
            x=alt.X("group:N", sort=None),
            y=alt.Y("fpr_lo:Q", title=""),
            y2="fpr_hi:Q",
        )
    )
    rule_fpr = (
        alt.Chart(pd.DataFrame({"y": [float(app_overall["fpr"])]}))
        .mark_rule(color="#334155", strokeDash=[5, 4], strokeWidth=1.6)
        .encode(y="y:Q")
    )
    st.altair_chart((bars_fpr + err_fpr + rule_fpr).properties(height=300), use_container_width=True)

st.caption(t["he_dashed_note"])

# What the numbers say
min_rec = f"{min(r['recall'] for r in rows_list):.1%}"
max_rec = f"{max(r['recall'] for r in rows_list):.1%}"
min_fpr = f"{min(r['fpr'] for r in rows_list):.1%}"
max_fpr = f"{max(r['fpr'] for r in rows_list):.1%}"
min_prev = f"{min(r['prevalence'] for r in rows_list):.1%}"
max_prev = f"{max(r['prevalence'] for r in rows_list):.1%}"

st.subheader(t["he_numbers_title"])
st.markdown(
    f"- {t['he_numbers_b1'].format(min_rec=min_rec, max_rec=max_rec)}\n"
    f"- {t['he_numbers_b2'].format(min_fpr=min_fpr, max_fpr=max_fpr)}\n"
    f"- {t['he_numbers_b3'].format(min_prev=min_prev, max_prev=max_prev)}"
)

with st.expander(t["he_exp1_title"]):
    st.markdown(t["he_exp1_body"])

with st.expander(t["he_exp2_title"]):
    sim_th = st.slider("Threshold", min_value=0.06, max_value=0.50, value=th_val, step=0.02, key="he_sim_th")
    scale_factor = (th_val / sim_th) ** 0.65
    sim_rows = []
    for r in rows_list:
        sim_rows.append(
            {
                "Group": r["group"],
                "n": r["n"],
                "Share with label": f"{r['prevalence']:.1%}",
                "Estimated Recall": f"{min(0.98, r['recall'] * scale_factor):.1%}",
                "Estimated FPR": f"{min(0.95, r['fpr'] * scale_factor):.1%}",
            }
        )
    st.dataframe(pd.DataFrame(sim_rows), use_container_width=True, hide_index=True)
