"""
app.py — Diabetes Screening Assistant (Main Page, 4 Languages: EN / VI / ZH / FR)
=================================================================================
User-Decision Statement:
"An adult answering questions at home uses the app to decide whether to get a blood sugar test."
"""

import pandas as pd
import streamlit as st
from i18n import (
    AGE_OPTIONS,
    FEATURE_LABELS_I18N,
    inject_friendly_css,
    load_artifacts,
    render_sidebar_and_get_lang,
    score_single_case,
)

st.set_page_config(
    page_title="Diabetes Screening Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_friendly_css()
lang_code, t = render_sidebar_and_get_lang()
main_pipe, base_pipe, meta = load_artifacts()

default_th = float(meta["cost_assumptions"]["chosen_threshold"])
if "active_threshold" not in st.session_state:
    st.session_state["active_threshold"] = default_th
active_th = float(st.session_state["active_threshold"])

# ==============================================================================
# 1. HEADER SECTION (MATCHING REFERENCE SCREENSHOT 1)
# ==============================================================================
st.markdown(f"<h1 style='font-size:2.35rem; font-weight:800; color:#1e293b; margin-bottom:8px;'>{t['app_title']}</h1>", unsafe_allow_html=True)
st.markdown(f"**{t['who_for_bold']}** {t['who_for_text']}")
st.caption(t["app_caption"])

GENHLTH_MAP = {
    1: t["genhlth_1"],
    2: t["genhlth_2"],
    3: t["genhlth_3"],
    4: t["genhlth_4"],
    5: t["genhlth_5"],
}

# ==============================================================================
# 2. 7-QUESTION USER-FRIENDLY INTAKE FORM (@st.form)
# ==============================================================================
with st.form("screening_assistant_form"):
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        age_code = st.selectbox(
            t["q_age"],
            options=list(AGE_OPTIONS.keys()),
            format_func=lambda k: AGE_OPTIONS[k],
            index=5,  # 45-49 default
        )
        height_cm = st.number_input(
            t["q_height"],
            min_value=120,
            max_value=230,
            value=165,
            step=1,
        )
        weight_kg = st.number_input(
            t["q_weight"],
            min_value=30,
            max_value=220,
            value=65,
            step=1,
        )
        gen_hlth = st.selectbox(
            t["q_genhlth"],
            options=list(GENHLTH_MAP.keys()),
            format_func=lambda k: GENHLTH_MAP[k],
            index=2,  # Good default
        )

    with col_right:
        high_bp = st.radio(
            t["q_highbp"],
            options=[0, 1],
            format_func=lambda v: t["no"] if v == 0 else t["yes"],
            horizontal=True,
            index=0,
        )
        high_chol = st.radio(
            t["q_highchol"],
            options=[0, 1],
            format_func=lambda v: t["no"] if v == 0 else t["yes"],
            horizontal=True,
            index=0,
        )
        heart_dis = st.radio(
            t["q_heart"],
            options=[0, 1],
            format_func=lambda v: t["no"] if v == 0 else t["yes"],
            horizontal=True,
            index=0,
        )
        diff_walk = st.radio(
            t["q_diffwalk"],
            options=[0, 1],
            format_func=lambda v: t["no"] if v == 0 else t["yes"],
            horizontal=True,
            index=0,
        )

    submitted = st.form_submit_button(t["btn_analyze"])

if submitted:
    st.session_state["has_analyzed"] = True

if not st.session_state.get("has_analyzed", False):
    st.info(t["prompt_analyze"])

# ==============================================================================
# 3. PREDICTION, ACTION RECOMMENDATION & PLAIN-ENGLISH EXPLANATION
# ==============================================================================
bmi_val = round(float(weight_kg) / ((float(height_cm) / 100.0) ** 2), 1)

input_df = pd.DataFrame(
    [
        {
            "Age": int(age_code),
            "BMI": float(bmi_val),
            "HighBP": int(high_bp),
            "HighChol": int(high_chol),
            "GenHlth": int(gen_hlth),
            "DiffWalk": int(diff_walk),
            "HeartDiseaseorAttack": int(heart_dis),
        }
    ]
)

prob_gb, prob_lr, contributions = score_single_case(input_df, main_pipe, base_pipe, meta)
is_flagged = prob_gb >= active_th

if st.session_state.get("has_analyzed", False):
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # Actionable Banner + Visual Risk Gauge
    if is_flagged:
        banner_bg = "#fff5f5"
        banner_border = "#e53e3e"
        banner_color = "#9b2c2c"
        banner_title = t["res_recommend_flag"]
    else:
        banner_bg = "#f0fdf4"
        banner_border = "#16a34a"
        banner_color = "#166534"
        banner_title = t["res_recommend_safe"]

    pin_pct = min(max(prob_gb * 100.0, 2.0), 98.0)
    th_pct = min(max(active_th * 100.0, 2.0), 98.0)
    nat_n = int(round(prob_gb * 100))

    result_card_html = (
        f"<div style='background:{banner_bg}; border:1px solid {banner_border}; border-left:6px solid {banner_border}; border-radius:12px; padding:18px 22px; margin-bottom:16px;'>"
        f"<h3 style='margin:0 0 6px 0; color:{banner_color}; font-size:1.25rem; font-weight:700;'>{banner_title}</h3>"
        f"<p style='margin:0; color:#334155; font-size:0.95rem;'>{t['res_natural_freq'].format(n=nat_n)}</p>"
        f"<div style='position:relative; height:18px; border-radius:999px; background:linear-gradient(90deg, #22c55e 0%, #facc15 25%, #f97316 55%, #ef4444 100%); margin:20px 0 22px 0;'>"
        f"<div style='position:absolute; left:{th_pct:.1f}%; top:-6px; bottom:-6px; width:2.5px; background:#0f172a;'></div>"
        f"<div style='position:absolute; left:{th_pct:.1f}%; top:-22px; transform:translateX(-50%); font-size:0.72rem; font-weight:700; color:#0f172a;'>Threshold {active_th:.2f}</div>"
        f"<div style='position:absolute; left:{pin_pct:.1f}%; top:50%; transform:translate(-50%, -50%); width:18px; height:18px; border-radius:50%; background:#ffffff; border:4px solid #0f172a; box-shadow:0 2px 6px rgba(0,0,0,0.25);'></div>"
        f"<div style='position:absolute; left:{pin_pct:.1f}%; bottom:-22px; transform:translateX(-50%); font-size:0.75rem; font-weight:700; color:#0f172a;'>{prob_gb:.1%}</div>"
        f"</div>"
        f"</div>"
    )
    st.markdown(result_card_html, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric(t["res_score_label"], f"{prob_gb:.1%}")
    m2.metric(t["res_threshold_label"], f"{active_th:.2f}")
    m3.metric(t["res_bmi_computed"], f"{bmi_val:.1f} kg/m²")

    # Top Factors Table (What pushed your score up or down?)
    st.markdown(f"### {t['res_top_factors_title']}")
    st.caption(t["res_top_factors_sub"])

    fname_map = FEATURE_LABELS_I18N.get(lang_code, FEATURE_LABELS_I18N["en"])
    answer_display = {
        "Age": AGE_OPTIONS[age_code],
        "BMI": f"{bmi_val:.1f} kg/m² ({height_cm} cm, {weight_kg} kg)",
        "GenHlth": GENHLTH_MAP[gen_hlth],
        "HighBP": t["yes"] if high_bp == 1 else t["no"],
        "HighChol": t["yes"] if high_chol == 1 else t["no"],
        "HeartDiseaseorAttack": t["yes"] if heart_dis == 1 else t["no"],
        "DiffWalk": t["yes"] if diff_walk == 1 else t["no"],
    }

    sorted_c = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    max_abs = max([abs(kv[1]) for kv in sorted_c] + [0.4])

    rows_html = ""
    for feat_k, val_c in sorted_c:
        bar_w = int(min(100, max(10, round((abs(val_c) / max_abs) * 100))))
        if val_c > 0.02:
            dir_txt = f"<span style='color:#c83e4d; font-weight:700;'>{t['pushes_up']}</span>"
            bar_col = "#c83e4d"
        elif val_c < -0.02:
            dir_txt = f"<span style='color:#0d9488; font-weight:700;'>{t['pushes_down']}</span>"
            bar_col = "#0d9488"
        else:
            dir_txt = f"<span style='color:#64748b; font-weight:600;'>{t['neutral']}</span>"
            bar_col = "#94a3b8"

        bar_div = (
            f"<div style='margin-top:4px; width:100%; background:#f1f5f9; border-radius:6px; height:7px;'>"
            f"<div style='width:{bar_w}%; height:7px; background:{bar_col}; border-radius:6px;'></div>"
            f"</div>"
        )
        rows_html += (
            f"<tr>"
            f"<td><b>{fname_map.get(feat_k, feat_k)}</b></td>"
            f"<td>{answer_display.get(feat_k, '')}</td>"
            f"<td style='min-width:190px;'>{dir_txt} <span style='color:#64748b; font-size:0.8rem;'>({val_c:+.2f})</span>{bar_div}</td>"
            f"</tr>"
        )

    factors_table = (
        "<table class='clean-table'>"
        "<thead>"
        f"<tr><th>{t['col_question']}</th><th>{t['col_your_answer']}</th><th>{t['col_impact']}</th></tr>"
        "</thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
    )
    st.markdown(factors_table, unsafe_allow_html=True)

    # What-If BMI drops by 2 points
    lower_bmi = max(18.5, round(bmi_val - 2.0, 1))
    whatif_df = input_df.copy()
    whatif_df["BMI"] = lower_bmi
    prob_whatif, _, _ = score_single_case(whatif_df, main_pipe, base_pipe, meta)
    delta_p = prob_whatif - prob_gb
    st.markdown(
        f"<div class='clean-card' style='margin-top:14px; background:#f8fafc;'>"
        f"<p style='margin:0 0 4px 0; font-weight:700; color:#1e293b;'>{t['what_if_title']}</p>"
        f"<p style='margin:0; font-size:0.9rem; color:#475569;'>"
        f"{t['what_if_desc'].format(bmi1=f'{bmi_val:.1f}', bmi2=f'{lower_bmi:.1f}', p1=f'{prob_gb:.1%}', p2=f'{prob_whatif:.1%}', delta=f'{delta_p:+.1%}')}"
        f"</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
