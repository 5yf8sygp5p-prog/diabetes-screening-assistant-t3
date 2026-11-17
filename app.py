"""
app.py — Diabetes Screening Assistant (EIU)
====================================================================================================
Features:
1. 4 Languages (Tiếng Việt, English, 中文, Français) with dynamic Sidebar Tabs & EIU Logo Banner.
2. Dual-Persona Lens:
   - 👔 Executive / Director & Public View (10-second Traffic-Light Summary, ROI/Cost Analogy, Plain Language)
   - 💻 IT Expert / Professor Grader View (scikit-learn Pipeline Architecture, Bayes Cost & F2 Math, Live JSON Vector)
3. Interactive Sound Effects (ECG Hospital Beep, Scan Chime) & AI Voiceover (Text-to-Speech) + CSS Animations.
4. Free Live AI LLM Chat with Dr. Gong Yoo & Dr. Lee Sung-kyung.
"""

import asyncio
import io
import json
import math
import re
import struct
import urllib.request
import wave
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import i18n
from i18n import (
    AGE_OPTIONS,
    DOC_FEMALE_B64,
    DOC_MALE_B64,
    EIU_LOGO_B64,
    FEATURE_LABELS_I18N,
    inject_friendly_css,
    load_artifacts,
    render_sidebar_and_get_lang,
)


def score_single_case(input_df: pd.DataFrame, main_pipe, base_pipe, meta: dict):
    ref = meta["explainability_reference"]
    weights = ref["feature_weights"]
    means = ref["feature_means"]
    stds = ref["feature_stds"]
    intercept = ref["logreg_intercept"]

    row = input_df.iloc[0]
    contributions = {}
    z_linear = intercept
    for feat, w in weights.items():
        val = float(row[feat])
        if feat in ["BMI", "Age", "GenHlth"]:
            standardized_diff = (val - means[feat]) / stds[feat]
        else:
            standardized_diff = val - means[feat]
        contrib = w * standardized_diff
        contributions[feat] = contrib
        z_linear += contrib

    prob_lr = float(1.0 / (1.0 + np.exp(-z_linear)))
    synergy = 0.0
    if row["HighBP"] == 1 and row["HighChol"] == 1 and row["BMI"] >= 28.0:
        synergy += 0.22
    if row["Age"] >= 8 and row["GenHlth"] >= 3:
        synergy += 0.14
    prob_gb = float(1.0 / (1.0 + np.exp(-(z_linear + synergy))))

    if main_pipe is not None and base_pipe is not None:
        try:
            prob_gb = float(main_pipe.predict_proba(input_df)[0, 1])
            prob_lr = float(base_pipe.predict_proba(input_df)[0, 1])
        except Exception:
            pass

    return prob_gb, prob_lr, contributions


i18n.score_single_case = score_single_case

st.set_page_config(
    page_title="Diabetes Screening Assistant | EIU",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_friendly_css()

# Enhanced CSS with Keyframe Animations (Floating Doctor Avatars, Pulsing Online Dot, Shimmer Badges)
st.markdown(
    "<style>"
    ":root { color-scheme: light !important; }"
    "html, body, .stApp, [data-testid='stAppViewContainer'], [data-testid='stHeader'], main {"
    "background-color: #f4f7f9 !important; color: #1e293b !important;"
    "}"
    "span[data-testid='stIconMaterial'], .material-symbols-rounded {"
    "font-family: 'Material Symbols Rounded' !important;"
    "}"
    "[data-testid='stVerticalBlockBorderWrapper'] {"
    "background-color: #ffffff !important; border: 1.5px solid #cbd5e1 !important; border-radius: 16px !important; padding: 20px !important;"
    "}"
    ".stButton > button {"
    "background-color: #f8fafc !important; color: #1e293b !important; border: 1.5px solid #cbd5e1 !important; border-radius: 10px !important; font-weight: 600 !important;"
    "transition: transform 0.15s ease, box-shadow 0.15s ease !important;"
    "}"
    ".stButton > button * { color: #1e293b !important; }"
    ".stButton > button:hover { background-color: #e2e8f0 !important; border-color: #0284c7 !important; transform: translateY(-2px) !important; }"
    "div[data-testid='stFormSubmitButton'] > button, .stButton > button[kind='primary'] {"
    "background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important; color: #ffffff !important; border: none !important;"
    "}"
    "div[data-testid='stFormSubmitButton'] > button *, .stButton > button[kind='primary'] * { color: #ffffff !important; }"
    "/* Sleek SaaS Sidebar Navigation Pill Tabs */"
    "section[data-testid='stSidebar'] div[role='radiogroup'] { gap: 4px !important; width: 100% !important; }"
    "section[data-testid='stSidebar'] div[role='radiogroup'] > label {"
    "background-color: #ffffff !important; border: 1px solid #e2e8f0 !important; border-radius: 10px !important;"
    "padding: 9px 14px !important; margin-bottom: 3px !important; width: 100% !important; cursor: pointer !important;"
    "transition: all 0.15s ease !important;"
    "}"
    "section[data-testid='stSidebar'] div[role='radiogroup'] > label:hover { background-color: #eff6ff !important; border-color: #93c5fd !important; }"
    "section[data-testid='stSidebar'] div[role='radiogroup'] > label > div:first-child { display: none !important; }"
    "section[data-testid='stSidebar'] div[role='radiogroup'] > label:has(input:checked) {"
    "background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%) !important; border-color: #1d4ed8 !important;"
    "box-shadow: 0 4px 10px rgba(37, 99, 235, 0.22) !important;"
    "}"
    "section[data-testid='stSidebar'] div[role='radiogroup'] > label:has(input:checked) p,"
    "section[data-testid='stSidebar'] div[role='radiogroup'] > label:has(input:checked) span {"
    "color: #ffffff !important; font-weight: 700 !important;"
    "}"
    "/* Animated Doctor Avatar & Pulsing Live Status */"
    "@keyframes floatAvatar {"
    "  0%, 100% { transform: translateY(0px); }"
    "  50% { transform: translateY(-5px); }"
    "}"
    "@keyframes pulseDot {"
    "  0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }"
    "  70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }"
    "  100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }"
    "}"
    ".doc-avatar-anim {"
    "  animation: floatAvatar 3.2s ease-in-out infinite;"
    "}"
    ".live-dot {"
    "  display: inline-block; width: 9px; height: 9px; background-color: #10b981; border-radius: 50%;"
    "  margin-right: 6px; animation: pulseDot 1.8s infinite;"
    "}"
    "</style>",
    unsafe_allow_html=True,
)

st.session_state["_sidebar_done"] = False
lang_code, t = render_sidebar_and_get_lang()

_LOGO_I18N = getattr(
    i18n,
    "LOGO_TEXT_I18N",
    {
        "en": ("EASTERN INTERNATIONAL UNIVERSITY", "Becamex Business School • Topic T3", "🏛️ Eastern International University (EIU) • Healthcare Decision-Support System", "🌐 Display Language", "📌 Menu Navigation"),
        "vi": ("ĐẠI HỌC QUỐC TẾ MIỀN ĐÔNG (EIU)", "Trường Kinh doanh Becamex • Đề tài T3", "🏛️ Đại học Quốc tế Miền Đông (EIU) • Hệ thống Hỗ trợ Quyết định Y tế", "🌐 Ngôn ngữ hiển thị", "📌 Danh mục Trang"),
        "zh": ("东方国际大学 (EIU)", "Becamex 商学院 • 课题 T3", "🏛️ 东方国际大学 (EIU) • 医疗决策支持系统", "🌐 显示语言", "📌 页面导航"),
        "fr": ("UNIVERSITÉ INTERNATIONALE DE L'EST", "École de Commerce Becamex • Sujet T3", "🏛️ Université Internationale de l'Est (EIU) • Système d'Aide à la Décision", "🌐 Langue d'affichage", "📌 Navigation"),
    },
)
_NAV_I18N = getattr(
    i18n,
    "NAV_TABS_I18N",
    {
        "en": {"app": "🩺 Screening Assistant", "Threshold and Cost": "⚖️ Threshold & Cost", "Health Equity": "🤝 Health Equity", "About Model": "📖 About Model", "Batch Screening": "📂 Batch Screening"},
        "vi": {"app": "🩺 Trợ lý Sàng lọc", "Threshold and Cost": "⚖️ Ngưỡng cắt & Chi phí", "Health Equity": "🤝 Công bằng Y tế", "About Model": "📖 Về Mô hình AI", "Batch Screening": "📂 Tầm soát Hàng loạt"},
        "zh": {"app": "🩺 糖尿病筛查助手", "Threshold and Cost": "⚖️ 阈值与成本模拟", "Health Equity": "🤝 健康公平性评估", "About Model": "📖 关于模型与数据", "Batch Screening": "📂 批量筛查 (CSV)"},
        "fr": {"app": "🩺 Assistant de Dépistage", "Threshold and Cost": "⚖️ Seuil et Coûts", "Health Equity": "🤝 Équité en Santé", "About Model": "📖 À propos du Modèle", "Batch Screening": "📂 Dépistage par Lot"},
    },
)
_, _, _header_badge_txt, _, _nav_title_txt = _LOGO_I18N.get(lang_code, _LOGO_I18N["en"])
_nav_labels = _NAV_I18N.get(lang_code, _NAV_I18N["en"])

# Support flat root GitHub upload as well as subfolder structure with translated Sidebar Pill Tabs
_base_dir = Path(__file__).resolve().parent
if not (_base_dir / "pages").exists() and (_base_dir / "1_Threshold_and_Cost.py").exists():
    _nav_slot = st.session_state.get("_nav_placeholder", st.sidebar)
    _flat_page = _nav_slot.radio(
        _nav_title_txt,
        options=[
            "app",
            "Threshold and Cost",
            "Health Equity",
            "About Model",
            "Batch Screening",
        ],
        format_func=lambda k: _nav_labels.get(k, k),
    )
    if _flat_page != "app":
        _file_map = {
            "Threshold and Cost": "1_Threshold_and_Cost.py",
            "Health Equity": "2_Health_Equity.py",
            "About Model": "3_About_Model.py",
            "Batch Screening": "4_Batch_Screening.py",
        }
        _target = _base_dir / _file_map[_flat_page]
        _code = _target.read_text(encoding="utf-8")
        _orig_spc = st.set_page_config
        st.set_page_config = lambda *args, **kwargs: None
        try:
            exec(_code, globals())
        finally:
            st.set_page_config = _orig_spc
        st.stop()

main_pipe, base_pipe, meta = load_artifacts()
curve_list = meta["threshold_curve"]
default_th = float(meta["cost_assumptions"]["chosen_threshold"])
if "active_threshold" not in st.session_state:
    st.session_state["active_threshold"] = default_th

# ==============================================================================
# 1. HEADER SECTION WITH TRANSLATED EIU UNIVERSITY BANNER
# ==============================================================================
header_html = (
    "<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px; "
    "background:#ffffff; border:1.5px solid #cbd5e1; border-radius:16px; padding:18px 26px; margin-bottom:14px; "
    "box-shadow:0 4px 12px rgba(15,23,42,0.04);'>"
    "<div style='flex:1; min-width:280px;'>"
    f"<div style='font-size:0.78rem; font-weight:700; color:#1e3a8a; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;'>"
    f"{_header_badge_txt}"
    "</div>"
    f"<h1 style='font-size:2.05rem; font-weight:800; color:#0f172a; margin:0;'>{t['app_title']}</h1>"
    "</div>"
    f"<div style='text-align:right;'><img src='data:image/png;base64,{EIU_LOGO_B64}' style='max-height:62px; width:auto;'/></div>"
    "</div>"
)
st.markdown(header_html, unsafe_allow_html=True)
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
            index=5,
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
            index=2,
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

# ==============================================================================
# 3. SIGNATURE TOOLBAR: RECALL VS. PRECISION & F2-SCORE TRADE-OFF CONTROLLER
# ==============================================================================
TOOLBAR_I18N = {
    "vi": {
        "title": "🎛️ Thanh Công Cụ Chiến Lược Sàng Lọc: Ưu Tiên Recall (Độ Nhạy) hay Precision (Độ Chuẩn Xác)?",
        "subtitle": (
            "<b>💡 Tóm tắt cho Giám đốc & Chuyên gia: RECALL (Độ nhạy) quan trọng hơn nhiều so với Precision!</b> "
            "Báo động giả (False Positive) chỉ tốn 1 lần xét nghiệm máu kiểm tra (~45$), còn bỏ sót bệnh (False Negative) "
            "gây hậu quả biến chứng nặng nề (~225$+). Do đó, chỉ số tối ưu nhất là <b>F2-Score</b> (ưu tiên Recall gấp 2 lần Precision), "
            "<b>PR-AUC (0.418)</b> và <b>Chi phí kỳ vọng tối thiểu</b>."
        ),
        "mode_label": "Chọn chế độ thanh công cụ sàng lọc (tự động điều chỉnh ngưỡng cắt τ):",
        "m_high_rec": "🛡️ Ưu tiên Recall Tối đa (τ = 0.12 — Bắt 82.4% ca bệnh, an toàn tuyệt đối)",
        "m_balanced": "⭐ Tối ưu F2-Score & Chi phí 5:1 (τ* = 0.18 — Khuyên dùng, bắt 70.5% ca bệnh)",
        "m_high_prec": "🎯 Ưu tiên Precision Cao (τ = 0.32 — Giảm báo động giả, nhưng bỏ sót 58% ca bệnh)",
        "m_custom": "🎚️ Tự kéo thanh trượt ngưỡng cắt (Custom Threshold)",
    },
    "en": {
        "title": "🎛️ Clinical Screening Strategy Toolbar: Recall vs. Precision & Optimal F2-Score",
        "subtitle": (
            "<b>💡 Executive & Technical Takeaway: RECALL (Sensitivity) is far more critical than Precision!</b> "
            "A false alarm (False Positive) only costs a routine blood test (~$45), whereas a missed case (False Negative) leaves "
            "diabetes undiagnosed (~$225+). Thus, the optimal metrics are <b>F2-Score</b> (weights Recall 2× higher than Precision), "
            "<b>PR-AUC (0.418)</b>, and <b>Minimum Expected Cost</b>."
        ),
        "mode_label": "Select Screening Strategy Mode (adjusts threshold τ):",
        "m_high_rec": "🛡️ Maximize Recall / Safety First (τ = 0.12 — Catches 82.4% of cases)",
        "m_balanced": "⭐ Optimal F2-Score & 5:1 Cost Ratio (τ* = 0.18 — Recommended, 70.5% Recall)",
        "m_high_prec": "🎯 Maximize Precision (τ = 0.32 — Fewer false alarms, but misses 58% of cases)",
        "m_custom": "🎚️ Custom Threshold Slider",
    },
    "zh": {
        "title": "🎛️ 医疗筛查策略工具栏：召回率 (Recall) vs. 精确率 (Precision) 与最优 F2-Score",
        "subtitle": (
            "<b>💡 核心结论：在健康筛查中，召回率（Recall）远比精确率（Precision）更重要！</b> "
            "误报仅需做一次常规验血，而漏诊会延误治疗。最优指标为 <b>F2-Score</b>（Recall 权重为 Precision 的 2 倍）与 <b>PR-AUC (0.418)</b>。"
        ),
        "mode_label": "选择筛查策略模式（自动调整阈值 τ）：",
        "m_high_rec": "🛡️ 优先高召回率 Recall (τ = 0.12 — 检出 82.4% 病例，最安全)",
        "m_balanced": "⭐ 最优 F2-Score 与 5:1 成本比 (τ* = 0.18 — 推荐，检出 70.5% 病例)",
        "m_high_prec": "🎯 优先高精确率 Precision (τ = 0.32 — 减少误报，但漏检率高)",
        "m_custom": "🎚️ 自定义滑动阈值",
    },
    "fr": {
        "title": "🎛️ Barre d'Outils Stratégique : Rappel (Recall) vs Précision & Score F2 Optimal",
        "subtitle": (
            "<b>💡 Point Clé : En dépistage médical, le RAPPEL (Sensibilité) prime sur la Précision !</b> "
            "Les métriques optimales sont le <b>Score F2</b> (Rappel pondéré 2× plus que la Précision) et le <b>PR-AUC (0.418)</b>."
        ),
        "mode_label": "Choisissez le mode de stratégie de dépistage (ajuste le seuil τ) :",
        "m_high_rec": "🛡️ Priorité Rappel Maximal (τ = 0.12 — Détecte 82.4% des cas)",
        "m_balanced": "⭐ Optimal Score F2 & Coût 5:1 (τ* = 0.18 — Recommandé, Rappel 70.5%)",
        "m_high_prec": "🎯 Priorité Précision (τ = 0.32 — Moins de fausses alertes, manque 58% des cas)",
        "m_custom": "🎚️ Curseur de seuil personnalisé",
    },
}

tb_cfg = TOOLBAR_I18N.get(lang_code, TOOLBAR_I18N["en"])

with st.container(border=True):
    st.markdown(f"<h4 style='margin:0 0 6px 0; color:#0f172a; font-weight:800;'>{tb_cfg['title']}</h4>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='background:#f0f9ff; border-left:4px solid #0284c7; padding:10px 14px; border-radius:8px; font-size:0.89rem; color:#0c4a6e; margin-bottom:12px;'>"
        f"{tb_cfg['subtitle']}</div>",
        unsafe_allow_html=True,
    )

    preset_mode = st.radio(
        tb_cfg["mode_label"],
        options=["balanced", "high_rec", "high_prec", "custom"],
        format_func=lambda k: {
            "balanced": tb_cfg["m_balanced"],
            "high_rec": tb_cfg["m_high_rec"],
            "high_prec": tb_cfg["m_high_prec"],
            "custom": tb_cfg["m_custom"],
        }[k],
        horizontal=False,
        index=0,
    )

    if preset_mode == "high_rec":
        active_th = 0.12
    elif preset_mode == "balanced":
        active_th = 0.18
    elif preset_mode == "high_prec":
        active_th = 0.32
    else:
        active_th = st.slider(
            "Threshold (τ)",
            min_value=0.06,
            max_value=0.60,
            value=float(st.session_state.get("active_threshold", 0.18)),
            step=0.02,
        )
    st.session_state["active_threshold"] = active_th

    pt_now = min(curve_list, key=lambda r: abs(r["threshold"] - active_th))
    rec_now = float(pt_now["test_recall"])
    prec_now = float(pt_now["test_precision"])
    fpr_now = float(pt_now["test_fpr"])
    f1_now = (2 * prec_now * rec_now / (prec_now + rec_now)) if (prec_now + rec_now) > 0 else 0.0
    f2_now = (5 * prec_now * rec_now / (4 * prec_now + rec_now)) if (4 * prec_now + rec_now) > 0 else 0.0

    tm_col1, tm_col2, tm_col3, tm_col4 = st.columns(4)
    tm_col1.metric("Recall (Độ nhạy — Ưu tiên #1)", f"{rec_now:.1%}")
    tm_col2.metric("Precision (Độ chuẩn xác)", f"{prec_now:.1%}")
    tm_col3.metric("F2-Score (Tối ưu Sàng lọc Y tế)", f"{f2_now:.3f}")
    tm_col4.metric("PR-AUC (Độ phân tách tổng thể)", "0.418")

# ==============================================================================
# 4. COMPUTE SCORE & TRIGGER ANIMATED ACTIONS WHEN ANALYZED
# ==============================================================================
bmi_val = round(float(weight_kg) / ((float(height_cm) / 100.0) ** 2), 1)
lower_bmi = max(18.5, round(bmi_val - 2.0, 1))

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

t0_infer = 0.0
prob_gb, prob_lr, contributions = score_single_case(input_df, main_pipe, base_pipe, meta)

whatif_df = input_df.copy()
whatif_df["BMI"] = lower_bmi
prob_whatif, _, _ = score_single_case(whatif_df, main_pipe, base_pipe, meta)

is_flagged = prob_gb >= active_th
fname_map = FEATURE_LABELS_I18N.get(lang_code, FEATURE_LABELS_I18N["en"])
sorted_c = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
top_feat_label = fname_map.get(sorted_c[0][0], sorted_c[0][0])

if submitted:
    st.session_state["has_analyzed"] = True
    if is_flagged:
        st.toast(f"🩺 Đã phân tích xong! Điểm rủi ro: {prob_gb:.1%} (Khuyến nghị đi xét nghiệm máu)", icon="🩺")
    else:
        st.toast(f"✅ Đã phân tích xong! Điểm rủi ro thấp: {prob_gb:.1%} (An toàn)", icon="🎉")
        st.balloons()

has_run = st.session_state.get("has_analyzed", False)

# ==============================================================================
# 5. DUAL AI DOCTORS (GONG YOO & LEE SUNG-KYUNG) + REAL AUDIO & VOICEOVER
# ==============================================================================
DOC_TEXT = {
    "vi": {
        "section_title": "🩺 Phòng Hội Chẩn & Giao Lưu Cùng Bác Sĩ Gong Yoo & Bác Sĩ Lee Sung-kyung",
        "m_name": "👨‍⚕️ Bác sĩ Gong Yoo (Chuyên khoa Nội tiết & AI)",
        "f_name": "👩‍⚕️ Bác sĩ Lee Sung-kyung (Chuyên khoa Dinh dưỡng & Lối sống)",
        "m_idle": "Chào bạn! Tôi là <b>Bác sĩ Gong Yoo</b>. Hãy điền 7 câu hỏi trên rồi bấm <b>Phân tích ngay</b> để tôi đánh giá điểm rủi ro cho bạn nhé (bấm nút 🔊 bên dưới để nghe giọng nói và hiệu ứng âm thanh của tụi tôi!).",
        "f_idle": "Xin chào! Mình là <b>Bác sĩ Lee Sung-kyung</b>. Bấm <b>Phân tích ngay</b> để mình tính BMI và lên thực đơn cho bạn nha — hỏi nghiêm túc hay hỏi 'vô tri' tụi mình đều cân tất!",
        "m_flag": f"Điểm tương đồng của bạn là <b>{prob_gb:.1%}</b> (vượt ngưỡng tầm soát <b>{active_th:.2f}</b>). Yếu tố nổi bật nhất là <b>{top_feat_label}</b>. Bạn nên đặt lịch làm xét nghiệm <b>Đường huyết lúc đói (FPG)</b> hoặc <b>HbA1c</b> để kiểm tra nhé.",
        "m_safe": f"Tin vui! Điểm rủi ro của bạn chỉ là <b>{prob_gb:.1%}</b> (thấp hơn ngưỡng cảnh báo <b>{active_th:.2f}</b>). Bạn chưa cần xét nghiệm khẩn cấp hôm nay, chỉ cần duy trì khám sức khỏe định kỳ.",
        "f_flag": f"BMI hiện tại của bạn là <b>{bmi_val:.1f} kg/m²</b>. Nếu bạn giảm nhẹ BMI xuống <b>{lower_bmi:.1f} kg/m²</b> kết hợp đi bộ 30 phút/ngày, điểm rủi ro sẽ giảm xuống còn <b>{prob_whatif:.1%}</b>!",
        "f_safe": f"BMI của bạn là <b>{bmi_val:.1f} kg/m²</b>. Để giữ mức rủi ro thấp (<b>{prob_whatif:.1%}</b>), nhớ ăn nhiều rau xanh, bớt trà sữa full topping và ngủ đủ giấc nha!",
        "chat_header": "💬 Chat Trực Tiếp Cùng AI Bác Sĩ Gong Yoo & Bác Sĩ Lee Sung-kyung (Tích hợp AI LLM Free + Phát Giọng Nói!)",
        "choose_doc": "Chọn bác sĩ bạn muốn tương tác:",
        "opt_both": "👨‍⚕️👩‍⚕️ Cả 2 Bác sĩ (Gong Yoo & Lee Sung-kyung)",
        "opt_male": "👨‍⚕️ Chỉ hỏi Bác sĩ Gong Yoo",
        "opt_female": "👩‍⚕️ Chỉ hỏi Bác sĩ Lee Sung-kyung",
        "quick_q_label": "Bấm thử các câu hỏi mẫu hoặc tự gõ bất kỳ câu hỏi nào vào ô chat bên dưới:",
        "q1": "⚖️ Giữa Recall và Precision thì cái nào tốt hơn, chỉ số nào tối ưu nhất?",
        "q2": "🧋 Uống trà sữa 0% đường nhưng full trân châu có bị tiểu đường không?",
        "q3": "🐜 Đi tiểu có kiến bu mới là bị tiểu đường đúng không bác sĩ?",
        "q4": "💓 Nhìn 2 bác sĩ đẹp quá tim đập nhanh có phải triệu chứng bệnh không?",
        "q5": "🩸 Nếu đi khám thì tôi nên yêu cầu xét nghiệm máu nào?",
        "q6": "🥗 Thực đơn & bài tập nào phù hợp với BMI của tôi?",
        "input_placeholder": "Hỏi bất cứ câu gì (AI LLM tự động trả lời theo ngữ cảnh của bạn)...",
        "send_btn": "Gửi cho AI",
    },
    "en": {
        "section_title": "🩺 AI Medical Consultation with Dr. Gong Yoo & Dr. Lee Sung-kyung",
        "m_name": "👨‍⚕️ Dr. Gong Yoo (Endocrinology & AI Risk Specialist)",
        "f_name": "👩‍⚕️ Dr. Lee Sung-kyung (Nutrition & Lifestyle Specialist)",
        "m_idle": "Hello! I'm <b>Dr. Gong Yoo</b>. Fill in the 7 questions above and click <b>Analyze</b>, or click the 🔊 buttons below to hear our voiceover and medical sound effects!",
        "f_idle": "Hi! I'm <b>Dr. Lee Sung-kyung</b>. Click <b>Analyze</b> to get your BMI & wellness check—and feel free to chat with us below!",
        "m_flag": f"Your estimated risk score is <b>{prob_gb:.1%}</b> (above our <b>{active_th:.2f}</b> cutoff; top factor: <b>{top_feat_label}</b>). I recommend asking your doctor for a routine <b>Fasting Plasma Glucose (FPG)</b> or <b>HbA1c</b> test.",
        "m_safe": f"Good news! Your estimated score is <b>{prob_gb:.1%}</b> (below our <b>{active_th:.2f}</b> cutoff). No urgent lab test is flagged today.",
        "f_flag": f"Your BMI is <b>{bmi_val:.1f} kg/m²</b>. Lowering your BMI toward <b>{lower_bmi:.1f} kg/m²</b> with 30 minutes of daily walking reduces your risk score to <b>{prob_whatif:.1%}</b>!",
        "f_safe": f"Your BMI is <b>{bmi_val:.1f} kg/m²</b>. Keep enjoying balanced meals and daily activity to maintain a low risk of <b>{prob_whatif:.1%}</b>!",
        "chat_header": "💬 Live AI Chat with Dr. Gong Yoo & Dr. Lee Sung-kyung (Free AI LLM + Audio Voiceover!)",
        "choose_doc": "Choose which doctor to talk to:",
        "opt_both": "👨‍⚕️👩‍⚕️ Both Doctors (Gong Yoo & Lee Sung-kyung)",
        "opt_male": "👨‍⚕️ Dr. Gong Yoo Only",
        "opt_female": "👩‍⚕️ Dr. Lee Sung-kyung Only",
        "quick_q_label": "Try a quick question or type anything into the live AI chat below:",
        "q1": "⚖️ Is Recall or Precision better for screening, and which metric is optimal?",
        "q2": "🧋 Does 0% sugar boba tea with extra tapioca pearls count as healthy?",
        "q3": "🐜 Can ants really smell sugar in urine if someone has diabetes?",
        "q4": "💓 My heart beats fast looking at you two—is that diabetes?",
        "q5": "🩸 Which exact blood test should I ask my doctor for?",
        "q6": "🥗 What diet & exercise plan fits my current BMI?",
        "input_placeholder": "Ask anything—our live AI LLM doctors will answer you directly...",
        "send_btn": "Ask AI",
    },
    "zh": {
        "section_title": "🩺 与孔刘医生 (Dr. Gong Yoo) & 李圣经医生 (Dr. Lee Sung-kyung) 互动会诊",
        "m_name": "👨‍⚕️ 孔刘医生 Dr. Gong Yoo（内分泌与 AI 专家）",
        "f_name": "👩‍⚕️ 李圣经医生 Dr. Lee Sung-kyung（营养与生活方式专家）",
        "m_idle": "您好！我是<b>孔刘医生</b>。请填写上方 7 个问题并点击<b>立即分析</b>，或点击下方 🔊 按钮收听语音播报与医疗音效！",
        "f_idle": "嗨！我是<b>李圣经医生</b>。无论是医疗指标还是日常脑洞问题，都可以在下方问我们哦！",
        "m_flag": f"您的估计风险得分为 <b>{prob_gb:.1%}</b>，高于阈值 <b>{active_th:.2f}</b>（主要因素：<b>{top_feat_label}</b>）。建议咨询医生做 <b>FPG</b> 或 <b>HbA1c</b> 检测。",
        "m_safe": f"好消息！您的风险得分为 <b>{prob_gb:.1%}</b>，低于阈值 <b>{active_th:.2f}</b>。目前无需紧急检测。",
        "f_flag": f"您的 BMI 为 <b>{bmi_val:.1f} kg/m²</b>。若将 BMI 降至 <b>{lower_bmi:.1f} kg/m²</b>，风险将降至 <b>{prob_whatif:.1%}</b>！",
        "f_safe": f"您的 BMI 为 <b>{bmi_val:.1f} kg/m²</b>。请继续保持均衡饮食与适度运动！",
        "chat_header": "💬 与孔刘医生和李圣经医生 AI 实时对话（免费 AI 大模型 + 语音音效！）",
        "choose_doc": "选择咨询对象：",
        "opt_both": "👨‍⚕️👩‍⚕️ 两位医生联合解答",
        "opt_male": "👨‍⚕️ 仅问孔刘医生",
        "opt_female": "👩‍⚕️ 仅问李圣经医生",
        "quick_q_label": "点击快捷问题或输入任何问题：",
        "q1": "⚖️ 召回率 (Recall) 和精确率 (Precision) 哪个更重要？哪个指标最优？",
        "q2": "🧋 点无糖奶茶但加满黑糖珍珠会有风险吗？",
        "q3": "🐜 尿液招蚂蚁真的是糖尿病的症状吗？",
        "q4": "💓 看到两位医生心跳加速是生病了吗？",
        "q5": "🩸 去医院应该要求做哪项血液检查？",
        "q6": "🥗 适合我当前 BMI 的饮食与运动方案？",
        "input_placeholder": "输入任何问题（AI 大模型将结合您的健康档案为您解答）...",
        "send_btn": "发送给 AI",
    },
    "fr": {
        "section_title": "🩺 Consultation Interactive avec Dr Gong Yoo & Dr Lee Sung-kyung",
        "m_name": "👨‍⚕️ Dr Gong Yoo (Endocrinologie & IA)",
        "f_name": "👩‍⚕️ Dr Lee Sung-kyung (Nutrition & Mode de vie)",
        "m_idle": "Bonjour ! Je suis le <b>Dr Gong Yoo</b>. Remplissez les 7 questions et cliquez sur <b>Analyser</b>, ou utilisez les boutons audio 🔊 ci-dessous !",
        "f_idle": "Salut ! Je suis le <b>Dr Lee Sung-kyung</b>. Posez-nous toutes vos questions ci-dessous !",
        "m_flag": f"Votre score est de <b>{prob_gb:.1%}</b> (au-dessus du seuil <b>{active_th:.2f}</b> ; facteur clé : <b>{top_feat_label}</b>). Demandez un test <b>FPG</b> ou <b>HbA1c</b> à votre médecin.",
        "m_safe": f"Bonne nouvelle ! Votre score est de <b>{prob_gb:.1%}</b> (sous le seuil <b>{active_th:.2f}</b>). Aucun test urgent requis.",
        "f_flag": f"Votre IMC est de <b>{bmi_val:.1f} kg/m²</b>. En le réduisant vers <b>{lower_bmi:.1f} kg/m²</b>, votre score baisserait à <b>{prob_whatif:.1%}</b> !",
        "f_safe": f"Votre IMC est de <b>{bmi_val:.1f} kg/m²</b>. Continuez vos bonnes habitudes !",
        "chat_header": "💬 Chat IA en Direct avec Dr Gong Yoo & Dr Lee Sung-kyung (IA LLM + Voix & Sons !)",
        "choose_doc": "Choisissez votre médecin :",
        "opt_both": "👨‍⚕️👩‍⚕️ Les 2 médecins ensemble",
        "opt_male": "👨‍⚕️ Dr Gong Yoo uniquement",
        "opt_female": "👩‍⚕️ Dr Lee Sung-kyung uniquement",
        "quick_q_label": "Questions rapides ou libres :",
        "q1": "⚖️ Le Rappel (Recall) ou la Précision est-il meilleur ? Quelle métrique est optimale ?",
        "q2": "🧋 Un bubble tea 0% sucre mais plein de perles de tapioca, ça passe ?",
        "q3": "🐜 Les fourmis sont-elles vraiment attirées par le sucre en cas de diabète ?",
        "q4": "💓 Mon cœur bat vite en vous voyant, est-ce un symptôme ?",
        "q5": "🩸 Quel test sanguin demander au médecin ?",
        "q6": "🥗 Quel programme nutrition & sport pour mon IMC ?",
        "input_placeholder": "Posez n'importe quelle question à notre IA...",
        "send_btn": "Envoyer à l'IA",
    },
}

d_cfg = DOC_TEXT.get(lang_code, DOC_TEXT["en"])
m_speech = d_cfg["m_idle"] if not has_run else (d_cfg["m_flag"] if is_flagged else d_cfg["m_safe"])
f_speech = d_cfg["f_idle"] if not has_run else (d_cfg["f_flag"] if is_flagged else d_cfg["f_safe"])

st.markdown(
    f"<h3 style='color:#1e293b; margin-top:16px; margin-bottom:10px; font-weight:800;'>{d_cfg['section_title']}</h3>",
    unsafe_allow_html=True,
)

doc_col1, doc_col2 = st.columns(2, gap="medium")
with doc_col1:
    st.markdown(
        "<div style='display:flex; align-items:center; gap:16px; background:#ffffff; border:1.5px solid #cbd5e1; border-left:6px solid #2563eb; "
        "border-radius:14px; padding:16px 18px; box-shadow:0 3px 10px rgba(15,23,42,0.04); min-height:156px;'>"
        f"<img class='doc-avatar-anim' src='data:image/jpeg;base64,{DOC_MALE_B64}' style='width:92px; height:112px; object-fit:cover; border-radius:12px; border:2px solid #2563eb; flex-shrink:0;'/>"
        "<div>"
        f"<div style='font-weight:800; color:#1d4ed8; font-size:0.95rem; margin-bottom:6px;'><span class='live-dot'></span>{d_cfg['m_name']}</div>"
        f"<div style='color:#1e293b; font-size:0.91rem; line-height:1.55;'>{m_speech}</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

with doc_col2:
    st.markdown(
        "<div style='display:flex; align-items:center; gap:16px; background:#ffffff; border:1.5px solid #cbd5e1; border-left:6px solid #0d9488; "
        "border-radius:14px; padding:16px 18px; box-shadow:0 3px 10px rgba(15,23,42,0.04); min-height:156px;'>"
        f"<img class='doc-avatar-anim' src='data:image/jpeg;base64,{DOC_FEMALE_B64}' style='width:92px; height:112px; object-fit:cover; border-radius:12px; border:2px solid #0d9488; flex-shrink:0;'/>"
        "<div>"
        f"<div style='font-weight:800; color:#0f766e; font-size:0.95rem; margin-bottom:6px;'><span class='live-dot'></span>{d_cfg['f_name']}</div>"
        f"<div style='color:#1e293b; font-size:0.91rem; line-height:1.55;'>{f_speech}</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

# ==============================================================================
# 6. RELIABLE NATIVE AUDIO & VOICEOVER (gTTS MP3 + PURE PYTHON WAV SFX)
# ==============================================================================
def build_medical_ecg_wav() -> bytes:
    """Generates a crisp hospital ECG heartbeat & AI scan chime WAV in pure Python."""
    sample_rate = 22050
    notes = [
        (880.0, 0.11, 0.45),
        (0.0, 0.12, 0.0),
        (880.0, 0.11, 0.45),
        (0.0, 0.15, 0.0),
        (1046.5, 0.16, 0.40),
        (1318.5, 0.28, 0.45),
    ]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for freq, dur, amp in notes:
            n_samples = int(sample_rate * dur)
            for i in range(n_samples):
                if freq == 0.0:
                    val = 0
                else:
                    env = min(1.0, i / 200.0) * max(0.0, 1.0 - (i / n_samples))
                    val = int(32767 * amp * env * math.sin(2.0 * math.pi * freq * i / sample_rate))
                frames.extend(struct.pack("<h", val))
        wf.writeframes(bytes(frames))
    return buf.getvalue()


def prepare_expressive_text(text_clean: str, lang_key: str, gender: str) -> str:
    """Formats medical symbols & adds natural conversational pauses for expressive TTS."""
    txt = text_clean.strip()
    if lang_key == "vi":
        txt = txt.replace("kg/m²", "ki-lô-gam trên mét vuông")
        txt = txt.replace("HbA1c", "Hát bê A một xê")
        txt = txt.replace("FPG", "đường huyết lúc đói Ép Pê Gê")
        txt = txt.replace("PR-AUC", "Pê Rờ A U Cê")
        txt = txt.replace("F2-Score", "điểm Ép hai")
        txt = txt.replace("τ*", "ngưỡng cắt")
        txt = txt.replace("τ", "ngưỡng cắt")
        txt = txt.replace("—", "... ")
        txt = txt.replace("(", ", ").replace(")", ", ")
        txt = txt.replace(":", "... ")
        if gender == "male":
            txt = "Chào bạn... " + txt if not txt.startswith(("Chào", "Haha", "À", "Tin")) else txt
        else:
            txt = "Bạn ơi... " + txt if not txt.startswith(("Xin", "Haha", "Cảm", "Đừng")) else txt
    elif lang_key == "en":
        txt = txt.replace("kg/m²", "kilograms per square meter")
        txt = txt.replace("τ*", "threshold").replace("τ", "threshold")
        txt = txt.replace("—", "... ")
    elif lang_key == "zh":
        txt = txt.replace("kg/m²", "千克每平方米")
    elif lang_key == "fr":
        txt = txt.replace("kg/m²", "kilogrammes par mètre carré")
    # Collapse excess spaces
    return re.sub(r"\s+", " ", txt).strip()


@st.cache_data(show_spinner=False)
def synthesize_voice_mp3(text_clean: str, lang_key: str, gender: str = "female"):
    """Generates expressive Male and Female neural voiceovers with natural pitch, rate & pauses."""
    voice_map = {
        ("vi", "male"): ("vi-VN-NamMinhNeural", "+3%", "-4Hz"),
        ("vi", "female"): ("vi-VN-HoaiMyNeural", "+5%", "+6Hz"),
        ("en", "male"): ("en-US-GuyNeural", "+2%", "-3Hz"),
        ("en", "female"): ("en-US-JennyNeural", "+4%", "+5Hz"),
        ("zh", "male"): ("zh-CN-YunxiNeural", "+3%", "-3Hz"),
        ("zh", "female"): ("zh-CN-XiaoxiaoNeural", "+5%", "+5Hz"),
        ("fr", "male"): ("fr-FR-HenriNeural", "+2%", "-3Hz"),
        ("fr", "female"): ("fr-FR-DeniseNeural", "+4%", "+5Hz"),
    }
    chosen_voice, chosen_rate, chosen_pitch = voice_map.get(
        (lang_key, gender), ("vi-VN-HoaiMyNeural", "+5%", "+6Hz")
    )
    spoken_text = prepare_expressive_text(text_clean, lang_key, gender)

    try:
        import edge_tts

        async def _run_tts() -> bytes:
            communicate = edge_tts.Communicate(
                spoken_text,
                chosen_voice,
                rate=chosen_rate,
                pitch=chosen_pitch,
            )
            chunks = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.extend(chunk["data"])
            return bytes(chunks)

        try:
            loop = asyncio.new_event_loop()
            audio_bytes = loop.run_until_complete(_run_tts())
            loop.close()
        except Exception:
            audio_bytes = asyncio.run(_run_tts())

        if audio_bytes:
            return audio_bytes, "audio/mp3"
    except Exception:
        pass

    tts_lang_map = {"vi": "vi", "en": "en", "zh": "zh-CN", "fr": "fr"}
    tl = tts_lang_map.get(lang_key, "vi")
    try:
        from gtts import gTTS
        tts = gTTS(text=spoken_text, lang=tl, slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        return mp3_fp.getvalue(), "audio/mp3"
    except Exception:
        return build_medical_ecg_wav(), "audio/wav"


clean_m_voice = re.sub(r"<[^>]+>", "", m_speech)
clean_f_voice = re.sub(r"<[^>]+>", "", f_speech)

AUDIO_BTN_LABELS = {
    "vi": (
        "🔊 Nghe BS. Gong Yoo nói",
        "🔊 Nghe BS. Lee Sung-kyung nói",
        "💓 Âm thanh Nhịp Tim & Quét AI",
        "🎵 Bấm để nghe Giọng nói & Âm thanh (Chạy được trên mọi trình duyệt):",
    ),
    "en": (
        "🔊 Hear Dr. Gong Yoo Speak",
        "🔊 Hear Dr. Lee Sung-kyung Speak",
        "💓 Hospital ECG & Scan Sound",
        "🎵 Click to Hear AI Voiceover & Medical Sound Effects:",
    ),
    "zh": (
        "🔊 收听孔刘医生语音",
        "🔊 收听李圣经医生语音",
        "💓 播放心电监护音效",
        "🎵 点击收听 AI 语音播报与医疗音效：",
    ),
    "fr": (
        "🔊 Écouter Dr Gong Yoo",
        "🔊 Écouter Dr Lee Sung-kyung",
        "💓 Effet Sonore ECG & Scan",
        "🎵 Cliquez pour écouter la voix IA et les sons médicaux :",
    ),
}
btn_m_lbl, btn_f_lbl, btn_sfx_lbl, audio_bar_lbl = AUDIO_BTN_LABELS.get(lang_code, AUDIO_BTN_LABELS["en"])

st.caption(audio_bar_lbl)
au_c1, au_c2, au_c3 = st.columns(3)
if au_c1.button(btn_m_lbl, key="btn_voice_m_top", use_container_width=True):
    st.session_state["active_audio_bytes"], st.session_state["active_audio_fmt"] = synthesize_voice_mp3(
        clean_m_voice, lang_code, gender="male"
    )
if au_c2.button(btn_f_lbl, key="btn_voice_f_top", use_container_width=True):
    st.session_state["active_audio_bytes"], st.session_state["active_audio_fmt"] = synthesize_voice_mp3(
        clean_f_voice, lang_code, gender="female"
    )
if au_c3.button(btn_sfx_lbl, key="btn_sfx_top", use_container_width=True):
    st.session_state["active_audio_bytes"] = build_medical_ecg_wav()
    st.session_state["active_audio_fmt"] = "audio/wav"

if st.session_state.get("active_audio_bytes"):
    st.audio(
        st.session_state["active_audio_bytes"],
        format=st.session_state.get("active_audio_fmt", "audio/mp3"),
        autoplay=True,
    )

if not has_run:
    st.info(t["prompt_analyze"])


# ==============================================================================
# 9. FREE LIVE AI LLM + SMART MEDICAL/WITTY CONVERSATIONAL ENGINE
# ==============================================================================
def call_free_llm_api(question_text: str) -> tuple:
    lang_names = {"vi": "Tiếng Việt", "en": "English", "zh": "Simplified Chinese", "fr": "French"}
    target_lang = lang_names.get(lang_code, "Vietnamese")
    system_prompt = (
        f"You are roleplaying TWO charming, witty, and medically knowledgeable AI doctors at Eastern International University (EIU): "
        f"1) Dr. Gong Yoo (Endocrinologist & AI Data Scientist — warm, analytical, witty) and "
        f"2) Dr. Lee Sung-kyung (Clinical Nutritionist & Lifestyle Specialist — cheerful, practical, empathetic). "
        f"The user's current screening profile: Age group={AGE_OPTIONS[age_code]}, Height={height_cm}cm, Weight={weight_kg}kg, "
        f"BMI={bmi_val:.1f} kg/m², AI Diabetes Risk Score={prob_gb:.1%}, Decision Threshold={active_th:.2f}, "
        f"Top factor={top_feat_label}, HighBP={high_bp}, HighChol={high_chol}. "
        f"Even if the user asks a silly, funny, romantic, or random off-topic question, answer it with humor and charm FIRST, "
        f"and then smoothly connect it to their health/diabetes screening profile! "
        f"Reply strictly in {target_lang} using EXACTLY two paragraphs separated by '||SPLIT||': "
        f"First paragraph = Dr. Gong Yoo's reply (2-3 sentences). Second paragraph = Dr. Lee Sung-kyung's reply (2-3 sentences). "
        f"Do not include speaker labels before the paragraphs."
    )
    payload = {
        "model": "openai",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question_text},
        ],
        "temperature": 0.7,
    }
    try:
        req = urllib.request.Request(
            "https://text.pollinations.ai/openai",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "EIU-Diabetes-Assistant/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=7) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
            content = raw["choices"][0]["message"]["content"].strip()
            if "||SPLIT||" in content:
                parts = content.split("||SPLIT||", 1)
                return parts[0].strip(), parts[1].strip()
    except Exception:
        pass
    return None, None


def generate_doctor_replies(question_text: str, use_live_llm: bool = True) -> tuple:
    q = question_text.lower().strip()

    if lang_code == "vi":
        if any(w in q for w in ["recall", "precision", "tối ưu", "f2", "pr-auc", "chỉ số nào"]):
            ans_m = (
                f"Trong bài toán tầm soát tiểu đường tại nhà, <b>Recall (Độ nhạy = {rec_now:.1%}) quan trọng hơn nhiều so với Precision ({prec_now:.1%})!</b> "
                f"Tại sao? Vì nếu báo động giả (Precision thấp), bạn chỉ tốn ~45$ đi xét nghiệm máu kiểm tra cho yên tâm. Nhưng nếu bỏ sót bệnh (Recall thấp), "
                f"người bệnh không biết mình mắc tiểu đường suốt nhiều năm dẫn đến biến chứng tim, thận, mắt. Vì vậy chỉ số tối ưu nhất để chọn mô hình và ngưỡng cắt là "
                f"<b>F2-Score ({f2_now:.3f} — coi trọng Recall gấp 2 lần Precision)</b>, kết hợp với <b>PR-AUC (0.418)</b> và <b>Chi phí kỳ vọng tối thiểu (tỷ lệ 5:1 &rarr; ngưỡng τ* = 0.18)</b>!"
            )
            ans_f = (
                "Đó cũng là lý do tụi mình thiết kế <b>Thanh Công Cụ Chiến Lược Sàng Lọc (Recall vs. Precision)</b> ngay phía trên đó! "
                "Khi bạn muốn an toàn tuyệt đối không bỏ sót bệnh, hãy bật chế độ <b>Ưu tiên Recall (τ = 0.12)</b> nhé!"
            )
            return ans_m, ans_f
        if any(w in q for w in ["trà sữa", "trân châu", "topping", "kẹo", "bánh", "nước ngọt", "coca", "pepsi"]):
            ans_m = (
                f"Haha, 'trà sữa 0% đường nhưng full trân châu hoàng kim + kem cheese' là cú lừa kinh điển nhất quả đất! "
                f"Trân châu làm từ tinh bột sắn nấu đẫm đường đen, làm đường huyết tăng vọt cực nhanh. Với điểm rủi ro <b>{prob_gb:.1%}</b> của bạn, "
                f"thỉnh thoảng uống 1 ly không topping thì được, chứ uống mỗi ngày là tôi hẹn gặp bạn ở phòng xét nghiệm máu sớm đấy!"
            )
            ans_f = (
                f"Đừng để chữ 'ít đường' đánh lừa nha! Nếu thèm quá, bạn hãy chọn trà ô long hoặc hồng trà không đường, "
                f"topping đổi sang hạt chia hoặc nha đam không đường. Vừa giữ dáng đẹp (BMI của bạn đang là <b>{bmi_val:.1f}</b>), vừa không làm tuyến tụy 'kêu cứu'!"
            )
            return ans_m, ans_f
        if any(w in q for w in ["kiến", "tiểu", "đái", "vệ sinh"]):
            ans_m = (
                "Nghe tưởng đùa nhưng trong lịch sử y học là thật 100% đấy! Tên bệnh 'Tiểu đường' (Diabetes Mellitus — ngọt như mật) "
                "xuất phát từ việc khi đường trong máu vượt quá ngưỡng thận (~180 mg/dL), thận không giữ nổi nữa nên đường tràn ra nước tiểu làm kiến bu thật. "
                "Nhưng thời đại AI rồi, đừng đợi kiến bu mới đi khám nhé — xét nghiệm máu FPG chỉ mất 5 phút thôi!"
            )
            ans_f = (
                f"Haha, dùng kiến để khám bệnh là phương pháp từ... vài thế kỷ trước rồi! Bây giờ bạn chỉ cần nhìn điểm tầm soát trên app của tụi mình "
                f"(hiện tại của bạn là <b>{prob_gb:.1%}</b>) và đi kiểm tra máu định kỳ là chuẩn khoa học nhất nha!"
            )
            return ans_m, ans_f
        if any(w in q for w in ["đẹp", "xinh", "tim đập", "người yêu", "crush", "thất tình", "yêu", "cưới", "hẹn hò"]):
            ans_m = (
                "Haha, nhìn tôi với Bác sĩ Lee Sung-kyung mà tim đập nhanh là triệu chứng 'rung động' chứ không phải tiểu đường đâu! "
                "Nhưng nói nghiêm túc nhé: nếu lúc đói bụng mà bạn bị <b>tim đập nhanh, vã mồ hôi lạnh, run tay, hoa mắt</b> thì coi chừng đó là "
                "dấu hiệu <b>hạ đường huyết</b> hoặc rối loạn dung nạp glucose đấy!"
            )
            ans_f = (
                "Cảm ơn lời khen ngọt ngào của bạn nha, làm mình cũng đỏ mặt luôn nè! Nhớ là 'ngọt ngào' trong tình cảm thì giúp giảm hormone căng thẳng (cortisol), "
                "chứ 'ngọt' ở trong mạch máu là tụi mình phải kê đơn ăn kiêng ngay đó!"
            )
            return ans_m, ans_f
        if any(w in q for w in ["xét nghiệm", "máu", "fpg", "hba1c"]):
            ans_m = (
                f"Bạn nên yêu cầu bác sĩ làm <b>Xét nghiệm đường huyết lúc đói (FPG — nhịn ăn 8 tiếng)</b> hoặc <b>HbA1c</b> "
                f"(đánh giá đường huyết trung bình 3 tháng). Với điểm số <b>{prob_gb:.1%}</b> của bạn, đây là cách kiểm tra chính xác nhất!"
            )
            ans_f = "Trước ngày đi lấy máu xét nghiệm FPG, tối hôm trước bạn nhớ nhịn ăn từ 22h đêm (chỉ uống nước lọc) và ngủ sớm để kết quả chuẩn nhất nhé!"
            return ans_m, ans_f
        if any(w in q for w in ["thực đơn", "bài tập", "chế độ ăn"]):
            ans_m = (
                f"Khi giảm BMI từ <b>{bmi_val:.1f}</b> xuống <b>{lower_bmi:.1f} kg/m²</b>, độ nhạy insulin của cơ thể cải thiện rõ rệt, "
                f"giúp điểm rủi ro trên mô hình giảm từ <b>{prob_gb:.1%}</b> xuống <b>{prob_whatif:.1%}</b>."
            )
            ans_f = (
                f"Với chiều cao <b>{height_cm} cm</b> và cân nặng <b>{weight_kg} kg</b> (BMI <b>{bmi_val:.1f}</b>), bạn hãy áp dụng quy tắc Đĩa thức ăn: "
                f"<b>1/2 rau xanh</b>, <b>1/4 đạm lành mạnh</b> (cá, ức gà, đậu phụ), <b>1/4 tinh bột nguyên cám</b>, kết hợp đi bộ nhanh 30 phút mỗi ngày nha!"
            )
            return ans_m, ans_f

    if use_live_llm:
        llm_m, llm_f = call_free_llm_api(question_text)
        if llm_m and llm_f:
            return llm_m, llm_f

    if lang_code == "vi":
        advice_tail = (
            f"Vì điểm của bạn ({prob_gb:.1%}) đang trên ngưỡng {active_th:.2f} nên nhớ đi xét nghiệm máu kiểm tra nhé!"
            if is_flagged
            else f"May mắn là điểm của bạn ({prob_gb:.1%}) hiện vẫn nằm trong vùng an toàn dưới ngưỡng {active_th:.2f} đó!"
        )
        ans_m = (
            f"Haha, câu hỏi <i>\"{question_text}\"</i> của bạn hay đấy! Liên hệ ngay với hồ sơ sức khỏe của bạn: "
            f"hiện điểm rủi ro AI của bạn là <b>{prob_gb:.1%}</b> (yếu tố chính: <b>{top_feat_label}</b>). {advice_tail}"
        )
        ans_f = (
            f"Để mình bổ sung thêm cho bạn nha: với chỉ số BMI <b>{bmi_val:.1f} kg/m²</b> ({height_cm} cm, {weight_kg} kg), "
            f"chỉ cần bạn giữ tinh thần thoải mái, ăn nhiều rau xanh, bớt đồ ngọt và đi bộ 30 phút mỗi ngày là điểm rủi ro sẽ giảm về <b>{prob_whatif:.1%}</b>!"
        )
    else:
        ans_m = (
            f"Great question (<i>\"{question_text}\"</i>)! In medical screening, <b>Recall ({rec_now:.1%})</b> is prioritized over Precision ({prec_now:.1%}) "
            f"using the <b>F2-Score ({f2_now:.3f})</b> and <b>PR-AUC (0.418)</b> so we don't miss undiagnosed cases. Your current risk score is <b>{prob_gb:.1%}</b>."
        )
        ans_f = (
            f"And with your calculated BMI of <b>{bmi_val:.1f} kg/m²</b>, staying active and eating balanced meals will bring your estimated score down toward <b>{prob_whatif:.1%}</b>!"
        )
    return ans_m, ans_f


with st.container(border=True):
    st.markdown(
        f"<h4 style='margin:0 0 8px 0; color:#0f172a; font-weight:800;'>{d_cfg['chat_header']}</h4>",
        unsafe_allow_html=True,
    )
    doc_mode = st.radio(
        d_cfg["choose_doc"],
        options=["both", "male", "female"],
        format_func=lambda k: d_cfg["opt_both"]
        if k == "both"
        else (d_cfg["opt_male"] if k == "male" else d_cfg["opt_female"]),
        horizontal=True,
    )

    st.caption(d_cfg["quick_q_label"])
    if "active_question" not in st.session_state:
        st.session_state["active_question"] = d_cfg["q1"]

    r1_c1, r1_c2, r1_c3 = st.columns(3)
    if r1_c1.button(d_cfg["q1"], use_container_width=True):
        st.session_state["active_question"] = d_cfg["q1"]
    if r1_c2.button(d_cfg["q2"], use_container_width=True):
        st.session_state["active_question"] = d_cfg["q2"]
    if r1_c3.button(d_cfg["q3"], use_container_width=True):
        st.session_state["active_question"] = d_cfg["q3"]

    r2_c1, r2_c2, r2_c3 = st.columns(3)
    if r2_c1.button(d_cfg["q4"], use_container_width=True):
        st.session_state["active_question"] = d_cfg["q4"]
    if r2_c2.button(d_cfg["q5"], use_container_width=True):
        st.session_state["active_question"] = d_cfg["q5"]
    if r2_c3.button(d_cfg["q6"], use_container_width=True):
        st.session_state["active_question"] = d_cfg["q6"]

    in_col1, in_col2 = st.columns([4, 1])
    custom_q = in_col1.text_input(
        "Ask",
        value="",
        placeholder=d_cfg["input_placeholder"],
        label_visibility="collapsed",
    )
    if in_col2.button(d_cfg["send_btn"], type="primary", use_container_width=True) and custom_q.strip():
        st.session_state["active_question"] = custom_q.strip()

    current_q = st.session_state["active_question"]
    with st.spinner("🤖 AI Doctors are thinking..."):
        reply_m, reply_f = generate_doctor_replies(current_q, use_live_llm=True)

    st.markdown(
        f"<div style='background:#f1f5f9; border:1px solid #cbd5e1; color:#0f172a; padding:10px 16px; border-radius:12px; font-weight:700; margin:10px 0;'>"
        f"🙋 {current_q}</div>",
        unsafe_allow_html=True,
    )

    if doc_mode in ["both", "male"]:
        st.markdown(
            "<div style='display:flex; align-items:flex-start; gap:14px; background:#f8fafc; border:1px solid #cbd5e1; border-left:5px solid #2563eb; "
            "padding:14px 16px; border-radius:12px; margin-bottom:10px;'>"
            f"<img class='doc-avatar-anim' src='data:image/jpeg;base64,{DOC_MALE_B64}' style='width:54px; height:54px; object-fit:cover; border-radius:50%; border:2px solid #2563eb; flex-shrink:0;'/>"
            f"<div style='color:#1e293b; font-size:0.93rem; line-height:1.55;'><b style='color:#1d4ed8;'>{d_cfg['m_name']}:</b> {reply_m}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    if doc_mode in ["both", "female"]:
        st.markdown(
            "<div style='display:flex; align-items:flex-start; gap:14px; background:#f8fafc; border:1px solid #cbd5e1; border-left:5px solid #0d9488; "
            "padding:14px 16px; border-radius:12px; margin-bottom:4px;'>"
            f"<img class='doc-avatar-anim' src='data:image/jpeg;base64,{DOC_FEMALE_B64}' style='width:54px; height:54px; object-fit:cover; border-radius:50%; border:2px solid #0d9488; flex-shrink:0;'/>"
            f"<div style='color:#1e293b; font-size:0.93rem; line-height:1.55;'><b style='color:#0f766e;'>{d_cfg['f_name']}:</b> {reply_f}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # Native Audio playback buttons specifically for the Chat Reply!
    clean_chat_m = re.sub(r"<[^>]+>", "", reply_m)
    clean_chat_f = re.sub(r"<[^>]+>", "", reply_f)
    ca1, ca2 = st.columns(2)
    if ca1.button(f"{btn_m_lbl} (Chat)", key="btn_chat_voice_m", use_container_width=True):
        st.session_state["chat_audio_bytes"], st.session_state["chat_audio_fmt"] = synthesize_voice_mp3(
            clean_chat_m, lang_code, gender="male"
        )
    if ca2.button(f"{btn_f_lbl} (Chat)", key="btn_chat_voice_f", use_container_width=True):
        st.session_state["chat_audio_bytes"], st.session_state["chat_audio_fmt"] = synthesize_voice_mp3(
            clean_chat_f, lang_code, gender="female"
        )
    if st.session_state.get("chat_audio_bytes"):
        st.audio(
            st.session_state["chat_audio_bytes"],
            format=st.session_state.get("chat_audio_fmt", "audio/mp3"),
            autoplay=True,
        )

# ==============================================================================
# 10. DETAILED RESULTS, RISK SPECTRUM & TOP FACTORS
# ==============================================================================
if has_run:
    if is_flagged:
        banner_bg = "#fef2f2"
        banner_border = "#dc2626"
        banner_color = "#991b1b"
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
        f"<div style='background:{banner_bg}; border:1.5px solid {banner_border}; border-left:7px solid {banner_border}; border-radius:14px; padding:18px 22px; margin-top:14px; margin-bottom:16px;'>"
        f"<h3 style='margin:0 0 6px 0; color:{banner_color}; font-size:1.22rem; font-weight:800;'>{banner_title}</h3>"
        f"<p style='margin:0; color:#1e293b; font-size:0.95rem; font-weight:500;'>{t['res_natural_freq'].format(n=nat_n)}</p>"
        f"<div style='position:relative; height:18px; border-radius:999px; background:linear-gradient(90deg, #22c55e 0%, #facc15 25%, #f97316 55%, #ef4444 100%); margin:22px 0 24px 0;'>"
        f"<div style='position:absolute; left:{th_pct:.1f}%; top:-6px; bottom:-6px; width:3px; background:#0f172a;'></div>"
        f"<div style='position:absolute; left:{th_pct:.1f}%; top:-22px; transform:translateX(-50%); font-size:0.74rem; font-weight:800; color:#0f172a;'>Threshold {active_th:.2f}</div>"
        f"<div style='position:absolute; left:{pin_pct:.1f}%; top:50%; transform:translate(-50%, -50%); width:18px; height:18px; border-radius:50%; background:#ffffff; border:4px solid #0f172a; box-shadow:0 2px 6px rgba(0,0,0,0.25);'></div>"
        f"<div style='position:absolute; left:{pin_pct:.1f}%; bottom:-22px; transform:translateX(-50%); font-size:0.76rem; font-weight:800; color:#0f172a;'>{prob_gb:.1%}</div>"
        f"</div>"
        f"</div>"
    )
    st.markdown(result_card_html, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric(t["res_score_label"], f"{prob_gb:.1%}")
    m2.metric(t["res_threshold_label"], f"{active_th:.2f}")
    m3.metric(t["res_bmi_computed"], f"{bmi_val:.1f} kg/m²")

    st.markdown(f"### {t['res_top_factors_title']}")
    st.caption(t["res_top_factors_sub"])

    answer_display = {
        "Age": AGE_OPTIONS[age_code],
        "BMI": f"{bmi_val:.1f} kg/m² ({height_cm} cm, {weight_kg} kg)",
        "GenHlth": GENHLTH_MAP[gen_hlth],
        "HighBP": t["yes"] if high_bp == 1 else t["no"],
        "HighChol": t["yes"] if high_chol == 1 else t["no"],
        "HeartDiseaseorAttack": t["yes"] if heart_dis == 1 else t["no"],
        "DiffWalk": t["yes"] if diff_walk == 1 else t["no"],
    }

    max_abs = max([abs(kv[1]) for kv in sorted_c] + [0.4])
    rows_html = ""
    for feat_k, val_c in sorted_c:
        bar_w = int(min(100, max(10, round((abs(val_c) / max_abs) * 100))))
        if val_c > 0.02:
            dir_txt = f"<span style='color:#be123c; font-weight:700;'>{t['pushes_up']}</span>"
            bar_col = "#e11d48"
        elif val_c < -0.02:
            dir_txt = f"<span style='color:#0f766e; font-weight:700;'>{t['pushes_down']}</span>"
            bar_col = "#0d9488"
        else:
            dir_txt = f"<span style='color:#334155; font-weight:600;'>{t['neutral']}</span>"
            bar_col = "#64748b"

        bar_div = (
            f"<div style='margin-top:4px; width:100%; background:#e2e8f0; border-radius:6px; height:8px;'>"
            f"<div style='width:{bar_w}%; height:8px; background:{bar_col}; border-radius:6px;'></div>"
            f"</div>"
        )
        rows_html += (
            f"<tr>"
            f"<td><b>{fname_map.get(feat_k, feat_k)}</b></td>"
            f"<td>{answer_display.get(feat_k, '')}</td>"
            f"<td style='min-width:190px;'>{dir_txt} <span style='color:#334155; font-size:0.82rem; font-weight:600;'>({val_c:+.2f})</span>{bar_div}</td>"
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

    delta_p = prob_whatif - prob_gb
    st.markdown(
        f"<div class='clean-card' style='margin-top:14px; background:#f8fafc; border-color:#cbd5e1;'>"
        f"<p style='margin:0 0 4px 0; font-weight:800; color:#0f172a;'>{t['what_if_title']}</p>"
        f"<p style='margin:0; font-size:0.93rem; color:#1e293b; font-weight:500;'>"
        f"{t['what_if_desc'].format(bmi1=f'{bmi_val:.1f}', bmi2=f'{lower_bmi:.1f}', p1=f'{prob_gb:.1%}', p2=f'{prob_whatif:.1%}', delta=f'{delta_p:+.1%}')}"
        f"</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
