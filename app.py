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
    intercept = float(ref["logreg_intercept"])

    row = input_df.iloc[0]
    contributions = {}
    z_linear = intercept
    for feat, w in weights.items():
        val = float(row[feat])
        if feat in ["BMI", "Age", "GenHlth"]:
            standardized_diff = (val - float(means[feat])) / max(float(stds[feat]), 1e-6)
        else:
            standardized_diff = val - float(means[feat])
        contrib = float(w) * standardized_diff
        contributions[feat] = contrib
        z_linear += contrib

    z_safe = float(np.clip(z_linear, -30.0, 30.0))
    prob_lr = float(1.0 / (1.0 + np.exp(-z_safe)))
    synergy = 0.0
    if row["HighBP"] == 1 and row["HighChol"] == 1 and float(row["BMI"]) >= 28.0:
        synergy += 0.22
    if float(row["Age"]) >= 8 and float(row["GenHlth"]) >= 3:
        synergy += 0.14
    prob_gb = float(1.0 / (1.0 + np.exp(-float(np.clip(z_safe + synergy, -30.0, 30.0)))))

    # Prioritize direct scikit-learn pipeline predictions independently for each loaded model
    if main_pipe is not None:
        try:
            prob_gb = float(main_pipe.predict_proba(input_df)[0, 1])
        except Exception:
            pass
    if base_pipe is not None:
        try:
            prob_lr = float(base_pipe.predict_proba(input_df)[0, 1])
        except Exception:
            pass

    prob_gb = float(np.clip(prob_gb, 0.0, 1.0))
    prob_lr = float(np.clip(prob_lr, 0.0, 1.0))
    return prob_gb, prob_lr, contributions


i18n.score_single_case = score_single_case

st.set_page_config(
    page_title="Diabetes Screening Assistant | EIU",
    page_icon="🎄",
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
        "en": ("EASTERN INTERNATIONAL UNIVERSITY", "Becamex Business School", "Eastern International University (EIU) • Healthcare Decision-Support System", "Display Language", "Menu Navigation"),
        "vi": ("ĐẠI HỌC QUỐC TẾ MIỀN ĐÔNG (EIU)", "Trường Kinh doanh Becamex", "Đại học Quốc tế Miền Đông (EIU) • Hệ thống Hỗ trợ Quyết định Y tế", "Ngôn ngữ hiển thị", "Danh mục Trang"),
        "zh": ("东方国际大学 (EIU)", "Becamex 商学院", "东方国际大学 (EIU) • 医疗决策支持系统", "显示语言", "页面导航"),
        "fr": ("UNIVERSITÉ INTERNATIONALE DE L'EST", "École de Commerce Becamex", "Université Internationale de l'Est (EIU) • Système d'Aide à la Décision", "Langue d'affichage", "Navigation"),
    },
)
_NAV_I18N = getattr(
    i18n,
    "NAV_TABS_I18N",
    {
        "en": {"app": "Screening Assistant", "Threshold and Cost": "Threshold & Cost", "Health Equity": "Health Equity", "About Model": "About Model", "Batch Screening": "Batch Screening"},
        "vi": {"app": "Trợ lý Sàng lọc", "Threshold and Cost": "Ngưỡng cắt & Chi phí", "Health Equity": "Công bằng Y tế", "About Model": "Về Mô hình AI", "Batch Screening": "Tầm soát Hàng loạt"},
        "zh": {"app": "糖尿病筛查助手", "Threshold and Cost": "阈值与成本模拟", "Health Equity": "健康公平性评估", "About Model": "关于模型与数据", "Batch Screening": "批量筛查 (CSV)"},
        "fr": {"app": "Assistant de Dépistage", "Threshold and Cost": "Seuil et Coûts", "Health Equity": "Équité en Santé", "About Model": "À propos du Modèle", "Batch Screening": "Dépistage par Lot"},
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
_active_theme = st.session_state.get("active_theme", "xmas")
_theme_i18n = getattr(i18n, "THEME_I18N", {}).get(lang_code, {})

if _active_theme == "xmas":
    _xmas_banner_msg = _theme_i18n.get("xmas_banner", "✨ Merry Christmas • Wishing you a warm & healthy holiday season! 🎁")
    header_html = (
        "<div style='background:linear-gradient(135deg, #ffffff 0%, #fff1f2 55%, #ecfdf5 100%); "
        "border:1.5px solid #fecdd3; border-radius:18px; padding:16px 26px 18px 26px; margin-bottom:14px; "
        "box-shadow:0 6px 18px rgba(220,38,38,0.08);'>"
        "<div class='xmas-candy-strip'></div>"
        "<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;'>"
        "<div style='flex:1; min-width:280px;'>"
        "<div style='display:inline-block; background:#fef2f2; border:1px solid #fca5a5; color:#b91c1c; "
        "font-size:0.76rem; font-weight:700; padding:3px 11px; border-radius:999px; margin-bottom:7px;'>"
        f"{_xmas_banner_msg}"
        "</div>"
        f"<div style='font-size:0.78rem; font-weight:700; color:#15803d; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;'>"
        f"🎄 {_header_badge_txt}"
        "</div>"
        f"<h1 style='font-size:2.05rem; font-weight:800; color:#991b1b; margin:0;'>🎅 {t['app_title']} ❄️</h1>"
        "</div>"
        "<div style='text-align:right;'>"
        f"<img src='data:image/png;base64,{EIU_LOGO_B64}' style='max-height:60px; width:auto;'/>"
        "<div style='font-size:1.15rem; margin-top:4px; letter-spacing:4px;'>🎄⛄🎁✨</div>"
        "</div>"
        "</div>"
        "</div>"
    )
elif _active_theme == "tet":
    _tet_banner_msg = _theme_i18n.get("tet_banner", "✨ Chúc Mừng Năm Mới • Vạn Sự Như Ý & Dồi Dào Sức Khỏe! 🧧🌸")
    header_html = (
        "<div style='background:linear-gradient(135deg, #ffffff 0%, #fffbeb 52%, #fef2f2 100%); "
        "border:1.5px solid #fcd34d; border-radius:18px; padding:16px 26px 18px 26px; margin-bottom:14px; "
        "box-shadow:0 6px 18px rgba(185,28,28,0.09);'>"
        "<div class='tet-gold-strip'></div>"
        "<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;'>"
        "<div style='flex:1; min-width:280px;'>"
        "<div style='display:inline-block; background:#fef2f2; border:1px solid #fbbf24; color:#b91c1c; "
        "font-size:0.76rem; font-weight:700; padding:3px 11px; border-radius:999px; margin-bottom:7px;'>"
        f"{_tet_banner_msg}"
        "</div>"
        f"<div style='font-size:0.78rem; font-weight:700; color:#b45309; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;'>"
        f"🏮 {_header_badge_txt}"
        "</div>"
        f"<h1 style='font-size:2.05rem; font-weight:800; color:#991b1b; margin:0;'>🧧 {t['app_title']} 🌸</h1>"
        "</div>"
        "<div style='text-align:right;'>"
        f"<img src='data:image/png;base64,{EIU_LOGO_B64}' style='max-height:60px; width:auto;'/>"
        "<div style='font-size:1.15rem; margin-top:4px; letter-spacing:4px;'>🌸🌼🧧🏮</div>"
        "</div>"
        "</div>"
        "</div>"
    )
else:
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
# 3. COMPUTE SCORE & TRIGGER ANIMATED ACTIONS WHEN ANALYZED
# ==============================================================================
active_th = float(st.session_state.get("active_threshold", default_th))
pt_now = min(curve_list, key=lambda r: abs(r["threshold"] - active_th))
rec_now = float(pt_now["test_recall"])
prec_now = float(pt_now["test_precision"])
f2_now = (5 * prec_now * rec_now / (4 * prec_now + rec_now)) if (4 * prec_now + rec_now) > 0 else 0.0
_app_metrics = meta.get("app_model_test_metrics", {})
pr_auc_live = float(_app_metrics.get("pr_auc_test", 0.418))
roc_auc_live = float(_app_metrics.get("roc_auc_test", 0.822))
cost_fp_live = float(meta.get("cost_assumptions", {}).get("cost_false_positive_usd", 45.0))

safe_height_m = max(0.5, float(height_cm) / 100.0)
safe_weight_kg = max(10.0, float(weight_kg))
raw_bmi = round(safe_weight_kg / (safe_height_m ** 2), 1)
bmi_val = float(np.clip(raw_bmi, 12.0, 70.0))
if raw_bmi < 12.0 or raw_bmi > 65.0:
    _bmi_warn = {
        "vi": f"⚠️ Chiều cao ({height_cm} cm) và cân nặng ({weight_kg} kg) cho ra BMI cực trị ({raw_bmi:.1f} kg/m²). Hệ thống tự động giới hạn trong ngưỡng hợp lệ ({bmi_val:.1f}) để đảm bảo mô hình chạy ổn định — vui lòng kiểm tra lại số đo nhé.",
        "en": f"⚠️ Extreme height ({height_cm} cm) / weight ({weight_kg} kg) gave BMI {raw_bmi:.1f} kg/m². Clamped safely to {bmi_val:.1f} for stable prediction — please double-check your entries.",
        "zh": f"⚠️ 输入的身高 ({height_cm} cm) 与体重 ({weight_kg} kg) 计算出极端 BMI ({raw_bmi:.1f})，系统已安全限制为 {bmi_val:.1f}，请核对输入数值。",
        "fr": f"⚠️ Taille ({height_cm} cm) / poids ({weight_kg} kg) extrêmes (IMC {raw_bmi:.1f}). Valeur bornée à {bmi_val:.1f} pour la stabilité — vérifiez vos mesures.",
    }
    st.warning(_bmi_warn.get(lang_code, _bmi_warn["en"]))
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
# 5. DUAL AI DOCTORS (ORIGINAL NAMES PER COUNTRY) + WHITE & MEDICAL BLUE THEME
# ==============================================================================
DOC_TEXT = {
    "vi": {
        "section_title": "🩺 Phòng Hội Chẩn & Giao Lưu Cùng BS. Minh Quân & BS. Thảo Vy",
        "m_name": "👨‍⚕️ BS. Minh Quân (Chuyên khoa Nội tiết & AI)",
        "f_name": "👩‍⚕️ BS. Thảo Vy (Chuyên khoa Dinh dưỡng & Lối sống)",
        "m_idle": "Chào bạn! Tôi là <b>BS. Minh Quân</b>. Tôi sẵn sàng hỗ trợ kiểm tra sức khỏe và giải đáp mọi thắc mắc của bạn.",
        "f_idle": "Xin chào! Mình là <b>BS. Thảo Vy</b>. Bạn cần tư vấn sức khỏe, dinh dưỡng hay hỏi bất kỳ thông tin gì cứ nhắn bên dưới nhé!",
        "m_flag": f"Điểm tương đồng của bạn là <b>{prob_gb:.1%}</b> (vượt ngưỡng tầm soát <b>{active_th:.2f}</b>). Yếu tố nổi bật nhất là <b>{top_feat_label}</b>. Bạn nên đặt lịch làm xét nghiệm <b>Đường huyết lúc đói (FPG)</b> hoặc <b>HbA1c</b> để kiểm tra nhé.",
        "m_safe": f"Tin vui! Điểm rủi ro của bạn chỉ là <b>{prob_gb:.1%}</b> (thấp hơn ngưỡng cảnh báo <b>{active_th:.2f}</b>). Bạn chưa cần xét nghiệm khẩn cấp hôm nay, chỉ cần duy trì khám sức khỏe định kỳ.",
        "f_flag": f"BMI hiện tại của bạn là <b>{bmi_val:.1f} kg/m²</b>. Nếu bạn giảm nhẹ BMI xuống <b>{lower_bmi:.1f} kg/m²</b> kết hợp đi bộ 30 phút/ngày, điểm rủi ro sẽ giảm xuống còn <b>{prob_whatif:.1%}</b>!",
        "f_safe": f"BMI của bạn là <b>{bmi_val:.1f} kg/m²</b>. Để giữ mức rủi ro thấp (<b>{prob_whatif:.1%}</b>), nhớ ăn nhiều rau xanh, bớt trà sữa full topping và ngủ đủ giấc nha!",
        "chat_header": "💬 Chat Trực Tiếp Cùng AI BS. Minh Quân & BS. Thảo Vy",
        "choose_doc": "Chọn bác sĩ bạn muốn tương tác:",
        "opt_both": "👨‍⚕️👩‍⚕️ Cả 2 Bác sĩ (Minh Quân & Thảo Vy)",
        "opt_male": "👨‍⚕️ Chỉ hỏi BS. Minh Quân",
        "opt_female": "👩‍⚕️ Chỉ hỏi BS. Thảo Vy",
        "quick_q_label": "Bấm thử các câu hỏi mẫu hoặc tự gõ bất kỳ câu hỏi nào (kể cả hỏi nhảm/vui) vào ô chat bên dưới:",
        "q1": "Chỉ số đường huyết lúc đói và HbA1c bao nhiêu là bình thường?",
        "q2": "Tiền tiểu đường có thể đảo ngược bằng ăn uống và tập luyện không?",
        "q3": "Cao huyết áp và mỡ máu cao làm tăng nguy cơ tiểu đường thế nào?",
        "q4": "Người thừa cân nên ăn uống và vận động ra sao để ngừa tiểu đường?",
        "q5": "Khi đi khám tầm soát tiểu đường tôi nên yêu cầu xét nghiệm gì?",
        "q6": "Vì sao mô hình tầm soát ưu tiên độ nhạy (Recall) hơn Precision?",
        "input_placeholder": "Nhập câu hỏi của bạn (VD: Chỉ số HbA1c bao nhiêu là cao, chế độ ăn cho người tiền tiểu đường...)",
        "send_btn": "Gửi cho AI",
    },
    "en": {
        "section_title": "🩺 AI Medical Consultation with Dr. Ethan Brooks & Dr. Clara Bennett",
        "m_name": "👨‍⚕️ Dr. Ethan Brooks (Endocrinology & AI Specialist)",
        "f_name": "👩‍⚕️ Dr. Clara Bennett (Nutrition & Lifestyle Specialist)",
        "m_idle": "Hello! I'm <b>Dr. Ethan Brooks</b>. I'm here to help with your health screening and answer any question you have.",
        "f_idle": "Hi! I'm <b>Dr. Clara Bennett</b>. Feel free to ask about nutrition, lifestyle, or anything on your mind below!",
        "m_flag": f"Your estimated risk score is <b>{prob_gb:.1%}</b> (above our <b>{active_th:.2f}</b> cutoff; top factor: <b>{top_feat_label}</b>). I recommend asking your doctor for a routine <b>Fasting Plasma Glucose (FPG)</b> or <b>HbA1c</b> test.",
        "m_safe": f"Good news! Your estimated score is <b>{prob_gb:.1%}</b> (below our <b>{active_th:.2f}</b> cutoff). No urgent lab test is flagged today.",
        "f_flag": f"Your BMI is <b>{bmi_val:.1f} kg/m²</b>. Lowering your BMI toward <b>{lower_bmi:.1f} kg/m²</b> with 30 minutes of daily walking reduces your risk score to <b>{prob_whatif:.1%}</b>!",
        "f_safe": f"Your BMI is <b>{bmi_val:.1f} kg/m²</b>. Keep enjoying balanced meals and daily activity to maintain a low risk of <b>{prob_whatif:.1%}</b>!",
        "chat_header": "💬 Live AI Chat with Dr. Ethan Brooks & Dr. Clara Bennett",
        "choose_doc": "Choose which doctor to talk to:",
        "opt_both": "👨‍⚕️👩‍⚕️ Both Doctors (Ethan Brooks & Clara Bennett)",
        "opt_male": "👨‍⚕️ Dr. Ethan Brooks Only",
        "opt_female": "👩‍⚕️ Dr. Clara Bennett Only",
        "quick_q_label": "Try a quick question or type anything into the live AI chat below:",
        "q1": "What are normal Fasting Blood Glucose (FPG) and HbA1c levels?",
        "q2": "Can prediabetes be reversed through diet and regular exercise?",
        "q3": "How do high blood pressure and high cholesterol raise diabetes risk?",
        "q4": "What daily meal plan and exercise help lower diabetes risk?",
        "q5": "Which blood tests should I ask my doctor for diabetes screening?",
        "q6": "Why does medical screening prioritize Recall over Precision?",
        "input_placeholder": "Type your question (e.g. Normal HbA1c range, diet tips to prevent diabetes...)",
        "send_btn": "Ask AI",
    },
    "zh": {
        "section_title": "🩺 与林明轩医生 (Dr. Lin) & 苏婉清医生 (Dr. Su) 互动会诊",
        "m_name": "👨‍⚕️ 林明轩医生（内分泌与 AI 专家）",
        "f_name": "👩‍⚕️ 苏婉清医生（营养与生活方式专家）",
        "m_idle": "您好！我是<b>林明轩医生</b>。随时为您解答健康筛查或任何日常疑问。",
        "f_idle": "您好！我是<b>苏婉清医生</b>。无论是营养建议还是任何问题，都欢迎在下方提问！",
        "m_flag": f"您的估计风险得分为 <b>{prob_gb:.1%}</b>，高于阈值 <b>{active_th:.2f}</b>（主要因素：<b>{top_feat_label}</b>）。建议咨询医生做 <b>FPG</b> 或 <b>HbA1c</b> 检测。",
        "m_safe": f"好消息！您的风险得分为 <b>{prob_gb:.1%}</b>，低于阈值 <b>{active_th:.2f}</b>。目前无需紧急检测。",
        "f_flag": f"您的 BMI 为 <b>{bmi_val:.1f} kg/m²</b>。若将 BMI 降至 <b>{lower_bmi:.1f} kg/m²</b>，风险将降至 <b>{prob_whatif:.1%}</b>！",
        "f_safe": f"您的 BMI 为 <b>{bmi_val:.1f} kg/m²</b>。请继续保持均衡饮食与适度运动！",
        "chat_header": "💬 与林明轩医生和苏婉清医生 AI 实时对话",
        "choose_doc": "选择咨询对象：",
        "opt_both": "👨‍⚕️👩‍⚕️ 两位医生联合解答",
        "opt_male": "👨‍⚕️ 仅问林明轩医生",
        "opt_female": "👩‍⚕️ 仅问苏婉清医生",
        "quick_q_label": "点击快捷问题或输入任何问题：",
        "q1": "空腹血糖和糖化血红蛋白 (HbA1c) 的正常范围是多少？",
        "q2": "糖尿病前期可以通过饮食和运动逆转吗？",
        "q3": "高血压和高胆固醇会如何增加糖尿病风险？",
        "q4": "如何通过日常饮食和锻炼有效预防糖尿病？",
        "q5": "去医院做糖尿病筛查应该要求哪些血液检查？",
        "q6": "为什么医疗筛查模型更重视召回率 (Recall)？",
        "input_placeholder": "请输入您的问题（例如：HbA1c 正常值是多少、预防糖尿病的饮食建议...）",
        "send_btn": "发送给 AI",
    },
    "fr": {
        "section_title": "🩺 Consultation Interactive avec Dr Julien Moreau & Dr Camille Laurent",
        "m_name": "👨‍⚕️ Dr Julien Moreau (Endocrinologie & IA)",
        "f_name": "👩‍⚕️ Dr Camille Laurent (Nutrition & Mode de vie)",
        "m_idle": "Bonjour ! Je suis le <b>Dr Julien Moreau</b>. Je suis là pour vous accompagner et répondre à toutes vos questions.",
        "f_idle": "Bonjour ! Je suis le <b>Dr Camille Laurent</b>. N'hésitez pas à poser toutes vos questions ci-dessous !",
        "m_flag": f"Votre score est de <b>{prob_gb:.1%}</b> (au-dessus du seuil <b>{active_th:.2f}</b> ; facteur clé : <b>{top_feat_label}</b>). Demandez un test <b>FPG</b> ou <b>HbA1c</b> à votre médecin.",
        "m_safe": f"Bonne nouvelle ! Votre score est de <b>{prob_gb:.1%}</b> (sous le seuil <b>{active_th:.2f}</b>). Aucun test urgent requis.",
        "f_flag": f"Votre IMC est de <b>{bmi_val:.1f} kg/m²</b>. En le réduisant vers <b>{lower_bmi:.1f} kg/m²</b>, votre score baisserait à <b>{prob_whatif:.1%}</b> !",
        "f_safe": f"Votre IMC est de <b>{bmi_val:.1f} kg/m²</b>. Continuez vos bonnes habitudes !",
        "chat_header": "💬 Chat IA en Direct avec Dr Julien Moreau & Dr Camille Laurent",
        "choose_doc": "Choisissez votre médecin :",
        "opt_both": "👨‍⚕️👩‍⚕️ Les 2 médecins ensemble",
        "opt_male": "👨‍⚕️ Dr Julien Moreau uniquement",
        "opt_female": "👩‍⚕️ Dr Camille Laurent uniquement",
        "quick_q_label": "Questions rapides ou libres :",
        "q1": "Quelles sont les valeurs normales de glycémie à jeun et d'HbA1c ?",
        "q2": "Le prédiabète peut-il être inversé par l'alimentation et l'exercice ?",
        "q3": "Comment l'hypertension et le cholestérol augmentent-ils le risque ?",
        "q4": "Quel régime et quelle activité physique aident à prévenir le diabète ?",
        "q5": "Quels tests sanguins demander au médecin pour le dépistage ?",
        "q6": "Pourquoi le dépistage médical privilégie-t-il le Rappel (Recall) ?",
        "input_placeholder": "Posez votre question (ex. taux normal d'HbA1c, conseils alimentaires...)",
        "send_btn": "Envoyer à l'IA",
    },
}

d_cfg = DOC_TEXT.get(lang_code, DOC_TEXT["en"])
m_speech = d_cfg["m_idle"] if not has_run else (d_cfg["m_flag"] if is_flagged else d_cfg["m_safe"])
f_speech = d_cfg["f_idle"] if not has_run else (d_cfg["f_flag"] if is_flagged else d_cfg["f_safe"])

st.markdown(
    f"<h3 style='color:#0f172a; margin-top:16px; margin-bottom:10px; font-weight:800;'>{d_cfg['section_title']}</h3>",
    unsafe_allow_html=True,
)

doc_col1, doc_col2 = st.columns(2, gap="medium")
with doc_col1:
    st.markdown(
        "<div style='display:flex; align-items:center; gap:16px; background:#ffffff; border:1.5px solid #bfdbfe; border-left:6px solid #1d4ed8; "
        "border-radius:14px; padding:16px 18px; box-shadow:0 3px 10px rgba(37,99,235,0.06); min-height:156px;'>"
        f"<img class='doc-avatar-anim' src='data:image/jpeg;base64,{DOC_MALE_B64}' style='width:92px; height:112px; object-fit:cover; border-radius:12px; border:2px solid #1d4ed8; flex-shrink:0;'/>"
        "<div>"
        f"<div style='font-weight:800; color:#1d4ed8; font-size:0.95rem; margin-bottom:6px;'><span class='live-dot'></span>{d_cfg['m_name']}</div>"
        f"<div style='color:#1e293b; font-size:0.91rem; line-height:1.55;'>{m_speech}</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

with doc_col2:
    st.markdown(
        "<div style='display:flex; align-items:center; gap:16px; background:#ffffff; border:1.5px solid #bae6fd; border-left:6px solid #0284c7; "
        "border-radius:14px; padding:16px 18px; box-shadow:0 3px 10px rgba(2,132,199,0.06); min-height:156px;'>"
        f"<img class='doc-avatar-anim' src='data:image/jpeg;base64,{DOC_FEMALE_B64}' style='width:92px; height:112px; object-fit:cover; border-radius:12px; border:2px solid #0284c7; flex-shrink:0;'/>"
        "<div>"
        f"<div style='font-weight:800; color:#0369a1; font-size:0.95rem; margin-bottom:6px;'><span class='live-dot'></span>{d_cfg['f_name']}</div>"
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
    """Formats medical symbols & adds gentle breathing pauses for warm, human-like intonation."""
    txt = text_clean.strip()
    if lang_key == "vi":
        txt = txt.replace("kg/m²", "ki-lô-gam trên mét vuông")
        txt = txt.replace("HbA1c", "Hát bê A một xê")
        txt = txt.replace("FPG", "đường huyết lúc đói")
        txt = txt.replace("PR-AUC", "Pê Rờ A U Cê")
        txt = txt.replace("F2-Score", "điểm Ép hai")
        txt = txt.replace("τ*", "mốc kiểm tra")
        txt = txt.replace("τ", "mốc kiểm tra")
        # Add natural micro-pauses for expressive emphasis
        txt = txt.replace("—", "... ").replace(":", "... ").replace(";", ", ")
        txt = txt.replace("(", ", ").replace(")", ", ")
        txt = txt.replace("!", "!... ")
    elif lang_key == "en":
        txt = txt.replace("kg/m²", "kilograms per square meter")
        txt = txt.replace("τ*", "threshold").replace("τ", "threshold")
        txt = txt.replace("—", "... ").replace("!", "!... ")
    elif lang_key == "zh":
        txt = txt.replace("kg/m²", "千克每平方米")
    elif lang_key == "fr":
        txt = txt.replace("kg/m²", "kilogrammes par mètre carré")
    return re.sub(r"\s+", " ", txt).strip()


@st.cache_data(show_spinner=False)
def synthesize_voice_mp3(text_clean: str, lang_key: str, gender: str = "female"):
    """
    Generates ONE calm, steady, warm, authoritative doctor voice per gender & language via edge-tts.
    """
    voice_map = {
        ("vi", "male"): ("vi-VN-NamMinhNeural", "-8%", "-6Hz"),
        ("vi", "female"): ("vi-VN-HoaiMyNeural", "-7%", "-2Hz"),
        ("en", "male"): ("en-US-AndrewMultilingualNeural", "-5%", "-3Hz"),
        ("en", "female"): ("en-US-AvaMultilingualNeural", "-4%", "+1Hz"),
        ("zh", "male"): ("zh-CN-YunjianNeural", "-6%", "-4Hz"),
        ("zh", "female"): ("zh-CN-XiaoyiNeural", "-5%", "+2Hz"),
        ("fr", "male"): ("fr-FR-RemyMultilingualNeural", "-5%", "-3Hz"),
        ("fr", "female"): ("fr-FR-VivienneMultilingualNeural", "-4%", "+1Hz"),
    }

    chosen_voice, chosen_rate, chosen_pitch = voice_map.get(
        (lang_key, gender), ("en-US-AvaMultilingualNeural", "-4%", "+1Hz")
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
    "vi": ("🔊 Nghe BS. Minh Quân nói", "🔊 Nghe BS. Thảo Vy nói"),
    "en": ("🔊 Hear Dr. Ethan Brooks", "🔊 Hear Dr. Clara Bennett"),
    "zh": ("🔊 收听林明轩医生语音", "🔊 收听苏婉清医生语音"),
    "fr": ("🔊 Écouter Dr Julien Moreau", "🔊 Écouter Dr Camille Laurent"),
}
btn_m_lbl, btn_f_lbl = AUDIO_BTN_LABELS.get(lang_code, AUDIO_BTN_LABELS["en"])

au_c1, au_c2 = st.columns(2)
if au_c1.button(btn_m_lbl, key="btn_voice_m_top", use_container_width=True):
    st.session_state["active_audio_bytes"], st.session_state["active_audio_fmt"] = synthesize_voice_mp3(
        clean_m_voice, lang_code, gender="male"
    )
if au_c2.button(btn_f_lbl, key="btn_voice_f_top", use_container_width=True):
    st.session_state["active_audio_bytes"], st.session_state["active_audio_fmt"] = synthesize_voice_mp3(
        clean_f_voice, lang_code, gender="female"
    )

if st.session_state.get("active_audio_bytes"):
    st.audio(
        st.session_state["active_audio_bytes"],
        format=st.session_state.get("active_audio_fmt", "audio/mp3"),
        autoplay=True,
    )

if not has_run:
    st.info(t["prompt_analyze"])


# ==============================================================================
# 9. FREE LIVE AI LLM + UNIVERSAL SMART/WITTY CONVERSATIONAL ENGINE
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=180)
def fetch_internet_knowledge(query: str, lang_key: str) -> str:
    """Fetches live real-time news (Google News RSS), DuckDuckGo, and Wikipedia facts."""
    import urllib.parse
    import xml.etree.ElementTree as ET

    snippets = []
    q_low = query.lower()
    news_cfg = {
        "vi": ("vi", "VN", "VN:vi"),
        "en": ("en-US", "US", "US:en"),
        "zh": ("zh-CN", "CN", "CN:zh-Hans"),
        "fr": ("fr", "FR", "FR:fr"),
    }
    hl, gl, ceid = news_cfg.get(lang_key, ("en-US", "US", "US:en"))

    # 0. DuckDuckGo HTML Web Search (Real Google-like organic web snippets for ANY topic!)
    try:
        import html as _html
        ddg_html_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        req_ddg = urllib.request.Request(
            ddg_html_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept-Language": f"{hl},en;q=0.8",
            },
        )
        with urllib.request.urlopen(req_ddg, timeout=5) as resp_ddg:
            raw_html = resp_ddg.read().decode("utf-8", errors="ignore")
            raw_snips = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', raw_html, flags=re.S)
            cleaned_snips = []
            for s in raw_snips[:4]:
                clean_s = _html.unescape(re.sub(r"<[^>]+>", "", s)).strip()
                clean_s = re.sub(r"\s+", " ", clean_s)
                if clean_s and len(clean_s) > 25:
                    cleaned_snips.append(clean_s)
            if cleaned_snips:
                snippets.append("[Kết quả tìm kiếm Web]: " + " | ".join(cleaned_snips))
    except Exception:
        pass

    # 1. Google News RSS (Live real-time news, gold price, weather, sports, events!)
    try:
        rss_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl={hl}&gl={gl}&ceid={ceid}"
        req_rss = urllib.request.Request(rss_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req_rss, timeout=4) as resp_rss:
            root = ET.fromstring(resp_rss.read())
            headlines = [it.find("title").text for it in root.findall(".//item")[:4] if it.find("title") is not None]
            if headlines:
                snippets.append("[Tin tức / Dữ liệu thời gian thực]: " + " | ".join(headlines))
    except Exception:
        pass

    # 2. DuckDuckGo Instant Answer
    try:
        ddg_url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
        req3 = urllib.request.Request(ddg_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req3, timeout=4) as resp3:
            ddata = json.loads(resp3.read().decode("utf-8"))
            abs_txt = ddata.get("AbstractText", "").strip()
            if abs_txt:
                snippets.append(f"[Web Summary]: {abs_txt[:420]}")
    except Exception:
        pass

    # 3. Wikipedia for encyclopedic / knowledge queries
    is_realtime = any(w in q_low for w in ["hôm nay", "giá", "today", "price", "weather", "thời tiết", "tỷ giá", "mấy giờ", "news", "tin tức", "xổ số", "kết quả"])
    if not is_realtime:
        wiki_lang = {"vi": "vi", "en": "en", "zh": "zh", "fr": "fr"}.get(lang_key, "en")
        try:
            search_url = (
                f"https://{wiki_lang}.wikipedia.org/w/api.php?action=query&list=search"
                f"&srsearch={urllib.parse.quote(query)}&format=json&utf8=1&srlimit=1"
            )
            req = urllib.request.Request(search_url, headers={"User-Agent": "EIU-Assistant/1.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                hits = data.get("query", {}).get("search", [])
                if hits:
                    title = hits[0]["title"]
                    sum_url = f"https://{wiki_lang}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
                    req2 = urllib.request.Request(sum_url, headers={"User-Agent": "EIU-Assistant/1.0"})
                    with urllib.request.urlopen(req2, timeout=4) as resp2:
                        sdata = json.loads(resp2.read().decode("utf-8"))
                        ext = sdata.get("extract", "").strip()
                        if ext:
                            snippets.append(f"[Wikipedia - {title}]: {ext[:450]}")
        except Exception:
            pass

    return "\n".join(snippets)


@st.cache_data(show_spinner=False, ttl=300)
def call_free_llm_api(question_text: str, lang_key: str, doc_m_label: str, doc_f_label: str) -> tuple:
    """
    Google-style Universal AI engine:
    0) Official Google Gemini 2.5 Flash (if GEMINI_API_KEY is set in st.secrets or env)
    1) Free Pollinations AI LLM (searchgpt / openai) grounded in live DuckDuckGo Web Search + Google News + Wikipedia
    2) Structured Google-style Web Search snippet synthesizer fallback (zero generic filler!)
    """
    import os
    import urllib.parse

    web_context = fetch_internet_knowledge(question_text, lang_key)
    lang_names = {"vi": "Vietnamese", "en": "English", "zh": "Simplified Chinese", "fr": "French"}
    target_lang = lang_names.get(lang_key, "Vietnamese")

    context_block = f"\nLive Web Search Results:\n{web_context}\n" if web_context else ""
    system_prompt = (
        f"You are an intelligent, Google-AI-Overview-level assistant roleplaying TWO warm, articulate, knowledgeable specialists: "
        f"1) {doc_m_label} (clear, direct, accurate, explains facts cleanly like Google Search AI Overview) and "
        f"2) {doc_f_label} (warm, practical, adds helpful tips, key takeaways, or extra context). "
        f"IMPORTANT: The user can ask ANY question on earth (general knowledge, news, math, coding, travel, food, sports, history, science, health, daily life)—"
        f"do NOT force the answer back to diabetes or the 7-question form unless the user specifically asks about diabetes/health. "
        f"Answer directly, accurately, and concisely in {target_lang}"
        f"{' grounded in the live Web Search results below' if web_context else ''}. "
        f"Format your reply as TWO concise paragraphs separated by '||SPLIT||': "
        f"Paragraph 1 = {doc_m_label}'s direct, factual Google-style answer (2-4 crisp sentences). "
        f"Paragraph 2 = {doc_f_label}'s helpful practical tip or extra detail (1-2 natural sentences). "
        f"Do not prefix paragraphs with speaker names.{context_block}"
    )

    def _split_reply(txt: str):
        txt = txt.strip()
        if not txt:
            return None, None
        if "||SPLIT||" in txt:
            p = txt.split("||SPLIT||", 1)
            return p[0].strip(), p[1].strip()
        paras = [p.strip() for p in txt.split("\n\n") if p.strip()]
        if len(paras) >= 2:
            return paras[0], " ".join(paras[1:])
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        if len(lines) >= 2:
            return lines[0], " ".join(lines[1:])
        sents = re.split(r"(?<=[.!?])\s+", txt)
        if len(sents) >= 2:
            mid = max(1, len(sents) // 2)
            return " ".join(sents[:mid]), " ".join(sents[mid:])
        return txt, (
            "Nếu bạn muốn tìm hiểu sâu hơn hoặc hỏi thêm chủ đề khác, cứ nhắn bên dưới nhé."
            if lang_key == "vi"
            else "Feel free to ask if you would like more details on this topic."
        )

    # Attempt 0: Official Google Gemini 2.5 Flash (Free tier API key via Streamlit Secrets or Env)
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        try:
            gemini_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            gemini_key = ""

    if gemini_key:
        try:
            gem_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
            gem_payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"{system_prompt}\n\nUser question: {question_text}"}],
                    }
                ],
                "generationConfig": {"temperature": 0.35, "maxOutputTokens": 500},
            }
            req_g = urllib.request.Request(
                gem_url,
                data=json.dumps(gem_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req_g, timeout=10) as resp_g:
                gdata = json.loads(resp_g.read().decode("utf-8"))
                g_text = gdata["candidates"][0]["content"]["parts"][0]["text"]
                m_ans, f_ans = _split_reply(g_text)
                if m_ans and f_ans:
                    return m_ans, f_ans
        except Exception:
            pass

    # Attempt 1: Pollinations OpenAI POST (openai / searchgpt)
    for _pmodel in ("openai", "searchgpt"):
        try:
            payload = {
                "model": _pmodel,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question_text},
                ],
                "temperature": 0.35,
            }
            req2 = urllib.request.Request(
                "https://text.pollinations.ai/openai",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                method="POST",
            )
            with urllib.request.urlopen(req2, timeout=12) as resp2:
                raw = json.loads(resp2.read().decode("utf-8"))
                content2 = raw["choices"][0]["message"]["content"]
                m_ans, f_ans = _split_reply(content2)
                if m_ans and f_ans:
                    return m_ans, f_ans
        except Exception:
            continue

    # Attempt 2: Pollinations GET (zero-key backup)
    try:
        full_prompt = f"{system_prompt}\n\nUser question: {question_text}"
        url = "https://text.pollinations.ai/" + urllib.parse.quote(full_prompt)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            content = resp.read().decode("utf-8")
            m_ans, f_ans = _split_reply(content)
            if m_ans and f_ans:
                return m_ans, f_ans
    except Exception:
        pass

    # Attempt 3: Clean Google-style synthesized answer directly from live Web Search / News / Wikipedia
    if web_context:
        parts = [re.sub(r"^\[[^\]]+\]:\s*", "", ln).strip() for ln in web_context.splitlines() if ln.strip()]
        primary_info = parts[0][:460] if parts else ""
        extra_info = parts[1][:260] if len(parts) > 1 else ""
        if lang_key == "vi":
            return (
                f"Theo kết quả tra cứu trực tuyến mới nhất: {primary_info}",
                f"Thông tin bổ sung: {extra_info}" if extra_info else "Bạn có thể đặt câu hỏi chi tiết hơn để mình tra cứu chính xác hơn nhé!",
            )
        return (
            f"Based on live web search results: {primary_info}",
            f"Additional context: {extra_info}" if extra_info else "Feel free to ask a more specific follow-up question!",
        )

    return None, None


def generate_doctor_replies(question_text: str, use_live_llm: bool = True) -> tuple:
    q = question_text.lower().strip()

    # Only keep exact app-specific medical/metric shortcuts; send ALL other custom questions to live Web Search + AI!
    if lang_code == "vi":
        if any(w in q for w in ["recall", "precision", "f2", "pr-auc"]):
            ans_m = (
                f"Trong bài toán tầm soát tiểu đường tại nhà (ngưỡng hiện tại <b>{active_th:.2f}</b>), "
                f"<b>Recall (Độ nhạy = {rec_now:.1%}) quan trọng hơn nhiều so với Precision ({prec_now:.1%})!</b> "
                f"Vì báo động giả chỉ tốn khoảng ~${cost_fp_live:.0f} xét nghiệm máu kiểm tra, còn bỏ sót bệnh gây biến chứng nặng. "
                f"Chỉ số tối ưu là <b>F2-Score ({f2_now:.3f})</b>, <b>PR-AUC ({pr_auc_live:.3f})</b> và <b>ROC-AUC ({roc_auc_live:.3f})</b>."
            )
            ans_f = (
                "Bạn có thể tự kéo thử tỷ lệ chi phí trên trang <b>Ngưỡng cắt & Chi phí</b> ở thanh menu bên trái để xem Recall và F2-Score thay đổi trực quan nhé!"
            )
            return ans_m, ans_f
        if any(w in q for w in ["xét nghiệm máu nào", "yêu cầu xét nghiệm", "xét nghiệm gì", "fpg", "hba1c"]):
            ans_m = (
                f"Bạn nên yêu cầu bác sĩ làm <b>Xét nghiệm đường huyết lúc đói (FPG — nhịn ăn 8 tiếng)</b> hoặc <b>HbA1c</b> "
                f"(đánh giá đường huyết trung bình 3 tháng). Với điểm số <b>{prob_gb:.1%}</b> của bạn (ngưỡng cắt <b>{active_th:.2f}</b>), đây là cách kiểm tra chính xác nhất!"
            )
            ans_f = "Tối hôm trước ngày đi lấy máu FPG bạn nhớ nhịn ăn từ 22h đêm (chỉ uống nước lọc) và ngủ sớm để kết quả chuẩn nhất nhé!"
            return ans_m, ans_f

    # Call Google-style Live Web Search + AI LLM for ANY arbitrary question!
    if use_live_llm:
        llm_m, llm_f = call_free_llm_api(question_text, lang_code, d_cfg["m_name"], d_cfg["f_name"])
        if llm_m and llm_f:
            return llm_m, llm_f

    if lang_code == "vi":
        ans_m = (
            f"Hiện tại kết nối tìm kiếm trực tuyến đang bận nên tôi chưa tải được dữ liệu mới nhất cho câu hỏi <i>\"{question_text}\"</i>. "
            f"Bạn hãy bấm gửi lại sau vài giây nhé!"
        )
        ans_f = "Nếu bạn muốn hỏi về sức khỏe, dinh dưỡng hoặc bất kỳ chủ đề nào khác, cứ thử hỏi lại ngay nhé!"
    else:
        ans_m = (
            f"The live search connection is momentarily busy for <i>\"{question_text}\"</i>. Please try sending your question again in a few seconds!"
        )
        ans_f = "Feel free to retry or ask another question anytime!"
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
            "<div style='display:flex; align-items:flex-start; gap:14px; background:#ffffff; border:1px solid #bfdbfe; border-left:5px solid #1d4ed8; "
            "padding:14px 16px; border-radius:12px; margin-bottom:10px;'>"
            f"<img class='doc-avatar-anim' src='data:image/jpeg;base64,{DOC_MALE_B64}' style='width:54px; height:54px; object-fit:cover; border-radius:50%; border:2px solid #1d4ed8; flex-shrink:0;'/>"
            f"<div style='color:#1e293b; font-size:0.93rem; line-height:1.55;'><b style='color:#1d4ed8;'>{d_cfg['m_name']}:</b> {reply_m}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    if doc_mode in ["both", "female"]:
        st.markdown(
            "<div style='display:flex; align-items:flex-start; gap:14px; background:#ffffff; border:1px solid #bae6fd; border-left:5px solid #0284c7; "
            "padding:14px 16px; border-radius:12px; margin-bottom:4px;'>"
            f"<img class='doc-avatar-anim' src='data:image/jpeg;base64,{DOC_FEMALE_B64}' style='width:54px; height:54px; object-fit:cover; border-radius:50%; border:2px solid #0284c7; flex-shrink:0;'/>"
            f"<div style='color:#1e293b; font-size:0.93rem; line-height:1.55;'><b style='color:#0369a1;'>{d_cfg['f_name']}:</b> {reply_f}</div>"
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
