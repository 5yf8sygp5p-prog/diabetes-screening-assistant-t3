"""
app.py — Diabetes Screening Assistant with EIU Logo & Interactive Dual AI Doctors (4 Languages)
===============================================================================================
User-Decision Statement:
"An adult answering questions at home uses the app to decide whether to get a blood sugar test."
"""

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

# Support flat root GitHub upload as well as subfolder structure
_base_dir = Path(__file__).resolve().parent
if not (_base_dir / "pages").exists() and (_base_dir / "1_Threshold_and_Cost.py").exists():
    _flat_page = st.sidebar.radio(
        "Navigation",
        options=[
            "app",
            "Threshold and Cost",
            "Health Equity",
            "About Model",
            "Batch Screening",
        ],
        label_visibility="collapsed",
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

lang_code, t = render_sidebar_and_get_lang()
main_pipe, base_pipe, meta = load_artifacts()

default_th = float(meta["cost_assumptions"]["chosen_threshold"])
if "active_threshold" not in st.session_state:
    st.session_state["active_threshold"] = default_th
active_th = float(st.session_state["active_threshold"])

# ==============================================================================
# 1. HEADER SECTION WITH EIU UNIVERSITY LOGO
# ==============================================================================
header_html = (
    "<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px; "
    "background:#ffffff; border:2px solid #bae6fd; border-radius:18px; padding:20px 28px; margin-bottom:16px; "
    "box-shadow:0 6px 20px rgba(2,132,199,0.08);'>"
    "<div style='flex:1; min-width:280px;'>"
    "<div style='font-size:0.8rem; font-weight:800; color:#0284c7; text-transform:uppercase; letter-spacing:0.6px; margin-bottom:4px;'>"
    "🏛️ Eastern International University (EIU) &bull; Healthcare Decision-Support System"
    "</div>"
    f"<h1 style='font-size:2.15rem; font-weight:800; color:#0f172a; margin:0;'>{t['app_title']}</h1>"
    "</div>"
    f"<div style='text-align:right;'><img src='data:image/png;base64,{EIU_LOGO_B64}' style='max-height:66px; width:auto;'/></div>"
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

if submitted:
    st.session_state["has_analyzed"] = True

# ==============================================================================
# 3. COMPUTE SCORE & RENDER DUAL AI DOCTORS CONSULTATION
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

prob_gb, prob_lr, contributions = score_single_case(input_df, main_pipe, base_pipe, meta)
whatif_df = input_df.copy()
whatif_df["BMI"] = lower_bmi
prob_whatif, _, _ = score_single_case(whatif_df, main_pipe, base_pipe, meta)

is_flagged = prob_gb >= active_th
fname_map = FEATURE_LABELS_I18N.get(lang_code, FEATURE_LABELS_I18N["en"])
sorted_c = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
top_feat_label = fname_map.get(sorted_c[0][0], sorted_c[0][0])
has_run = st.session_state.get("has_analyzed", False)

DOC_TEXT = {
    "vi": {
        "section_title": "🩺 Phòng Hội Chẩn & Tương Tác Trực Tiếp Cùng 2 Bác Sĩ AI",
        "m_name": "👨‍⚕️ Bác sĩ Minh (Chuyên khoa Nội tiết & AI)",
        "f_name": "👩‍⚕️ Bác sĩ Lan (Chuyên khoa Dinh dưỡng & Dự phòng)",
        "m_idle": "Chào bạn! Tôi phụ trách phân tích điểm rủi ro AI và chỉ định xét nghiệm. Hãy điền 7 câu hỏi phía trên rồi bấm <b>Phân tích ngay</b> để tôi đánh giá hồ sơ cho bạn nhé!",
        "f_idle": "Xin chào! Tôi phụ trách tư vấn chỉ số BMI, chế độ ăn uống và vận động. Sau khi bạn bấm <b>Phân tích ngay</b>, tôi sẽ gợi ý kế hoạch chăm sóc sức khỏe tại nhà cho bạn!",
        "m_flag": f"Điểm tương đồng của bạn là <b>{prob_gb:.1%}</b> (cao hơn ngưỡng tầm soát <b>{active_th:.2f}</b>). Yếu tố nổi bật nhất của bạn là <b>{top_feat_label}</b>. Tôi khuyên bạn nên đặt lịch khám để làm xét nghiệm <b>Đường huyết lúc đói (FPG)</b> hoặc <b>HbA1c</b>.",
        "m_safe": f"Tin vui cho bạn! Điểm rủi ro ước tính của bạn là <b>{prob_gb:.1%}</b> (nằm dưới ngưỡng cảnh báo <b>{active_th:.2f}</b>). Bạn chưa cần xét nghiệm máu khẩn cấp ngay hôm nay, chỉ cần kiểm tra định kỳ hàng năm.",
        "f_flag": f"Chỉ số BMI hiện tại của bạn là <b>{bmi_val:.1f} kg/m²</b>. Nếu bạn giảm nhẹ BMI xuống <b>{lower_bmi:.1f} kg/m²</b> kết hợp đi bộ 30 phút/ngày và giảm tinh bột nhanh, điểm rủi ro sẽ giảm xuống còn <b>{prob_whatif:.1%}</b>!",
        "f_safe": f"Chỉ số BMI của bạn là <b>{bmi_val:.1f} kg/m²</b>. Để giữ mức rủi ro thấp bền vững (<b>{prob_whatif:.1%}</b>), bạn hãy duy trì ăn nhiều rau xanh, hạn chế nước ngọt có gas và ngủ đủ 7–8 tiếng mỗi ngày nhé!",
        "chat_header": "💬 Trò chuyện & Đặt câu hỏi trực tiếp với Bác sĩ Minh và Bác sĩ Lan",
        "choose_doc": "Chọn bác sĩ bạn muốn tương tác:",
        "opt_both": "👨‍⚕️👩‍⚕️ Cả 2 Bác sĩ cùng tư vấn",
        "opt_male": "👨‍⚕️ Chỉ hỏi Bác sĩ Minh (Nội tiết & Xét nghiệm)",
        "opt_female": "👩‍⚕️ Chỉ hỏi Bác sĩ Lan (Dinh dưỡng & Lối sống)",
        "quick_q_label": "Bấm nhanh câu hỏi thường gặp hoặc tự gõ câu hỏi bên dưới:",
        "q1": "🩸 Tôi nên làm xét nghiệm máu nào?",
        "q2": "🥗 Chế độ ăn & tập luyện phù hợp với BMI của tôi?",
        "q3": "⚖️ Tại sao ngưỡng cảnh báo là 0.18 mà không phải 0.50?",
        "input_placeholder": "Nhập câu hỏi của bạn cho 2 bác sĩ (ví dụ: Tôi bị huyết áp cao thì cần lưu ý gì?)...",
        "send_btn": "Gửi câu hỏi",
    },
    "en": {
        "section_title": "🩺 Interactive AI Medical Consultation with Dr. Minh & Dr. Lan",
        "m_name": "👨‍⚕️ Dr. Minh (Endocrinology & AI Risk Specialist)",
        "f_name": "👩‍⚕️ Dr. Lan (Nutrition & Preventive Care Specialist)",
        "m_idle": "Hello! I analyze your AI screening probability and lab testing threshold. Fill in the 7 questions above and click <b>Analyze</b> so I can review your profile!",
        "f_idle": "Hi there! I focus on your BMI, daily nutrition, and lifestyle prevention. Press <b>Analyze</b> above and I will prepare a personalized wellness plan for you!",
        "m_flag": f"Your estimated risk score is <b>{prob_gb:.1%}</b>, which exceeds our screening cutoff of <b>{active_th:.2f}</b>. Your top factor is <b>{top_feat_label}</b>. I recommend asking your physician for a routine <b>Fasting Plasma Glucose (FPG)</b> or <b>HbA1c</b> blood test.",
        "m_safe": f"Great news! Your estimated score is <b>{prob_gb:.1%}</b>, which is safely below our screening cutoff of <b>{active_th:.2f}</b>. No urgent lab test is flagged today—simply continue routine annual checkups.",
        "f_flag": f"Your calculated BMI is <b>{bmi_val:.1f} kg/m²</b>. If you gradually lower your BMI toward <b>{lower_bmi:.1f} kg/m²</b> with 30 minutes of daily brisk walking, your risk score drops to <b>{prob_whatif:.1%}</b>!",
        "f_safe": f"Your calculated BMI is <b>{bmi_val:.1f} kg/m²</b>. Keep maintaining balanced meals rich in fiber, low sugary drinks, and regular physical activity to keep your risk down at <b>{prob_whatif:.1%}</b>!",
        "chat_header": "💬 Chat & Ask Questions Directly to Dr. Minh and Dr. Lan",
        "choose_doc": "Choose which doctor to consult:",
        "opt_both": "👨‍⚕️👩‍⚕️ Consult Both Doctors Together",
        "opt_male": "👨‍⚕️ Ask Dr. Minh (Lab Tests & AI Model)",
        "opt_female": "👩‍⚕️ Ask Dr. Lan (BMI, Diet & Exercise)",
        "quick_q_label": "Quick-ask buttons or type your own question below:",
        "q1": "🩸 Which blood test should I ask for?",
        "q2": "🥗 What diet & exercise fit my BMI?",
        "q3": "⚖️ Why is the cutoff 0.18 instead of 0.50?",
        "input_placeholder": "Type your question for the doctors (e.g., How does high blood pressure affect blood sugar?)...",
        "send_btn": "Send Question",
    },
    "zh": {
        "section_title": "🩺 与两位 AI 医生互动会诊（明医生 & 兰医生）",
        "m_name": "👨‍⚕️ 明医生（内分泌与 AI 风险专家）",
        "f_name": "👩‍⚕️ 兰医生（临床营养与预防医学专家）",
        "m_idle": "您好！我负责评估您的 AI 筛查风险得分与化验建议。请填写上方 7 个问题并点击 <b>立即分析</b>！",
        "f_idle": "您好！我负责为您分析 BMI 指数、饮食与运动方案。点击上方 <b>立即分析</b> 后我将为您提供健康指导！",
        "m_flag": f"您的估计风险得分为 <b>{prob_gb:.1%}</b>，高于筛查阈值 <b>{active_th:.2f}</b>（主要驱动因素：<b>{top_feat_label}</b>）。建议您向医生咨询进行 <b>空腹血糖 (FPG)</b> 或 <b>糖化血红蛋白 (HbA1c)</b> 血液检测。",
        "m_safe": f"好消息！您的风险得分为 <b>{prob_gb:.1%}</b>，低于阈值 <b>{active_th:.2f}</b>。目前无需紧急检测，保持年度常规体检即可。",
        "f_flag": f"您当前的 BMI 为 <b>{bmi_val:.1f} kg/m²</b>。如果通过均衡饮食和每天快走 30 分钟将 BMI 降至 <b>{lower_bmi:.1f} kg/m²</b>，您的风险得分将降至 <b>{prob_whatif:.1%}</b>！",
        "f_safe": f"您的 BMI 为 <b>{bmi_val:.1f} kg/m²</b>。请继续保持高纤维饮食、减少含糖饮料并坚持每周运动！",
        "chat_header": "💬 直接向明医生与兰医生提问互动",
        "choose_doc": "选择咨询对象：",
        "opt_both": "👨‍⚕️👩‍⚕️ 两位医生联合解答",
        "opt_male": "👨‍⚕️ 咨询明医生（内分泌与血糖检测）",
        "opt_female": "👩‍⚕️ 咨询兰医生（营养、BMI 与生活方式）",
        "quick_q_label": "快捷提问按钮：",
        "q1": "🩸 我去医院应该做哪项血液检查？",
        "q2": "🥗 适合我当前 BMI 的饮食与运动建议？",
        "q3": "⚖️ 为什么阈值设为 0.18 而不是 0.50？",
        "input_placeholder": "输入您想咨询两位医生的问题...",
        "send_btn": "发送问题",
    },
    "fr": {
        "section_title": "🩺 Consultation Interactive avec nos 2 Médecins IA (Dr Minh & Dr Lan)",
        "m_name": "👨‍⚕️ Dr Minh (Spécialiste Endocrinologie & IA)",
        "f_name": "👩‍⚕️ Dr Lan (Spécialiste Nutrition & Prévention)",
        "m_idle": "Bonjour ! J'analyse votre score de risque IA. Remplissez les 7 questions ci-dessus et cliquez sur <b>Analyser</b> !",
        "f_idle": "Bonjour ! Je vous conseille sur votre IMC, votre alimentation et l'activité physique. Cliquez sur <b>Analyser</b> pour voir mon bilan !",
        "m_flag": f"Votre score estimé est de <b>{prob_gb:.1%}</b> (au-dessus du seuil <b>{active_th:.2f}</b> ; facteur principal : <b>{top_feat_label}</b>). Je vous recommande de demander un test de <b>glycémie à jeun (FPG)</b> ou <b>HbA1c</b> à votre médecin.",
        "m_safe": f"Bonne nouvelle ! Votre score est de <b>{prob_gb:.1%}</b> (inférieur au seuil <b>{active_th:.2f}</b>). Aucun test urgent n'est requis aujourd'hui.",
        "f_flag": f"Votre IMC calculé est de <b>{bmi_val:.1f} kg/m²</b>. En abaissant votre IMC vers <b>{lower_bmi:.1f} kg/m²</b> avec 30 min de marche quotidienne, votre score descendrait à <b>{prob_whatif:.1%}</b> !",
        "f_safe": f"Votre IMC est de <b>{bmi_val:.1f} kg/m²</b>. Continuez à privilégier une alimentation riche en fibres et l'exercice régulier !",
        "chat_header": "💬 Discutez et posez vos questions au Dr Minh et au Dr Lan",
        "choose_doc": "Choisissez votre médecin :",
        "opt_both": "👨‍⚕️👩‍⚕️ Consulter les 2 médecins ensemble",
        "opt_male": "👨‍⚕️ Demander au Dr Minh (Tests & Modèle IA)",
        "opt_female": "👩‍⚕️ Demander au Dr Lan (IMC, Nutrition & Sport)",
        "quick_q_label": "Questions rapides :",
        "q1": "🩸 Quel test sanguin demander ?",
        "q2": "🥗 Quel régime et sport pour mon IMC ?",
        "q3": "⚖️ Pourquoi le seuil est-il de 0.18 ?",
        "input_placeholder": "Posez votre question aux médecins...",
        "send_btn": "Envoyer",
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
        "<div style='display:flex; align-items:center; gap:16px; background:#ffffff; border:2px solid #7dd3fc; "
        "border-radius:16px; padding:16px 18px; box-shadow:0 6px 16px rgba(2,132,199,0.08); min-height:158px;'>"
        f"<img src='data:image/jpeg;base64,{DOC_MALE_B64}' style='width:95px; height:116px; object-fit:cover; border-radius:14px; border:2.5px solid #0284c7; flex-shrink:0;'/>"
        "<div>"
        f"<div style='font-weight:800; color:#0369a1; font-size:0.96rem; margin-bottom:6px;'>{d_cfg['m_name']}</div>"
        f"<div style='color:#0f172a; font-size:0.92rem; line-height:1.55;'>{m_speech}</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

with doc_col2:
    st.markdown(
        "<div style='display:flex; align-items:center; gap:16px; background:#ffffff; border:2px solid #5eead4; "
        "border-radius:16px; padding:16px 18px; box-shadow:0 6px 16px rgba(13,148,136,0.08); min-height:158px;'>"
        f"<img src='data:image/jpeg;base64,{DOC_FEMALE_B64}' style='width:95px; height:116px; object-fit:cover; border-radius:14px; border:2.5px solid #0d9488; flex-shrink:0;'/>"
        "<div>"
        f"<div style='font-weight:800; color:#0f766e; font-size:0.96rem; margin-bottom:6px;'>{d_cfg['f_name']}</div>"
        f"<div style='color:#0f172a; font-size:0.92rem; line-height:1.55;'>{f_speech}</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

if not has_run:
    st.info(t["prompt_analyze"])


# ==============================================================================
# 4. INTERACTIVE CHAT BOX WITH BOTH DOCTORS
# ==============================================================================
def generate_doctor_replies(question_text: str) -> tuple:
    q_low = question_text.lower()
    if lang_code == "vi":
        if any(w in q_low for w in ["xét nghiệm", "máu", "test", "fpg", "hba1c", "khám"]):
            ans_m = (
                f"Bạn nên yêu cầu bác sĩ làm <b>Xét nghiệm đường huyết lúc đói (FPG — nhịn ăn 8 tiếng)</b> hoặc "
                f"<b>HbA1c</b> (đánh giá đường huyết trung bình 3 tháng). Với điểm số <b>{prob_gb:.1%}</b> của bạn, đây là cách kiểm tra chính xác nhất!"
            )
            ans_f = (
                "Trước ngày đi lấy máu xét nghiệm FPG, tối hôm trước bạn nhớ nhịn ăn từ 22h đêm (chỉ uống nước lọc) "
                "và ngủ sớm để kết quả đường huyết chính xác nhất nhé!"
            )
        elif any(w in q_low for w in ["ăn", "tập", "bmi", "cân", "dinh dưỡng", "diet"]):
            ans_m = (
                f"Khi giảm BMI từ <b>{bmi_val:.1f}</b> xuống <b>{lower_bmi:.1f} kg/m²</b>, độ nhạy insulin của cơ thể cải thiện rõ rệt, "
                f"giúp điểm rủi ro trên mô hình giảm từ <b>{prob_gb:.1%}</b> xuống <b>{prob_whatif:.1%}</b>."
            )
            ans_f = (
                f"Với chiều cao <b>{height_cm} cm</b> và cân nặng <b>{weight_kg} kg</b> (BMI <b>{bmi_val:.1f}</b>), bạn nên áp dụng quy tắc "
                f"Đĩa thức ăn: <b>1/2 rau xanh</b>, <b>1/4 đạm lành mạnh</b> (cá, ức gà, đậu phụ), <b>1/4 tinh bột nguyên cám</b>, kết hợp đi bộ nhanh 30 phút sau bữa tối!"
            )
        elif any(w in q_low for w in ["0.18", "ngưỡng", "threshold", "0.50", "chi phí"]):
            ans_m = (
                "Chúng tôi chọn ngưỡng <b>0.18</b> trên tập Validation vì bỏ sót 1 ca tiểu đường nguy hiểm gấp <b>5 lần</b> "
                "so với 1 lần xét nghiệm kiểm tra. Nhờ ngưỡng 0.18, mô hình phát hiện được tới <b>70.5%</b> số ca bệnh!"
            )
            ans_f = (
                "Đúng vậy! Phát hiện sớm ở giai đoạn tiền tiểu đường giúp bạn chỉ cần điều chỉnh lối sống mà chưa phải dùng thuốc suốt đời."
            )
        else:
            advice_tail = "Bạn nên đi xét nghiệm máu kiểm tra sớm nhé!" if is_flagged else "Hiện tại chỉ số của bạn vẫn trong vùng an toàn."
            ans_m = (
                f"Về câu hỏi <i>\"{question_text}\"</i>: Hồ sơ của bạn hiện có điểm rủi ro <b>{prob_gb:.1%}</b> "
                f"(ngưỡng cắt <b>{active_th:.2f}</b>), yếu tố chính là <b>{top_feat_label}</b>. {advice_tail}"
            )
            ans_f = (
                f"Tôi bổ sung thêm: Với BMI <b>{bmi_val:.1f} kg/m²</b> và sức khỏe tổng quát mức <b>{GENHLTH_MAP[gen_hlth]}</b>, "
                f"việc kiểm soát huyết áp, mỡ máu và vận động 150 phút/tuần sẽ bảo vệ bạn rất tốt!"
            )
    elif lang_code == "zh":
        ans_m = (
            f"针对您的问题：您当前的筛查得分为 <b>{prob_gb:.1%}</b>（阈值 <b>{active_th:.2f}</b>），"
            f"最关键因素为 <b>{top_feat_label}</b>。建议咨询医生进行空腹血糖 (FPG) 或 HbA1c 检测。"
        )
        ans_f = (
            f"从营养与生活方式来看，您的 BMI 为 <b>{bmi_val:.1f} kg/m²</b>。"
            f"保持每日 30 分钟有氧运动与低糖高纤维饮食可将风险降至 <b>{prob_whatif:.1%}</b>。"
        )
    elif lang_code == "fr":
        ans_m = (
            f"Concernant votre question : votre score est de <b>{prob_gb:.1%}</b> (seuil <b>{active_th:.2f}</b>), "
            f"principalement lié à <b>{top_feat_label}</b>. Un test FPG ou HbA1c en laboratoire est recommandé en cas de doute."
        )
        ans_f = (
            f"Côté mode de vie, votre IMC est de <b>{bmi_val:.1f} kg/m²</b>. "
            f"Une alimentation équilibrée et 30 minutes de marche par jour réduisent votre risque à <b>{prob_whatif:.1%}</b>."
        )
    else:
        if any(w in q_low for w in ["test", "blood", "lab", "fpg", "hba1c", "doctor"]):
            ans_m = (
                f"Ask your doctor for a <b>Fasting Plasma Glucose (FPG)</b> test or an <b>HbA1c</b> test. "
                f"Given your score of <b>{prob_gb:.1%}</b>, a quick venous blood draw is the gold standard."
            )
            ans_f = (
                "Tip for your lab visit: fast for at least 8 hours overnight (water is fine!) before a Fasting Plasma Glucose test so your reading is accurate."
            )
        elif any(w in q_low for w in ["bmi", "diet", "food", "exercise", "weight"]):
            ans_m = (
                f"Lowering your BMI by just 2 points (from <b>{bmi_val:.1f}</b> to <b>{lower_bmi:.1f} kg/m²</b>) "
                f"improves insulin sensitivity and reduces your model score from <b>{prob_gb:.1%}</b> to <b>{prob_whatif:.1%}</b>."
            )
            ans_f = (
                f"At <b>{height_cm} cm</b> and <b>{weight_kg} kg</b> (BMI <b>{bmi_val:.1f}</b>), try the Plate Method: "
                f"fill half your plate with non-starchy vegetables, one-quarter with lean protein, and take a 20-minute walk after meals!"
            )
        else:
            ans_m = (
                f"Regarding <i>\"{question_text}\"</i>: Your current screening score is <b>{prob_gb:.1%}</b> vs. our "
                f"<b>{active_th:.2f}</b> threshold (top factor: <b>{top_feat_label}</b>). The 0.18 cutoff is chosen because missing diabetes is ~5x worse than a routine lab test."
            )
            ans_f = (
                f"And from a preventive care perspective, keeping your BMI (<b>{bmi_val:.1f} kg/m²</b>) and blood pressure in check through daily movement is the best long-term protection!"
            )
    return ans_m, ans_f


with st.expander(d_cfg["chat_header"], expanded=True):
    doc_mode = st.radio(
        d_cfg["choose_doc"],
        options=["both", "male", "female"],
        format_func=lambda k: d_cfg["opt_both"]
        if k == "both"
        else (d_cfg["opt_male"] if k == "male" else d_cfg["opt_female"]),
        horizontal=True,
    )

    st.caption(d_cfg["quick_q_label"])
    q_col1, q_col2, q_col3 = st.columns(3)
    if "active_question" not in st.session_state:
        st.session_state["active_question"] = d_cfg["q1"]

    if q_col1.button(d_cfg["q1"], use_container_width=True):
        st.session_state["active_question"] = d_cfg["q1"]
    if q_col2.button(d_cfg["q2"], use_container_width=True):
        st.session_state["active_question"] = d_cfg["q2"]
    if q_col3.button(d_cfg["q3"], use_container_width=True):
        st.session_state["active_question"] = d_cfg["q3"]

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
    reply_m, reply_f = generate_doctor_replies(current_q)

    st.markdown(
        f"<div style='background:#e0f2fe; color:#0c4a6e; padding:10px 16px; border-radius:12px; font-weight:700; margin:8px 0;'>"
        f"🙋 {current_q}</div>",
        unsafe_allow_html=True,
    )

    if doc_mode in ["both", "male"]:
        st.markdown(
            "<div style='display:flex; align-items:flex-start; gap:12px; background:#f8fafc; border:1px solid #bae6fd; border-left:5px solid #0284c7; "
            "padding:12px 16px; border-radius:12px; margin-bottom:8px;'>"
            f"<img src='data:image/jpeg;base64,{DOC_MALE_B64}' style='width:50px; height:50px; object-fit:cover; border-radius:50%; border:2px solid #0284c7; flex-shrink:0;'/>"
            f"<div style='color:#0f172a; font-size:0.93rem; line-height:1.5;'><b>{d_cfg['m_name']}:</b> {reply_m}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    if doc_mode in ["both", "female"]:
        st.markdown(
            "<div style='display:flex; align-items:flex-start; gap:12px; background:#f0fdfa; border:1px solid #99f6e4; border-left:5px solid #0d9488; "
            "padding:12px 16px; border-radius:12px; margin-bottom:4px;'>"
            f"<img src='data:image/jpeg;base64,{DOC_FEMALE_B64}' style='width:50px; height:50px; object-fit:cover; border-radius:50%; border:2px solid #0d9488; flex-shrink:0;'/>"
            f"<div style='color:#0f172a; font-size:0.93rem; line-height:1.5;'><b>{d_cfg['f_name']}:</b> {reply_f}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

# ==============================================================================
# 5. DETAILED RESULTS, RISK SPECTRUM & TOP FACTORS
# ==============================================================================
if has_run:
    if is_flagged:
        banner_bg = "#fff1f2"
        banner_border = "#e11d48"
        banner_color = "#881337"
        banner_title = t["res_recommend_flag"]
    else:
        banner_bg = "#f0fdf4"
        banner_border = "#16a34a"
        banner_color = "#14532d"
        banner_title = t["res_recommend_safe"]

    pin_pct = min(max(prob_gb * 100.0, 2.0), 98.0)
    th_pct = min(max(active_th * 100.0, 2.0), 98.0)
    nat_n = int(round(prob_gb * 100))

    result_card_html = (
        f"<div style='background:{banner_bg}; border:1.5px solid {banner_border}; border-left:7px solid {banner_border}; border-radius:14px; padding:18px 22px; margin-top:14px; margin-bottom:16px;'>"
        f"<h3 style='margin:0 0 6px 0; color:{banner_color}; font-size:1.25rem; font-weight:800;'>{banner_title}</h3>"
        f"<p style='margin:0; color:#0f172a; font-size:0.95rem; font-weight:500;'>{t['res_natural_freq'].format(n=nat_n)}</p>"
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
        f"<div class='clean-card' style='margin-top:14px; background:#f0f9ff; border-color:#7dd3fc;'>"
        f"<p style='margin:0 0 4px 0; font-weight:800; color:#0c4a6e;'>{t['what_if_title']}</p>"
        f"<p style='margin:0; font-size:0.93rem; color:#0f172a; font-weight:500;'>"
        f"{t['what_if_desc'].format(bmi1=f'{bmi_val:.1f}', bmi2=f'{lower_bmi:.1f}', p1=f'{prob_gb:.1%}', p2=f'{prob_whatif:.1%}', delta=f'{delta_p:+.1%}')}"
        f"</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
