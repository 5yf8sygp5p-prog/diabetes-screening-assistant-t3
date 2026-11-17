"""
app.py — Diabetes Screening Assistant with EIU Logo & Interactive AI Doctors (Gong Yoo & Lee Sung-kyung)
========================================================================================================
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
# Extra dark-mode-proof & soothing color overrides embedded directly in app.py
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
    "}"
    ".stButton > button * { color: #1e293b !important; }"
    ".stButton > button:hover { background-color: #e2e8f0 !important; border-color: #0284c7 !important; }"
    "div[data-testid='stFormSubmitButton'] > button, .stButton > button[kind='primary'] {"
    "background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important; color: #ffffff !important; border: none !important;"
    "}"
    "div[data-testid='stFormSubmitButton'] > button *, .stButton > button[kind='primary'] * { color: #ffffff !important; }"
    "</style>",
    unsafe_allow_html=True,
)

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
    "background:#ffffff; border:1.5px solid #cbd5e1; border-radius:16px; padding:18px 26px; margin-bottom:16px; "
    "box-shadow:0 4px 12px rgba(15,23,42,0.04);'>"
    "<div style='flex:1; min-width:280px;'>"
    "<div style='font-size:0.78rem; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:0.6px; margin-bottom:4px;'>"
    "🏛️ Eastern International University (EIU) &bull; Healthcare Decision-Support System"
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

if submitted:
    st.session_state["has_analyzed"] = True

# ==============================================================================
# 3. COMPUTE SCORE & RENDER DUAL AI DOCTORS (GONG YOO & LEE SUNG-KYUNG)
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
        "section_title": "🩺 Phòng Hội Chẩn & Giao Lưu Cùng Bác Sĩ Gong Yoo & Bác Sĩ Lee Sung-kyung",
        "m_name": "👨‍⚕️ Bác sĩ Gong Yoo (Chuyên khoa Nội tiết & AI)",
        "f_name": "👩‍⚕️ Bác sĩ Lee Sung-kyung (Chuyên khoa Dinh dưỡng & Lối sống)",
        "m_idle": "Chào bạn! Tôi là <b>Bác sĩ Gong Yoo</b>. Hãy điền 7 câu hỏi trên rồi bấm <b>Phân tích ngay</b> để tôi đánh giá điểm rủi ro cho bạn nhé (hoặc kéo xuống dưới hỏi tôi bất cứ câu hỏi 'vô tri' nào cũng được!).",
        "f_idle": "Xin chào! Mình là <b>Bác sĩ Lee Sung-kyung</b>. Bấm <b>Phân tích ngay</b> để mình tính BMI và lên thực đơn cho bạn nha — kể cả hỏi chuyện trà sữa, thức khuya hay thất tình mình cũng giải đáp hết!",
        "m_flag": f"Điểm tương đồng của bạn là <b>{prob_gb:.1%}</b> (vượt ngưỡng tầm soát <b>{active_th:.2f}</b>). Yếu tố nổi bật nhất là <b>{top_feat_label}</b>. Bạn nên đặt lịch làm xét nghiệm <b>Đường huyết lúc đói (FPG)</b> hoặc <b>HbA1c</b> để kiểm tra nhé.",
        "m_safe": f"Tin vui! Điểm rủi ro của bạn chỉ là <b>{prob_gb:.1%}</b> (thấp hơn ngưỡng cảnh báo <b>{active_th:.2f}</b>). Bạn chưa cần xét nghiệm khẩn cấp hôm nay, chỉ cần duy trì khám sức khỏe định kỳ.",
        "f_flag": f"BMI hiện tại của bạn là <b>{bmi_val:.1f} kg/m²</b>. Nếu bạn giảm nhẹ BMI xuống <b>{lower_bmi:.1f} kg/m²</b> kết hợp đi bộ 30 phút/ngày, điểm rủi ro sẽ giảm xuống còn <b>{prob_whatif:.1%}</b>!",
        "f_safe": f"BMI của bạn là <b>{bmi_val:.1f} kg/m²</b>. Để giữ mức rủi ro thấp (<b>{prob_whatif:.1%}</b>), nhớ ăn nhiều rau xanh, bớt trà sữa full topping và ngủ đủ giấc nha!",
        "chat_header": "💬 Hỏi Xoáy Đáp Xoay Cùng Bác Sĩ Gong Yoo & Bác Sĩ Lee Sung-kyung (Hỏi nghiêm túc hay hỏi 'vô tri' đều trả lời được hết!)",
        "choose_doc": "Chọn bác sĩ bạn muốn tương tác:",
        "opt_both": "👨‍⚕️👩‍⚕️ Cả 2 Bác sĩ (Gong Yoo & Lee Sung-kyung)",
        "opt_male": "👨‍⚕️ Chỉ hỏi Bác sĩ Gong Yoo",
        "opt_female": "👩‍⚕️ Chỉ hỏi Bác sĩ Lee Sung-kyung",
        "quick_q_label": "Bấm thử các câu hỏi (cả nghiêm túc lẫn 'câu hỏi ngây ngô') hoặc tự gõ câu hỏi bất kỳ bên dưới:",
        "q1": "🧋 Uống trà sữa 0% đường nhưng full trân châu có bị tiểu đường không?",
        "q2": "🐜 Đi tiểu có kiến bu mới là bị tiểu đường đúng không bác sĩ?",
        "q3": "💓 Nhìn 2 bác sĩ đẹp quá tim đập nhanh có phải triệu chứng bệnh không?",
        "q4": "🩸 Nếu đi khám thì tôi nên yêu cầu xét nghiệm máu nào?",
        "q5": "🥗 Thực đơn & bài tập nào phù hợp với BMI của tôi?",
        "q6": "🤫 Nhịn ăn 2 ngày trước khi đi khám để 'lừa' máy xét nghiệm được không?",
        "input_placeholder": "Gõ bất kỳ câu hỏi nào (VD: Ăn sầu riêng có sao không? Bác sĩ có người yêu chưa? Tui lười tập quá thì sao?)...",
        "send_btn": "Gửi câu hỏi",
    },
    "en": {
        "section_title": "🩺 AI Medical Consultation with Dr. Gong Yoo & Dr. Lee Sung-kyung",
        "m_name": "👨‍⚕️ Dr. Gong Yoo (Endocrinology & AI Risk Specialist)",
        "f_name": "👩‍⚕️ Dr. Lee Sung-kyung (Nutrition & Lifestyle Specialist)",
        "m_idle": "Hello! I'm <b>Dr. Gong Yoo</b>. Fill in the 7 questions above and click <b>Analyze</b>, or ask me any serious or silly question below!",
        "f_idle": "Hi! I'm <b>Dr. Lee Sung-kyung</b>. Click <b>Analyze</b> to get your BMI & wellness check—and feel free to ask me about boba tea, late-night snacks, or anything else!",
        "m_flag": f"Your estimated risk score is <b>{prob_gb:.1%}</b> (above our <b>{active_th:.2f}</b> cutoff; top factor: <b>{top_feat_label}</b>). I recommend asking your doctor for a routine <b>Fasting Plasma Glucose (FPG)</b> or <b>HbA1c</b> test.",
        "m_safe": f"Good news! Your estimated score is <b>{prob_gb:.1%}</b> (below our <b>{active_th:.2f}</b> cutoff). No urgent lab test is flagged today.",
        "f_flag": f"Your BMI is <b>{bmi_val:.1f} kg/m²</b>. Lowering your BMI toward <b>{lower_bmi:.1f} kg/m²</b> with 30 minutes of daily walking reduces your risk score to <b>{prob_whatif:.1%}</b>!",
        "f_safe": f"Your BMI is <b>{bmi_val:.1f} kg/m²</b>. Keep enjoying balanced meals and daily activity to maintain a low risk of <b>{prob_whatif:.1%}</b>!",
        "chat_header": "💬 Interactive Chat with Dr. Gong Yoo & Dr. Lee Sung-kyung (Serious or silly questions welcome!)",
        "choose_doc": "Choose which doctor to talk to:",
        "opt_both": "👨‍⚕️👩‍⚕️ Both Doctors (Gong Yoo & Lee Sung-kyung)",
        "opt_male": "👨‍⚕️ Dr. Gong Yoo Only",
        "opt_female": "👩‍⚕️ Dr. Lee Sung-kyung Only",
        "quick_q_label": "Try a quick question (serious or silly!) or type your own below:",
        "q1": "🧋 Does 0% sugar boba tea with extra tapioca pearls count as healthy?",
        "q2": "🐜 Can ants really smell sugar in urine if someone has diabetes?",
        "q3": "💓 My heart beats fast looking at you two—is that diabetes?",
        "q4": "🩸 Which exact blood test should I ask my doctor for?",
        "q5": "🥗 What diet & exercise plan fits my current BMI?",
        "q6": "🤫 Can I starve myself for 1 day to 'trick' the blood test?",
        "input_placeholder": "Ask anything (e.g., Do skinny people get diabetes? Can I drink Coke Zero?)...",
        "send_btn": "Send Question",
    },
    "zh": {
        "section_title": "🩺 与孔刘医生 (Dr. Gong Yoo) & 李圣经医生 (Dr. Lee Sung-kyung) 互动会诊",
        "m_name": "👨‍⚕️ 孔刘医生 Dr. Gong Yoo（内分泌与 AI 专家）",
        "f_name": "👩‍⚕️ 李圣经医生 Dr. Lee Sung-kyung（营养与生活方式专家）",
        "m_idle": "您好！我是<b>孔刘医生</b>。请填写上方 7 个问题并点击<b>立即分析</b>，或者在下方问我任何严肃或脑洞大开的问题！",
        "f_idle": "嗨！我是<b>李圣经医生</b>。无论是 BMI、饮食还是“无糖珍珠奶茶会不会胖”，都可以问我哦！",
        "m_flag": f"您的估计风险得分为 <b>{prob_gb:.1%}</b>，高于阈值 <b>{active_th:.2f}</b>（主要因素：<b>{top_feat_label}</b>）。建议咨询医生做 <b>FPG</b> 或 <b>HbA1c</b> 检测。",
        "m_safe": f"好消息！您的风险得分为 <b>{prob_gb:.1%}</b>，低于阈值 <b>{active_th:.2f}</b>。目前无需紧急检测。",
        "f_flag": f"您的 BMI 为 <b>{bmi_val:.1f} kg/m²</b>。若将 BMI 降至 <b>{lower_bmi:.1f} kg/m²</b>，风险将降至 <b>{prob_whatif:.1%}</b>！",
        "f_safe": f"您的 BMI 为 <b>{bmi_val:.1f} kg/m²</b>。请继续保持均衡饮食与适度运动！",
        "chat_header": "💬 与孔刘医生和李圣经医生互动问答（严肃或趣味问题皆可！）",
        "choose_doc": "选择咨询对象：",
        "opt_both": "👨‍⚕️👩‍⚕️ 两位医生联合解答",
        "opt_male": "👨‍⚕️ 仅问孔刘医生",
        "opt_female": "👩‍⚕️ 仅问李圣经医生",
        "quick_q_label": "点击下方快捷问题或自行输入：",
        "q1": "🧋 点无糖奶茶但加满黑糖珍珠会有风险吗？",
        "q2": "🐜 尿液招蚂蚁真的是糖尿病的症状吗？",
        "q3": "💓 看到两位医生心跳加速是生病了吗？",
        "q4": "🩸 去医院应该要求做哪项血液检查？",
        "q5": "🥗 适合我当前 BMI 的饮食与运动方案？",
        "q6": "🤫 抽血前突击饿肚子能骗过化验单吗？",
        "input_placeholder": "输入任何问题（例如：瘦子也会得糖尿病吗？熬夜看剧有影响吗？）...",
        "send_btn": "发送问题",
    },
    "fr": {
        "section_title": "🩺 Consultation Interactive avec Dr Gong Yoo & Dr Lee Sung-kyung",
        "m_name": "👨‍⚕️ Dr Gong Yoo (Endocrinologie & IA)",
        "f_name": "👩‍⚕️ Dr Lee Sung-kyung (Nutrition & Mode de vie)",
        "m_idle": "Bonjour ! Je suis le <b>Dr Gong Yoo</b>. Remplissez les 7 questions et cliquez sur <b>Analyser</b>, ou posez-moi vos questions (même les plus drôles) !",
        "f_idle": "Salut ! Je suis le <b>Dr Lee Sung-kyung</b>. Posez-moi toutes vos questions sur l'IMC, le bubble tea ou le sport !",
        "m_flag": f"Votre score est de <b>{prob_gb:.1%}</b> (au-dessus du seuil <b>{active_th:.2f}</b> ; facteur clé : <b>{top_feat_label}</b>). Demandez un test <b>FPG</b> ou <b>HbA1c</b> à votre médecin.",
        "m_safe": f"Bonne nouvelle ! Votre score est de <b>{prob_gb:.1%}</b> (sous le seuil <b>{active_th:.2f}</b>). Aucun test urgent requis.",
        "f_flag": f"Votre IMC est de <b>{bmi_val:.1f} kg/m²</b>. En le réduisant vers <b>{lower_bmi:.1f} kg/m²</b>, votre score baisserait à <b>{prob_whatif:.1%}</b> !",
        "f_safe": f"Votre IMC est de <b>{bmi_val:.1f} kg/m²</b>. Continuez vos bonnes habitudes !",
        "chat_header": "💬 Discutez avec Dr Gong Yoo & Dr Lee Sung-kyung (Questions sérieuses ou insolites !)",
        "choose_doc": "Choisissez votre médecin :",
        "opt_both": "👨‍⚕️👩‍⚕️ Les 2 médecins ensemble",
        "opt_male": "👨‍⚕️ Dr Gong Yoo uniquement",
        "opt_female": "👩‍⚕️ Dr Lee Sung-kyung uniquement",
        "quick_q_label": "Questions rapides ou libres :",
        "q1": "🧋 Un bubble tea 0% sucre mais plein de perles de tapioca, ça passe ?",
        "q2": "🐜 Les fourmis sont-elles vraiment attirées par le sucre en cas de diabète ?",
        "q3": "💓 Mon cœur bat vite en vous voyant, est-ce un symptôme ?",
        "q4": "🩸 Quel test sanguin demander au médecin ?",
        "q5": "🥗 Quel programme nutrition & sport pour mon IMC ?",
        "q6": "🤫 Peut-on jeûner 2 jours pour tromper la prise de sang ?",
        "input_placeholder": "Posez votre question (ex: Les personnes minces peuvent-elles avoir le diabète ?)...",
        "send_btn": "Envoyer",
    },
}

d_cfg = DOC_TEXT.get(lang_code, DOC_TEXT["en"])
m_speech = d_cfg["m_idle"] if not has_run else (d_cfg["m_flag"] if is_flagged else d_cfg["m_safe"])
f_speech = d_cfg["f_idle"] if not has_run else (d_cfg["f_flag"] if is_flagged else d_cfg["f_safe"])

st.markdown(
    f"<h3 style='color:#1e293b; margin-top:18px; margin-bottom:10px; font-weight:800;'>{d_cfg['section_title']}</h3>",
    unsafe_allow_html=True,
)

doc_col1, doc_col2 = st.columns(2, gap="medium")
with doc_col1:
    st.markdown(
        "<div style='display:flex; align-items:center; gap:16px; background:#ffffff; border:1.5px solid #cbd5e1; border-left:6px solid #2563eb; "
        "border-radius:14px; padding:16px 18px; box-shadow:0 3px 10px rgba(15,23,42,0.04); min-height:152px;'>"
        f"<img src='data:image/jpeg;base64,{DOC_MALE_B64}' style='width:92px; height:112px; object-fit:cover; border-radius:12px; border:2px solid #2563eb; flex-shrink:0;'/>"
        "<div>"
        f"<div style='font-weight:800; color:#1d4ed8; font-size:0.95rem; margin-bottom:6px;'>{d_cfg['m_name']}</div>"
        f"<div style='color:#1e293b; font-size:0.91rem; line-height:1.55;'>{m_speech}</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

with doc_col2:
    st.markdown(
        "<div style='display:flex; align-items:center; gap:16px; background:#ffffff; border:1.5px solid #cbd5e1; border-left:6px solid #0d9488; "
        "border-radius:14px; padding:16px 18px; box-shadow:0 3px 10px rgba(15,23,42,0.04); min-height:152px;'>"
        f"<img src='data:image/jpeg;base64,{DOC_FEMALE_B64}' style='width:92px; height:112px; object-fit:cover; border-radius:12px; border:2px solid #0d9488; flex-shrink:0;'/>"
        "<div>"
        f"<div style='font-weight:800; color:#0f766e; font-size:0.95rem; margin-bottom:6px;'>{d_cfg['f_name']}</div>"
        f"<div style='color:#1e293b; font-size:0.91rem; line-height:1.55;'>{f_speech}</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

if not has_run:
    st.info(t["prompt_analyze"])


# ==============================================================================
# 4. SMART & WITTY CONVERSATIONAL AI ENGINE (HANDLES SILLY + CLINICAL QUESTIONS)
# ==============================================================================
def generate_doctor_replies(question_text: str) -> tuple:
    q = question_text.lower().strip()

    if lang_code == "vi":
        # 1. Trà sữa / ngọt / kẹo / nước ngọt / bánh
        if any(w in q for w in ["trà sữa", "trân châu", "topping", "kẹo", "bánh", "nước ngọt", "coca", "pepsi", "đường", "ngọt"]):
            ans_m = (
                f"Haha, 'trà sữa 0% đường nhưng full trân châu hoàng kim + kem cheese' là cú lừa kinh điển nhất quả đất! "
                f"Trân châu làm từ tinh bột sắn nấu đẫm đường đen, làm đường huyết tăng vọt cực nhanh. Với điểm rủi ro <b>{prob_gb:.1%}</b> của bạn, "
                f"thỉnh thoảng uống 1 ly không topping thì được, chứ uống mỗi ngày là tôi hẹn gặp bạn ở phòng xét nghiệm máu sớm đấy!"
            )
            ans_f = (
                f"Đừng để chữ 'ít đường' đánh lừa nha! Nếu thèm quá, bạn hãy chọn trà ô long hoặc hồng trà không đường, "
                f"topping đổi sang hạt chia hoặc nha đam không đường. Vừa giữ dáng đẹp (BMI của bạn đang là <b>{bmi_val:.1f}</b>), vừa không làm tuyến tụy 'kêu cứu'!"
            )
        # 2. Kiến bu / nước tiểu
        elif any(w in q for w in ["kiến", "tiểu", "đái", "vệ sinh"]):
            ans_m = (
                "Nghe tưởng đùa nhưng trong lịch sử y học là thật 100% đấy! Tên bệnh 'Tiểu đường' (Diabetes Mellitus — ngọt như mật) "
                "xuất phát từ việc khi đường trong máu vượt quá ngưỡng thận (~180 mg/dL), thận không giữ nổi nữa nên đường tràn ra nước tiểu làm kiến bu thật. "
                "Nhưng thời đại AI rồi, đừng đợi kiến bu mới đi khám nhé — xét nghiệm máu FPG chỉ mất 5 phút thôi!"
            )
            ans_f = (
                f"Haha, dùng kiến để khám bệnh là phương pháp từ... vài thế kỷ trước rồi! Bây giờ bạn chỉ cần nhìn điểm tầm soát trên app của tụi mình "
                f"(hiện tại của bạn là <b>{prob_gb:.1%}</b>) và đi kiểm tra máu định kỳ là chuẩn khoa học nhất nha!"
            )
        # 3. Khen đẹp trai / xinh gái / tim đập nhanh / người yêu / thất tình / crush
        elif any(w in q for w in ["đẹp", "xinh", "tim đập", "người yêu", "crush", "thất tình", "yêu", "cưới", "hẹn hò", "thích"]):
            ans_m = (
                "Haha, nhìn tôi với Bác sĩ Lee Sung-kyung mà tim đập nhanh là triệu chứng 'rung động' chứ không phải tiểu đường đâu! "
                "Nhưng nói nghiêm túc nhé: nếu lúc đói bụng mà bạn bị <b>tim đập nhanh, vã mồ hôi lạnh, run tay, hoa mắt</b> thì coi chừng đó là "
                "dấu hiệu <b>hạ đường huyết</b> hoặc rối loạn dung nạp glucose đấy!"
            )
            ans_f = (
                "Cảm ơn lời khen ngọt ngào của bạn nha, làm mình cũng đỏ mặt luôn nè! Nhớ là 'ngọt ngào' trong tình cảm thì giúp giảm hormone căng thẳng (cortisol), "
                "chứ 'ngọt' ở trong mạch máu là tụi mình phải kê đơn ăn kiêng ngay đó!"
            )
        # 4. Nhịn ăn gian lận xét nghiệm / nhịn thở / uống nhiều nước lọc
        elif any(w in q for w in ["lừa", "gian lận", "nhịn", "đói", "qua mặt"]):
            ans_m = (
                "Haha, bạn định 'qua mặt' tôi và máy xét nghiệm hả? Không dễ vậy đâu! Nếu bạn nhịn đói 2 ngày trước khi khám thì chỉ làm giảm chỉ số đường huyết "
                "tức thời lúc đó thôi, nhưng chỉ số <b>HbA1c</b> sẽ 'bóc trần' lượng đường trung bình gắn vào hồng cầu của bạn suốt <b>3 tháng (90 ngày)</b> vừa qua — không chạy đi đâu được!"
            )
            ans_f = (
                "Tuyệt đối đừng nhịn ăn tiêu cực kẻo xỉu ngay trước cửa phòng khám nha! Trước khi đi xét nghiệm máu lúc đói (FPG), bạn chỉ cần nhịn ăn 8 tiếng qua đêm "
                "(từ 10h tối hôm trước, sáng hôm sau vẫn được uống nước lọc bình thường) là kết quả chuẩn nhất rồi."
            )
        # 5. Người gầy / ốm / béo / mập
        elif any(w in q for w in ["gầy", "ốm", "mập", "béo", "skinny"]):
            ans_m = (
                f"Rất nhiều người lầm tưởng 'chỉ người béo mới bị tiểu đường'! Thực tế người Việt Nam và châu Á rất hay gặp tình trạng <b>'Skinny Fat' "
                f"(nhìn ngoài gầy nhưng tích mỡ nội tạng quanh gan và tụy)</b>, nên chỉ số BMI từ 23 trở lên đã bắt đầu có nguy cơ rồi. "
                f"Hiện tại BMI của bạn là <b>{bmi_val:.1f} kg/m²</b> và điểm rủi ro là <b>{prob_gb:.1%}</b>."
            )
            ans_f = (
                "Đúng rồi đó! Dù gầy hay đầy đặn thì việc xây dựng khối cơ (tập kháng lực nhẹ, đi bộ nhanh) cũng cực kỳ quan trọng vì cơ bắp chính là 'nhà máy' tiêu thụ đường tốt nhất của cơ thể!"
            )
        # 6. Lười tập / thức khuya / ngủ / stress / học bài / thi cử
        elif any(w in q for w in ["lười", "thức khuya", "ngủ", "stress", "học", "thi", "mệt", "nằm"]):
            ans_m = (
                "Haha, ai cũng có lúc lười mà! Nhưng bạn biết không: khi bạn <b>thức khuya cày phim hay ôn thi căng thẳng</b>, cơ thể tiết ra hormone <b>Cortisol</b> "
                "làm gan bơm thêm đường vào máu và gây kháng insulin! Thiếu ngủ 1 tuần liên tục có thể làm độ nhạy insulin giảm tới 20–25% đấy."
            )
            ans_f = (
                "Nếu bạn 'lười' đi phòng gym thì áp dụng mẹo siêu dễ này của mình nha: sau mỗi bữa ăn chính, chỉ cần <b>đứng dậy đi lại nhẹ nhàng hoặc dọn dẹp phòng 10–15 phút</b> "
                "(đừng nằm ườn ngay sau khi ăn) là đã giúp giảm đỉnh đường huyết sau ăn tới 20% rồi!"
            )
        # 7. Cơm trắng / sầu riêng / xoài / trái cây / bia / rượu / cà phê
        elif any(w in q for w in ["cơm", "sầu riêng", "xoài", "trái cây", "hoa quả", "bia", "rượu", "cà phê", "cafe", "ăn gì"]):
            ans_m = (
                "Cơm trắng, xôi, bánh mì trắng hay sầu riêng, xoài chín đều có chỉ số đường huyết (GI) khá cao. Bạn không cần phải kiêng tuyệt đối khổ sở đâu, "
                "mà quan trọng là <b>khẩu phần</b> và <b>thứ tự ăn</b> để đường không tăng đột ngột!"
            )
            ans_f = (
                "Mẹo vàng của mình khi ăn cơm hoặc đồ ngon nè: Hãy <b>ăn rau xanh trước &rarr; ăn thịt/cá/đậu phụ &rarr; ăn cơm trắng cuối cùng</b>! "
                "Chất xơ từ rau sẽ tạo thành 'tấm lưới' làm chậm hấp thu đường, giúp bạn ăn ngon mà đường huyết vẫn êm ru!"
            )
        # 8. Hỏi xét nghiệm máu
        elif any(w in q for w in ["xét nghiệm", "máu", "test", "fpg", "hba1c", "khám"]):
            ans_m = (
                f"Bạn nên yêu cầu bác sĩ làm <b>Xét nghiệm đường huyết lúc đói (FPG — nhịn ăn 8 tiếng)</b> hoặc <b>HbA1c</b> "
                f"(đánh giá đường huyết trung bình 3 tháng). Với điểm số <b>{prob_gb:.1%}</b> của bạn, đây là cách kiểm tra chính xác nhất!"
            )
            ans_f = (
                "Trước ngày đi lấy máu xét nghiệm FPG, tối hôm trước bạn nhớ nhịn ăn từ 22h đêm (chỉ uống nước lọc) và ngủ sớm để kết quả chuẩn nhất nhé!"
            )
        # 9. Hỏi về BMI / thực đơn / giảm cân
        elif any(w in q for w in ["bmi", "cân", "dinh dưỡng", "thực đơn", "tập"]):
            ans_m = (
                f"Khi giảm BMI từ <b>{bmi_val:.1f}</b> xuống <b>{lower_bmi:.1f} kg/m²</b>, độ nhạy insulin của cơ thể cải thiện rõ rệt, "
                f"giúp điểm rủi ro trên mô hình giảm từ <b>{prob_gb:.1%}</b> xuống <b>{prob_whatif:.1%}</b>."
            )
            ans_f = (
                f"Với chiều cao <b>{height_cm} cm</b> và cân nặng <b>{weight_kg} kg</b> (BMI <b>{bmi_val:.1f}</b>), bạn hãy áp dụng quy tắc Đĩa thức ăn: "
                f"<b>1/2 rau xanh</b>, <b>1/4 đạm lành mạnh</b> (cá, ức gà, đậu phụ), <b>1/4 tinh bột nguyên cám</b>, kết hợp đi bộ nhanh 30 phút mỗi ngày nha!"
            )
        # 10. Ngưỡng 0.18
        elif any(w in q for w in ["0.18", "ngưỡng", "threshold", "0.50", "chi phí"]):
            ans_m = (
                "Chúng tôi chọn ngưỡng <b>0.18</b> trên tập Validation vì bỏ sót 1 ca tiểu đường nguy hiểm gấp <b>5 lần</b> "
                "so với 1 lần xét nghiệm kiểm tra. Nhờ ngưỡng 0.18, mô hình phát hiện được tới <b>70.5%</b> số ca bệnh!"
            )
            ans_f = "Phát hiện sớm ở giai đoạn tiền tiểu đường giúp bạn chỉ cần điều chỉnh lối sống nhẹ nhàng mà không phải uống thuốc suốt đời!"
        # 11. Mọi câu hỏi ngẫu nhiên / hài hước / "ngu ngốc" khác
        else:
            advice_tail = (
                "Vì điểm của bạn đang trên ngưỡng 0.18 nên nhớ đi làm xét nghiệm máu kiểm tra nhé!"
                if is_flagged
                else "May mắn là điểm của bạn hiện vẫn đang trong vùng an toàn dưới 0.18 đó!"
            )
            ans_m = (
                f"Haha, câu hỏi <i>\"{question_text}\"</i> của bạn thú vị thật đấy! Dù bạn hỏi câu gì thì với tư cách là bác sĩ nội tiết, "
                f"tôi vẫn phải nhắc nhẹ: Hồ sơ của bạn đang có điểm rủi ro là <b>{prob_gb:.1%}</b> (yếu tố chính: <b>{top_feat_label}</b>). {advice_tail}"
            )
            ans_f = (
                f"Bác sĩ Gong Yoo lúc nào cũng nghiêm túc hết! Để mình trả lời thêm cho bạn nha: với chỉ số BMI <b>{bmi_val:.1f} kg/m²</b> "
                f"({height_cm} cm, {weight_kg} kg), bạn cứ giữ tinh thần vui vẻ, cười nhiều như lúc hỏi câu này, bớt đồ ngọt và vận động đều đặn là cơ thể luôn khỏe mạnh!"
            )
    elif lang_code == "zh":
        if any(w in q for w in ["奶茶", "珍珠", "糖", "可乐", "甜"]):
            ans_m = f"哈哈，“无糖奶茶加满黑糖珍珠”可是最经典的血糖陷阱！珍珠本身就是高升糖碳水化合物。结合您当前 <b>{prob_gb:.1%}</b> 的风险得分，偶尔喝一杯纯茶可以，天天喝珍珠奶茶可不行哦！"
            ans_f = f"如果您实在想喝，可以换成无糖乌龙茶或绿茶，配料选奇亚籽或无糖芦荟，既能保持您 <b>{bmi_val:.1f}</b> 的 BMI 身材，又不升血糖！"
        elif any(w in q for w in ["蚂蚁", "尿"]):
            ans_m = "听起来像玩笑，但医学史上是真的！当血糖严重超标（>180 mg/dL）时，糖分会随尿液排出招来蚂蚁。不过现在做个空腹血糖 (FPG) 只要几分钟，可别等蚂蚁来提醒您！"
            ans_f = f"哈哈，用我们的 AI 筛查工具（您当前得分为 <b>{prob_gb:.1%}</b>）比找蚂蚁科学多啦！"
        elif any(w in q for w in ["帅", "美", "心跳", "喜欢", "恋爱"]):
            ans_m = "哈哈，看到我和李圣经医生心跳加速是“心动”，不是糖尿病！但如果在饥饿时出现心慌、手抖、出冷汗，那要小心低血糖或胰岛素抵抗哦！"
            ans_f = "谢谢您的夸奖！语言甜一点有益身心，但血液太“甜”可就不健康啦！"
        else:
            ans_m = f"哈哈，您问的<i>“{question_text}”</i>很有意思！结合您的档案：当前风险得分为 <b>{prob_gb:.1%}</b>（阈值 <b>{active_th:.2f}</b>，关键因素：<b>{top_feat_label}</b>）。建议定期检查空腹血糖或 HbA1c。"
            ans_f = f"您的 BMI 为 <b>{bmi_val:.1f} kg/m²</b>，保持愉快心情、充足睡眠和每天 30 分钟运动，就能把风险降到 <b>{prob_whatif:.1%}</b>！"
    elif lang_code == "fr":
        if any(w in q for w in ["bubble", "tapioca", "sucre", "bonbon", "soda"]):
            ans_m = f"Haha, le « bubble tea 0% sucre mais plein de perles de tapioca au caramel » est un piège classique ! Les perles sont de l'amidon pur à indice glycémique élevé. Avec votre score de <b>{prob_gb:.1%}</b>, gardez-le pour les occasions rares !"
            ans_f = f"Optez plutôt pour un thé vert ou oolong sans sucre avec des graines de chia pour protéger votre IMC (actuellement <b>{bmi_val:.1f} kg/m²</b>) !"
        elif any(w in q for w in ["fourmi", "urine"]):
            ans_m = "C'est une vraie anecdote historique ! Le mot « diabète sucré » (mellitus) vient du fait que le glucose déborde dans les urines quand la glycémie est très élevée. Mais une prise de sang FPG est bien plus fiable que les fourmis !"
            ans_f = f"Heureusement, notre outil IA (votre score : <b>{prob_gb:.1%}</b>) est beaucoup plus moderne !"
        else:
            ans_m = f"Haha, excellente question (<i>« {question_text} »</i>) ! Côté médical, votre score est de <b>{prob_gb:.1%}</b> (seuil <b>{active_th:.2f}</b>, facteur principal : <b>{top_feat_label}</b>)."
            ans_f = f"Et avec votre IMC de <b>{bmi_val:.1f} kg/m²</b>, garder la bonne humeur, bien dormir et marcher 30 min par jour réduira votre risque à <b>{prob_whatif:.1%}</b> !"
    else:
        if any(w in q for w in ["boba", "bubble tea", "tapioca", "candy", "soda", "coke", "sugar", "sweet"]):
            ans_m = (
                f"Haha, '0% sugar boba tea with double brown-sugar tapioca pearls and cheese foam' is the oldest trick in the book! "
                f"Tapioca pearls are pure fast-digesting starch soaked in syrup. With your score at <b>{prob_gb:.1%}</b>, enjoy plain unsweetened tea instead!"
            )
            ans_f = (
                f"Don't let the '0% sugar' label fool you! Swap tapioca pearls for chia seeds or aloe vera to keep your BMI (currently <b>{bmi_val:.1f} kg/m²</b>) and blood sugar happy!"
            )
        elif any(w in q for w in ["ant", "pee", "urine"]):
            ans_m = (
                "It sounds funny, but in ancient medicine physicians actually noticed ants gathering where people with untreated diabetes urinated "
                "(when blood glucose exceeds ~180 mg/dL, sugar spills into urine). Thankfully, today a 5-minute Fasting Glucose blood test is much easier!"
            )
            ans_f = f"Haha, no need to recruit ants when our AI screening tool already calculated your score at <b>{prob_gb:.1%}</b>!"
        elif any(w in q for w in ["handsome", "pretty", "beautiful", "heart beat", "crush", "love", "cute"]):
            ans_m = (
                "Haha, a fast heartbeat while looking at me or Dr. Lee Sung-kyung is called a 'crush', not diabetes! "
                "However, if you ever feel a racing heart, shaky hands, and cold sweats when hungry, that can be a sign of reactive hypoglycemia!"
            )
            ans_f = "Aww, thank you for the sweet compliment! Just remember: being sweet in conversation is great for your heart, but too much sugar in your bloodstream isn't!"
        elif any(w in q for w in ["trick", "cheat", "starve", "fasting"]):
            ans_m = (
                "Haha, trying to 'cram' for a blood test by starving yourself for a day won't fool the <b>HbA1c</b> test! "
                "HbA1c measures your average blood sugar attached to red blood cells over the entire past <b>90 days</b>!"
            )
            ans_f = "Please don't starve yourself! Just fast normally for 8 hours overnight (plain water is totally fine) before a morning Fasting Plasma Glucose test."
        else:
            ans_m = (
                f"Haha, I love how you asked <i>\"{question_text}\"</i>! Tying that back to your health check: your estimated risk score is "
                f"<b>{prob_gb:.1%}</b> vs. our <b>{active_th:.2f}</b> cutoff (top factor: <b>{top_feat_label}</b>)."
            )
            ans_f = (
                f"And with your calculated BMI of <b>{bmi_val:.1f} kg/m²</b>, staying active, sleeping 7–8 hours, and keeping a great sense of humor will help bring your risk down toward <b>{prob_whatif:.1%}</b>!"
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
    reply_m, reply_f = generate_doctor_replies(current_q)

    st.markdown(
        f"<div style='background:#f1f5f9; border:1px solid #cbd5e1; color:#0f172a; padding:10px 16px; border-radius:12px; font-weight:700; margin:10px 0;'>"
        f"🙋 {current_q}</div>",
        unsafe_allow_html=True,
    )

    if doc_mode in ["both", "male"]:
        st.markdown(
            "<div style='display:flex; align-items:flex-start; gap:14px; background:#f8fafc; border:1px solid #cbd5e1; border-left:5px solid #2563eb; "
            "padding:14px 16px; border-radius:12px; margin-bottom:10px;'>"
            f"<img src='data:image/jpeg;base64,{DOC_MALE_B64}' style='width:54px; height:54px; object-fit:cover; border-radius:50%; border:2px solid #2563eb; flex-shrink:0;'/>"
            f"<div style='color:#1e293b; font-size:0.93rem; line-height:1.55;'><b style='color:#1d4ed8;'>{d_cfg['m_name']}:</b> {reply_m}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    if doc_mode in ["both", "female"]:
        st.markdown(
            "<div style='display:flex; align-items:flex-start; gap:14px; background:#f8fafc; border:1px solid #cbd5e1; border-left:5px solid #0d9488; "
            "padding:14px 16px; border-radius:12px; margin-bottom:4px;'>"
            f"<img src='data:image/jpeg;base64,{DOC_FEMALE_B64}' style='width:54px; height:54px; object-fit:cover; border-radius:50%; border:2px solid #0d9488; flex-shrink:0;'/>"
            f"<div style='color:#1e293b; font-size:0.93rem; line-height:1.55;'><b style='color:#0f766e;'>{d_cfg['f_name']}:</b> {reply_f}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

# ==============================================================================
# 5. DETAILED RESULTS, RISK SPECTRUM & TOP FACTORS
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
