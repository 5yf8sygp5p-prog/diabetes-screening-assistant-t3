"""
app.py — Diabetes Screening Assistant with EIU Logo & Interactive AI Detective Assistant (4 Languages)
======================================================================================================
User-Decision Statement:
"An adult answering questions at home uses the app to decide whether to get a blood sugar test."
"""

import base64
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import i18n
from i18n import (
    AGE_OPTIONS,
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


def get_image_b64(attr_name: str, filename: str) -> str:
    val = getattr(i18n, attr_name, "")
    if val:
        return val
    base_dir = Path(__file__).resolve().parent
    for cand in [base_dir / filename, base_dir.parent / filename]:
        if cand.exists():
            return base64.b64encode(cand.read_bytes()).decode("ascii")
    return ""


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

eiu_b64 = get_image_b64("EIU_LOGO_B64", "eiu_logo.png")
mascot_b64 = get_image_b64("AI_MASCOT_B64", "ai_assistant.png")

default_th = float(meta["cost_assumptions"]["chosen_threshold"])
if "active_threshold" not in st.session_state:
    st.session_state["active_threshold"] = default_th
active_th = float(st.session_state["active_threshold"])

# ==============================================================================
# 1. HEADER SECTION WITH EIU UNIVERSITY LOGO
# ==============================================================================
logo_img_tag = (
    f"<img src='data:image/png;base64,{eiu_b64}' style='max-height:64px; width:auto; object-fit:contain;'/>"
    if eiu_b64
    else ""
)

header_html = (
    "<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px; "
    "background:#ffffff; border:1px solid #e2e8f0; border-radius:16px; padding:20px 26px; margin-bottom:16px; "
    "box-shadow:0 4px 14px rgba(15,23,42,0.04);'>"
    "<div style='flex:1; min-width:280px;'>"
    "<div style='font-size:0.76rem; font-weight:700; color:#9a7b3e; text-transform:uppercase; letter-spacing:0.6px; margin-bottom:4px;'>"
    "🏛️ Eastern International University (EIU) &bull; Healthcare Decision-Support System"
    "</div>"
    f"<h1 style='font-size:2.15rem; font-weight:800; color:#1e293b; margin:0;'>{t['app_title']}</h1>"
    "</div>"
    f"<div style='text-align:right;'>{logo_img_tag}</div>"
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
# 3. COMPUTE SCORE & RENDER AI DETECTIVE ASSISTANT (INSPECTOR MEGURE)
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
fname_map = FEATURE_LABELS_I18N.get(lang_code, FEATURE_LABELS_I18N["en"])
sorted_c = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
top_feat_label = fname_map.get(sorted_c[0][0], sorted_c[0][0])

# AI Assistant Speech Bubble Content (4 Languages)
AI_SPEECH = {
    "en": {
        "title": "🕵️‍♂️ AI Health Detective Assistant",
        "idle": (
            "\"Hmm... Let me examine the clues! Fill in the 7 everyday health questions above and press "
            "<b>Analyze</b>. I will compare your profile against 253,680 US survey records and deduce whether you "
            "should ask your doctor for a blood glucose test!\""
        ),
        "flagged": (
            f"\"Aha! Based on my investigation, your estimated risk score is <b>{prob_gb:.1%}</b>, which is above our "
            f"screening cutoff of <b>{active_th:.2f}</b> (your calculated BMI is <b>{bmi_val:.1f} kg/m²</b>, and your strongest "
            f"factor is <b>{top_feat_label}</b>). My recommendation: <b>Schedule a routine Fasting Glucose or HbA1c blood test "
            f"with your doctor</b> just to be safe!\""
        ),
        "safe": (
            f"\"Good news from my investigation! Your estimated risk score is <b>{prob_gb:.1%}</b>, which stays below our "
            f"screening threshold of <b>{active_th:.2f}</b> (with a calculated BMI of <b>{bmi_val:.1f} kg/m²</b>). "
            f"No urgent test is flagged today — keep up a healthy lifestyle and routine annual checkups!\""
        ),
        "faq_label": "💬 Ask the AI Detective a quick question:",
        "faqs": {
            "Why is the cutoff 0.18 instead of 0.50?": (
                "Because missing undiagnosed diabetes (False Negative) is judged ~5 times more costly than taking a simple "
                "blood test (False Positive). Lowering the threshold to 0.18 catches over 70% of true cases!"
            ),
            "What blood test should I ask my doctor for?": (
                "Ask for a Fasting Plasma Glucose (FPG) test or an HbA1c test. They are quick, inexpensive standard lab tests "
                "that clinically measure your blood sugar."
            ),
            "Does a high score mean I definitely have diabetes?": (
                "No! This tool only measures how similar your survey answers are to people who reported diabetes/prediabetes "
                "in the 2015 CDC survey. Only a doctor's lab test can diagnose diabetes."
            ),
        },
    },
    "vi": {
        "title": "🕵️‍♂️ Trợ lý Thám tử AI Sức khỏe (Thanh tra Megure)",
        "idle": (
            "\"Hmm... Để ta suy luận xem nào! Bạn hãy điền 7 manh mối sức khỏe ở bảng trên rồi bấm nút "
            "<b>Phân tích ngay</b> nhé. Ta sẽ đối chiếu hồ sơ của bạn với 253.680 ca khảo sát CDC để kết luận xem bạn "
            "có cần đi xét nghiệm đường huyết không!\""
        ),
        "flagged": (
            f"\"À há! Qua điều tra các manh mối, điểm rủi ro của bạn là <b>{prob_gb:.1%}</b>, vượt qua ngưỡng cảnh báo "
            f"<b>{active_th:.2f}</b> (chỉ số BMI của bạn là <b>{bmi_val:.1f} kg/m²</b>, và yếu tố tác động mạnh nhất là "
            f"<b>{top_feat_label}</b>). Lời khuyên của ta: <b>Bạn nên đến gặp bác sĩ để làm xét nghiệm đường huyết lúc đói (FPG) "
            f"hoặc HbA1c</b> kiểm tra cho yên tâm nhé!\""
        ),
        "safe": (
            f"\"Tin tốt đây! Sau khi xem xét kỹ 7 manh mối, điểm rủi ro của bạn chỉ ở mức <b>{prob_gb:.1%}</b>, thấp hơn "
            f"ngưỡng cắt <b>{active_th:.2f}</b> (chỉ số BMI là <b>{bmi_val:.1f} kg/m²</b>). Hiện tại chưa có dấu hiệu đáng ngại — "
            f"hãy tiếp tục ăn uống lành mạnh, tập thể dục và khám sức khỏe định kỳ nhé!\""
        ),
        "faq_label": "💬 Hỏi nhanh Trợ lý Thám tử AI:",
        "faqs": {
            "Tại sao ngưỡng cắt lại là 0.18 mà không phải 0.50?": (
                "Bởi vì việc bỏ sót bệnh tiểu đường (False Negative) nguy hiểm gấp khoảng 5 lần so với việc đi xét nghiệm máu "
                "kiểm tra (False Positive). Hạ ngưỡng xuống 0.18 giúp phát hiện tới hơn 70% số ca bệnh thực tế!"
            ),
            "Nếu đi khám thì tôi nên hỏi bác sĩ xét nghiệm gì?": (
                "Bạn hãy hỏi bác sĩ làm xét nghiệm Đường huyết lúc đói (Fasting Plasma Glucose - FPG) hoặc xét nghiệm HbA1c. "
                "Đây là 2 xét nghiệm máu cơ bản, nhanh chóng và chính xác nhất."
            ),
            "Điểm cao có nghĩa là tôi chắc chắn bị tiểu đường không?": (
                "Không phải nhé! Điểm số này chỉ cho biết câu trả lời của bạn giống với những người có nguy cơ trong khảo sát "
                "CDC 2015. Chỉ có xét nghiệm máu tại cơ sở y tế mới chẩn đoán được bệnh."
            ),
        },
    },
    "zh": {
        "title": "🕵️‍♂️ AI 健康侦探助手（目暮警官）",
        "idle": (
            "“嗯……让我来推理一下！请填写上方的 7 个日常健康线索并点击 <b>立即分析</b>。"
            "我将比对 253,680 份美国疾控中心调查记录，为您判断是否需要去医院做血糖检测！”"
        ),
        "flagged": (
            f"“原来如此！根据我的调查，您的风险得分为 <b>{prob_gb:.1%}</b>，高于筛查阈值 <b>{active_th:.2f}</b>"
            f"（您的 BMI 为 <b>{bmi_val:.1f} kg/m²</b>，影响最大的因素是 <b>{top_feat_label}</b>）。"
            f"我的建议：<b>请咨询医生进行空腹血糖 (FPG) 或糖化血红蛋白 (HbA1c) 检测</b>！”"
        ),
        "safe": (
            f"“好消息！经过核查，您的风险得分为 <b>{prob_gb:.1%}</b>，低于筛查阈值 <b>{active_th:.2f}</b>"
            f"（BMI 为 <b>{bmi_val:.1f} kg/m²</b>）。目前无需紧急检测，请继续保持健康饮食与规律运动！”"
        ),
        "faq_label": "💬 向 AI 侦探助手提问：",
        "faqs": {
            "为什么阈值是 0.18 而不是 0.50？": "因为漏诊糖尿病的代价约为一次普通验血的 5 倍。将阈值设为 0.18 可以捕获超过 70% 的潜在患者！",
            "去医院应该向医生咨询哪项检测？": "您可以咨询空腹血浆葡萄糖 (FPG) 或糖化血红蛋白 (HbA1c) 血液检查。",
            "得分高代表我一定患有糖尿病吗？": "不是！本工具仅反映问卷回答的统计相似度，唯有医院血液化验才能确诊。",
        },
    },
    "fr": {
        "title": "🕵️‍♂️ Assistant IA Détective Santé (Inspecteur Megure)",
        "idle": (
            "« Hmm... Laissez-moi examiner les indices ! Remplissez les 7 questions ci-dessus et cliquez sur "
            "<b>Analyser</b>. Je vais comparer votre profil aux 253 680 réponses de l'enquête CDC ! »"
        ),
        "flagged": (
            f"« Eurêka ! D'après mon enquête, votre score de risque est de <b>{prob_gb:.1%}</b>, ce qui dépasse notre seuil "
            f"de <b>{active_th:.2f}</b> (IMC calculé : <b>{bmi_val:.1f} kg/m²</b>, facteur principal : <b>{top_feat_label}</b>). "
            f"Mon conseil : <b>Demandez un test sanguin (glycémie à jeun ou HbA1c) à votre médecin</b> par précaution ! »"
        ),
        "safe": (
            f"« Bonne nouvelle ! Votre score de risque est de <b>{prob_gb:.1%}</b>, bien en dessous du seuil de "
            f"<b>{active_th:.2f}</b> (IMC : <b>{bmi_val:.1f} kg/m²</b>). Aucun test urgent n'est nécessaire aujourd'hui ! »"
        ),
        "faq_label": "💬 Posez une question rapide au Détective IA :",
        "faqs": {
            "Pourquoi le seuil est-il de 0.18 au lieu de 0.50 ?": (
                "Manquer un cas de diabète est jugé ~5 fois plus grave qu'un test sanguin inutile. Le seuil de 0.18 détecte plus de 70 % des cas !"
            ),
            "Quel test sanguin dois-je demander à mon médecin ?": (
                "Demandez un test de glycémie à jeun (FPG) ou d'hémoglobine glyquée (HbA1c)."
            ),
            "Un score élevé signifie-t-il que j'ai le diabète ?": (
                "Non ! Cet outil estime uniquement une similarité statistique. Seul un test de laboratoire permet un diagnostic."
            ),
        },
    },
}

ai_cfg = AI_SPEECH.get(lang_code, AI_SPEECH["en"])
has_run = st.session_state.get("has_analyzed", False)
speech_text = ai_cfg["Idle"] if not has_run else (ai_cfg["flagged"] if is_flagged else ai_cfg["safe"]) if "Idle" in ai_cfg else (ai_cfg["idle"] if not has_run else (ai_cfg["flagged"] if is_flagged else ai_cfg["safe"]))

mascot_img_tag = (
    f"<img src='data:image/png;base64,{mascot_b64}' style='max-height:145px; width:auto; object-fit:contain; filter:drop-shadow(0 4px 8px rgba(0,0,0,0.15));'/>"
    if mascot_b64
    else "🕵️‍♂️"
)

assistant_card_html = (
    "<div style='display:flex; align-items:center; gap:20px; background:linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); "
    "border:2px solid #f59e0b; border-radius:16px; padding:16px 22px; margin-top:14px; margin-bottom:16px; "
    "box-shadow:0 6px 18px rgba(245,158,11,0.12);'>"
    f"<div style='flex-shrink:0; text-align:center;'>{mascot_img_tag}</div>"
    "<div style='flex:1; background:#ffffff; border:1px solid #fcd34d; border-radius:14px; padding:14px 18px; position:relative;'>"
    f"<div style='font-weight:800; color:#92400e; font-size:0.98rem; margin-bottom:6px;'>{ai_cfg['title']}</div>"
    f"<div style='color:#1e293b; font-size:0.94rem; line-height:1.55;'>{speech_text}</div>"
    "</div>"
    "</div>"
)
st.markdown(assistant_card_html, unsafe_allow_html=True)

if not has_run:
    st.info(t["prompt_analyze"])

# Interactive FAQ with the AI Detective Assistant
with st.expander(ai_cfg["faq_label"]):
    faq_q = st.selectbox("", options=list(ai_cfg["faqs"].keys()), label_visibility="collapsed")
    st.markdown(f"**🕵️‍♂️ {ai_cfg['title']}:** {ai_cfg['faqs'][faq_q]}")

# ==============================================================================
# 4. DETAILED RESULTS, RISK SPECTRUM & TOP FACTORS
# ==============================================================================
if has_run:
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
