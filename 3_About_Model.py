"""
pages/3_About_Model.py — About the Model, 4-Model Comparison, Permutation Importance & Limitations
==================================================================================================
"""

import altair as alt
import pandas as pd
import streamlit as st
from i18n import FEATURE_LABELS_I18N, inject_friendly_css, load_artifacts, render_sidebar_and_get_lang

st.set_page_config(
    page_title="About Model | Diabetes Screening Assistant",
    page_icon="📖",
    layout="wide",
)

inject_friendly_css()
lang_code, t = render_sidebar_and_get_lang()
_, _, meta = load_artifacts()

ds = meta["dataset_summary"]
app_m = meta["app_model_test_metrics"]
th_default = float(meta["cost_assumptions"]["chosen_threshold"])
fname_map = FEATURE_LABELS_I18N.get(lang_code, FEATURE_LABELS_I18N["en"])

st.title(t["ab_title"])
st.markdown(f"<div class='edu-warning-box'>{t['disclaimer_long']}</div>", unsafe_allow_html=True)

st.subheader(t["ab_decision_title"])
st.markdown(t["ab_decision_text"])

st.subheader(t["ab_data_title"])
if lang_code == "vi":
    st.markdown(
        f"- **Nguồn dữ liệu:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015, *Diabetes Health Indicators* (`diabetes_binary_health_indicators_BRFSS2015.csv`), **{ds['n_raw']:,}** câu trả lời khảo sát.\n"
        f"- **Biến mục tiêu:** `Diabetes_binary = 1` nếu người trả lời **báo cáo** mắc tiểu đường hoặc tiền tiểu đường (nhãn tự khai báo).\n"
        f"- **Làm sạch dữ liệu:** Tập dữ liệu sạch gồm **{ds['n_clean']:,}** dòng (**{ds['prevalence_clean']:.1%}** nhãn dương tính). Không có giá trị khuyết thiếu.\n"
        f"- **Phân chia dữ liệu (Split):** Phân tầng **60 / 20 / 20** (Train **{ds['n_train']:,}** · Validation **{ds['n_val']:,}** · Test **{ds['n_test']:,}**). Ngưỡng cắt được chọn trên tập Validation; mọi chỉ số báo cáo đều tính trên tập Test độc lập.\n"
        f"- **Pipeline:** Toàn bộ bước tiền xử lý (`ColumnTransformer`) và mô hình được đóng gói chung trong một `scikit-learn` `Pipeline` (`models/pipeline.joblib` & `models/inydb`)."
    )
elif lang_code == "zh":
    st.markdown(
        f"- **数据来源：** 美国疾控中心 (CDC) BRFSS 2015 糖尿病健康指标调查 (`diabetes_binary_health_indicators_BRFSS2015.csv`)，共 **{ds['n_raw']:,}** 份调查回复。\n"
        f"- **目标变量：** `Diabetes_binary = 1` 表示受访者**自我报告**患有糖尿病或糖尿病前期。\n"
        f"- **数据清洗：** 清洗后保留 **{ds['n_clean']:,}** 行（阳性比例 **{ds['prevalence_clean']:.1%}**），无缺失值。\n"
        f"- **数据集划分：** 分层抽样 **60 / 20 / 20**（训练集 **{ds['n_train']:,}** · 验证集 **{ds['n_val']:,}** · 测试集 **{ds['n_test']:,}**）。阈值在验证集上选出，所有最终指标均在独立测试集上评估。\n"
        f"- **Pipeline 封装：** 预处理 (`ColumnTransformer`) 与分类模型保存在同一个 `scikit-learn` `Pipeline` 中 (`models/pipeline.joblib` 与 `models/inydb`)。"
    )
elif lang_code == "fr":
    st.markdown(
        f"- **Source :** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 (`diabetes_binary_health_indicators_BRFSS2015.csv`), **{ds['n_raw']:,}** réponses.\n"
        f"- **Cible :** `Diabetes_binary = 1` si le répondant a **déclaré** un diabète ou un prédiabète.\n"
        f"- **Nettoyage :** **{ds['n_clean']:,}** lignes (**{ds['prevalence_clean']:.1%}** positives), aucune valeur manquante.\n"
        f"- **Séparation :** Stratifiée **60 / 20 / 20** (entraînement **{ds['n_train']:,}** · validation **{ds['n_val']:,}** · test **{ds['n_test']:,}**). Le seuil est choisi sur la validation ; les métriques sont évaluées sur le jeu de test.\n"
        f"- **Pipeline :** Le prétraitement (`ColumnTransformer`) et le modèle sont enregistrés ensemble dans un seul `Pipeline` `scikit-learn`."
    )
else:
    st.markdown(
        f"- **Source:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015, *Diabetes Health Indicators* (`diabetes_binary_health_indicators_BRFSS2015.csv`), **{ds['n_raw']:,}** survey responses, 21 health indicators.\n"
        f"- **Target:** `Diabetes_binary = 1` if the respondent **reported** diabetes or prediabetes (self-reported label).\n"
        f"- **Cleaning:** **{ds['n_clean']:,}** clean survey rows (**{ds['prevalence_clean']:.1%}** positive label). No missing values.\n"
        f"- **Split:** stratified **60 / 20 / 20** (train **{ds['n_train']:,}** · validation **{ds['n_val']:,}** · test **{ds['n_test']:,}**). The threshold is chosen on validation; all reported metrics use the untouched test set.\n"
        f"- **Pipeline:** preprocessing (`ColumnTransformer`) and model are saved together in one scikit-learn `Pipeline` (`models/pipeline.joblib` and `models/inydb`)."
    )

# Models Compared Table
st.subheader(t["ab_models_title"])
mc_rows = ""
for m in meta["models_compared"]:
    mc_rows += (
        f"<tr>"
        f"<td>{m['model']}</td>"
        f"<td style='text-align:right;'>{m['questions']}</td>"
        f"<td style='text-align:right;'>{m['pr_auc_train']:.3f}</td>"
        f"<td style='text-align:right;'>{m['pr_auc_test']:.3f}</td>"
        f"<td style='text-align:right;'>{m['roc_auc_test']:.3f}</td>"
        f"<td style='text-align:right;'>{m['recall']:.3f}</td>"
        f"<td style='text-align:right;'>{m['precision']:.3f}</td>"
        f"<td style='text-align:right;'>{m['f1']:.3f}</td>"
        f"</tr>"
    )

mc_table = (
    "<table class='clean-table'>"
    "<thead>"
    "<tr>"
    "<th>Model</th><th style='text-align:right;'>Questions</th>"
    "<th style='text-align:right;'>PR-AUC (train)</th><th style='text-align:right;'>PR-AUC (test)</th>"
    "<th style='text-align:right;'>ROC-AUC (test)</th><th style='text-align:right;'>Recall</th>"
    "<th style='text-align:right;'>Precision</th><th style='text-align:right;'>F1</th>"
    "</tr>"
    "</thead>"
    f"<tbody>{mc_rows}</tbody>"
    "</table>"
)
st.markdown(mc_table, unsafe_allow_html=True)
st.caption(
    "Recall / precision / F1 use each model's own threshold (cost ratio 5:1, tuned on validation). "
    "PR-AUC of a random scorer would be about 0.14 (the positive rate), so 0.42–0.44 is a clear signal."
)
st.markdown(
    "**Model in the app:** Histogram Gradient Boosting using only **7 easy questions** "
    "(Age group, Body mass index (BMI), High blood pressure, High cholesterol, Self-rated general health, "
    "Difficulty walking / climbing stairs, Heart disease or heart attack). "
    "Rule: best validation PR-AUC among the short-questionnaire models. Using the short questionnaire instead of all variables "
    "costs almost nothing in PR-AUC, and every question can be answered at home without a lab test."
)

# Performance of the app model (test set)
st.subheader(t["ab_perf_title"])
p1, p2, p3, p4, p5 = st.columns(5)
p1.metric("PR-AUC", f"{app_m['pr_auc_test']:.3f}")
p2.metric("ROC-AUC", f"{app_m['roc_auc_test']:.3f}")
p3.metric("Recall", f"{app_m['recall']:.1%}")
p4.metric("Precision", f"{app_m['precision']:.1%}")
p5.metric("False-positive rate", f"{app_m['fpr']:.1%}")
st.caption(
    f"At threshold {th_default:.2f}. With an imbalanced label (~14% positive), PR-AUC and Recall/FPR are far more informative than raw accuracy."
)

st.subheader(t["ab_thresh_title"])
st.markdown(t["ab_thresh_text"].format(th=f"{th_default:.2f}"))

# Which answers matter most?
st.subheader(t["ab_matters_title"])
w_left, w_right = st.columns([1.2, 1.0], gap="large")

with w_left:
    perm_df = pd.DataFrame(meta["permutation_importance"])
    perm_df["label"] = perm_df["feature"].map(lambda k: fname_map.get(k, k))
    perm_df["lo"] = perm_df["importance_mean"] - perm_df["importance_std"]
    perm_df["hi"] = perm_df["importance_mean"] + perm_df["importance_std"]

    bars_p = (
        alt.Chart(perm_df)
        .mark_bar(color="#1f618d", height=22)
        .encode(
            y=alt.Y("label:N", sort="-x", title=""),
            x=alt.X("importance_mean:Q", title="Drop in PR-AUC when the answer is shuffled"),
            tooltip=[
                alt.Tooltip("label:N", title="Question"),
                alt.Tooltip("importance_mean:Q", format=".4f", title="Mean PR-AUC Drop"),
            ],
        )
    )
    err_p = (
        alt.Chart(perm_df)
        .mark_errorbar(color="#0f172a")
        .encode(
            y=alt.Y("label:N", sort="-x"),
            x=alt.X("lo:Q", title=""),
            x2="hi:Q",
        )
    )
    st.markdown(f"**{t['ab_perm_title']}**")
    st.altair_chart((bars_p + err_p).properties(height=280), use_container_width=True)

with w_right:
    st.markdown(f"**{t['ab_or_title']}**")
    or_rows = ""
    for item in meta["odds_ratios"]:
        lbl = fname_map.get(item["feature"], item["feature"])
        or_rows += f"<tr><td>{lbl}</td><td style='text-align:right;'><b>{item['odds_ratio']:.2f}</b></td></tr>"
    or_table = (
        "<table class='clean-table'>"
        "<thead><tr><th>Answer</th><th style='text-align:right;'>Odds ratio</th></tr></thead>"
        f"<tbody>{or_rows}</tbody>"
        "</table>"
    )
    st.markdown(or_table, unsafe_allow_html=True)

st.caption("Importance describes what the model relies on. It does not show that any factor causes diabetes.")

with st.expander("Why these 7 questions? (Plain-English justification & zero data leakage)"):
    st.markdown(
        "1. **Zero Data Leakage:** All 7 questions (`Age`, `BMI` from height & weight, `HighBP`, `HighChol`, `GenHlth`, `DiffWalk`, `HeartDiseaseorAttack`) are known by an adult at home *before* ordering a blood test.\n"
        "2. **Top Signal Concentration:** In permutation importance across the survey, `GenHlth`, `BMI`, `Age`, `HighBP`, and `HighChol` account for over **94%** of total predictive signal.\n"
        "3. **Privacy & Low Friction:** Excluding income and education from the user intake form makes the screening assistant faster (under 1 minute) and less intrusive while preserving `ROC-AUC = 0.822`."
    )

# 6 Limitations
st.subheader(t["ab_limits_title"])
if lang_code == "vi":
    st.markdown(
        "1. **Nhãn và câu trả lời tự khai báo (Self-reported):** Không có xét nghiệm máu thực tế nào được đo trong khảo sát; 'tiểu đường hoặc tiền tiểu đường' là do người dân tự báo cáo lại lời bác sĩ.\n"
        "2. **Quần thể người trưởng thành Hoa Kỳ năm 2015:** Dữ liệu không đại diện trực tiếp cho người Việt Nam hoặc châu Á. Ở người châu Á, nguy cơ tiểu đường đã tăng cao ở mức BMI thấp hơn ($\ge 23\text{ kg/m}^2$), nên mô hình có thể đánh giá thấp rủi ro ở người châu Á có tạng người nhỏ.\n"
        "3. **Tương quan thống kê, không phải nhân quả (Association, not cause):** Điểm quan trọng của biến chỉ phản ánh quy luật dự báo, không chứng minh quan hệ nhân quả y học.\n"
        "4. **Độ phân tách mức độ vừa phải:** Khoảng 34% số người được gắn cờ thực sự có báo cáo bệnh (Precision) và khoảng 28–29% số ca bệnh bị bỏ sót ở ngưỡng mặc định.\n"
        "5. **Sai số khác nhau giữa các nhóm kinh tế – xã hội:** Xem chi tiết tại trang *Công bằng Y tế (Health Equity)*; một ngưỡng cắt duy nhất không mang lại cùng tỷ lệ sai số cho mọi nhóm thu nhập.\n"
        "6. **Công cụ khảo sát, không phải phòng khám:** Hãy xem kết quả như một gợi ý để hỏi ý kiến bác sĩ, tuyệt đối không dùng làm kết luận chẩn đoán."
    )
elif lang_code == "zh":
    st.markdown(
        "1. **自我报告标签与回答偏差：** 调查未直接测量受访者血糖，标签仅来自受访者是否曾被医生告知患病（约五分之一的糖尿病患者尚未确诊）。\n"
        "2. **2015年美国成年人样本局限：** 数据不能直接代表越南或亚洲人群。亚洲人群在较低 BMI（$\ge 23\text{ kg/m}^2$）时即面临较高糖尿病风险。\n"
        "3. **相关性而非因果关系：** 模型学习的是统计关联，特征重要性不代表病因机制。\n"
        "4. **中等区分能力：** 在默认阈值下，约 34% 被提醒者的确报告患病（精确率），约 28%–29% 的病例可能被漏检。\n"
        "5. **不同社会经济群体间表现不均：** 详见“健康公平性评估”页面；单一阈值无法让所有群体同时获得相同的召回率与误报率。\n"
        "6. **仅作筛查提示而非临床诊断：** 请将本工具输出视为咨询医生的提示，绝不能替代医院的静脉血糖或 HbA1c 检测。"
    )
elif lang_code == "fr":
    st.markdown(
        "1. **Étiquette et réponses auto-déclarées :** La glycémie n'a pas été mesurée en laboratoire ; le statut repose sur la déclaration des participants.\n"
        "2. **Adultes américains, 2015 :** Les données ne représentent pas directement le Vietnam ou l'Asie, où le risque de diabète apparaît à un IMC plus bas ($\ge 23\text{ kg/m}^2$).\n"
        "3. **Association, pas causalité :** Le modèle capture des corrélations statistiques et non des liens de cause à effet.\n"
        "4. **Discrimination modérée :** Environ 34 % des personnes signalées déclarent la maladie (précision) et environ 28–29 % des cas sont manqués au seuil par défaut.\n"
        "5. **Performance inégale selon les groupes :** Voir la page *Équité en Santé* ; un seuil unique produit des taux d'erreur différents selon le revenu.\n"
        "6. **Enquête, pas clinique :** Utilisez ce résultat uniquement pour engager la discussion avec un médecin."
    )
else:
    st.markdown(
        "1. **Self-reported label and answers.** Nobody's glucose was measured; \"diabetes or prediabetes\" is what respondents reported.\n"
        "2. **US adults, 2015.** The data do not represent Vietnam or other countries, and health patterns have changed since 2015. In Asian populations, risk can appear at lower BMI than in this US sample, so scores may under-flag some people.\n"
        "3. **Association, not cause.** The model learns statistical patterns; importance and \"what pushed your score\" are not causal claims.\n"
        "4. **Moderate discrimination.** About 34% of flagged respondents report the label (precision) and about 28–29% of cases are missed at the default threshold.\n"
        "5. **Uneven performance across groups.** See the Health Equity page; a single threshold does not give every group the same error rates.\n"
        "6. **Survey, not a clinic.** Treat the output as a prompt to talk to a professional, never as a diagnosis."
    )
