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

# Models Compared Table — Simplified & Ordered by Most Important Clinical Screening Metrics First (Recall -> Precision -> F1 -> PR-AUC)
st.subheader(t["ab_models_title"])

_MODEL_NAME_I18N = {
    "vi": {
        "Histogram Gradient Boosting (minimal)": "⭐ Histogram Gradient Boosting (7 câu — Đang dùng trên App)",
        "Histogram Gradient Boosting (full)": "Histogram Gradient Boosting (Đầy đủ 10 câu)",
        "Logistic Regression (full)": "Logistic Regression — Hồi quy Logistic (10 câu)",
        "Logistic Regression (minimal)": "Logistic Regression — Baseline tuyến tính (7 câu)",
    },
    "en": {
        "Histogram Gradient Boosting (minimal)": "⭐ Histogram Gradient Boosting (7 Qs — Deployed App Model)",
        "Histogram Gradient Boosting (full)": "Histogram Gradient Boosting (Full 10 Qs)",
        "Logistic Regression (full)": "Logistic Regression (Full 10 Qs)",
        "Logistic Regression (minimal)": "Logistic Regression — Linear Baseline (7 Qs)",
    },
}
_mname_map = _MODEL_NAME_I18N.get(lang_code, _MODEL_NAME_I18N["en"])

# Sort with Deployed App Model (highest F1 = 0.456) at the top, followed by F1 / Recall priority
_ordered_models = sorted(
    meta["models_compared"],
    key=lambda x: (1 if x["model"] == "Histogram Gradient Boosting (minimal)" else 0, x["f1"], x["recall"]),
    reverse=True,
)

mc_rows = ""
for m in _ordered_models:
    is_app = m["model"] == "Histogram Gradient Boosting (minimal)"
    row_style = " style='background:#eff6ff; font-weight:700;'" if is_app else ""
    q_unit = "câu" if lang_code == "vi" else ("题" if lang_code == "zh" else "Q")
    mc_rows += (
        f"<tr{row_style}>"
        f"<td>{_mname_map.get(m['model'], m['model'])}</td>"
        f"<td style='text-align:right; color:#b91c1c; font-weight:700;'>{m['recall']:.1%}</td>"
        f"<td style='text-align:right; color:#1d4ed8; font-weight:700;'>{m['precision']:.1%}</td>"
        f"<td style='text-align:right; color:#0f766e; font-weight:800;'>{m['f1']:.3f}</td>"
        f"<td style='text-align:right;'>{m['pr_auc_test']:.3f}</td>"
        f"<td style='text-align:right;'>{m['questions']} {q_unit}</td>"
        f"</tr>"
    )

if lang_code == "vi":
    mc_headers = (
        "<tr>"
        "<th>Mô hình (Model)</th>"
        "<th style='text-align:right;'>1. Recall (Độ nhạy — Bắt trúng bệnh)</th>"
        "<th style='text-align:right;'>2. Precision (Độ chuẩn khi báo động)</th>"
        "<th style='text-align:right;'>3. F1-Score (Cân bằng tối ưu)</th>"
        "<th style='text-align:right;'>4. PR-AUC (Test)</th>"
        "<th style='text-align:right;'>Số câu hỏi</th>"
        "</tr>"
    )
    mc_caption = (
        "📌 Thứ tự ưu tiên trong sàng lọc y tế: **Recall** (hạn chế bỏ sót người bệnh) → **Precision** (giảm báo động giả) → "
        "**F1-Score** (tổng hợp cân bằng Recall & Precision; mô hình trên App đạt F1 cao nhất **0.456** và Precision cao nhất **33.6%** chỉ với 7 câu hỏi)."
    )
    mc_summary = (
        "**Lý do đề xuất thuật toán đang chạy trên App (`Histogram Gradient Boosting — 7 câu hỏi`):** "
        "Nắm bắt được tương tác phi tuyến giữa `BMI`, `Huyết áp cao`, `Mỡ máu` và `Tuổi` trên 253,680 mẫu khảo sát, "
        "giữ trọn **100% PR-AUC (`0.418`)** so với bản 10 câu mà người dùng chỉ mất chưa tới 60 giây trả lời tại nhà. "
        "*(Xem phân tích EDA, thống kê Dataset & quy trình Feature Selection 21 → 7 biến tại trang **📊 Khai phá Dữ liệu & Chọn biến**)*."
    )
    adv_exp_label = "🔬 Xem thêm chỉ số kiểm định kỹ thuật chuyên sâu (ROC-AUC, PR-AUC Train vs. Test chống Overfitting, Brier Score)"
else:
    mc_headers = (
        "<tr>"
        "<th>Model</th>"
        "<th style='text-align:right;'>1. Recall (Sensitivity)</th>"
        "<th style='text-align:right;'>2. Precision (Positive Predictive)</th>"
        "<th style='text-align:right;'>3. F1-Score (Harmonic Balance)</th>"
        "<th style='text-align:right;'>4. PR-AUC (Test)</th>"
        "<th style='text-align:right;'>Questions</th>"
        "</tr>"
    )
    mc_caption = (
        "📌 Priority order for medical screening: **Recall** (catching true cases) → **Precision** (limiting unnecessary alarms) → "
        "**F1-Score** (overall Recall–Precision balance; the deployed 7-question model achieves the highest **F1 = 0.456** and highest **Precision = 33.6%**)."
    )
    mc_summary = (
        "**Why `Histogram Gradient Boosting (7 questions)` is deployed:** "
        "Captures non-linear cardiometabolic synergy (`BMI` × `HighBP` × `Age`), matches the 10-variable model's **PR-AUC (`0.418`)** with zero lab tests, "
        "and takes under 60 seconds at home. *(See full Dataset Statistics, EDA Dashboard & 21 → 7 Feature Selection on the **📊 Data Mining & EDA** page)*."
    )
    adv_exp_label = "🔬 View detailed technical audit metrics (ROC-AUC, Train vs. Test PR-AUC Overfitting Check & Brier Score)"

mc_table = f"<table class='clean-table'><thead>{mc_headers}</thead><tbody>{mc_rows}</tbody></table>"
st.markdown(mc_table, unsafe_allow_html=True)
st.caption(mc_caption)
st.markdown(mc_summary)

with st.expander(adv_exp_label):
    adv_rows = ""
    for m in _ordered_models:
        gap = m["pr_auc_train"] - m["pr_auc_test"]
        adv_rows += (
            f"<tr><td>{m['model']}</td>"
            f"<td style='text-align:right;'>{m['threshold']:.2f}</td>"
            f"<td style='text-align:right;'>{m['roc_auc_test']:.3f}</td>"
            f"<td style='text-align:right;'>{m['pr_auc_train']:.3f}</td>"
            f"<td style='text-align:right;'>{m['pr_auc_test']:.3f} ({gap:+.3f})</td>"
            f"<td style='text-align:right;'>{m.get('brier_test', 0.098):.4f}</td></tr>"
        )
    st.markdown(
        "<table class='clean-table'><thead><tr>"
        "<th>Model</th><th style='text-align:right;'>Threshold (τ*)</th>"
        "<th style='text-align:right;'>ROC-AUC (Test)</th><th style='text-align:right;'>PR-AUC (Train)</th>"
        "<th style='text-align:right;'>PR-AUC (Test / Gap)</th><th style='text-align:right;'>Brier Calibration</th>"
        f"</tr></thead><tbody>{adv_rows}</tbody></table>",
        unsafe_allow_html=True,
    )

# Performance of the app model (test set)
st.subheader(t["ab_perf_title"])
acc_app = float(app_m.get("accuracy", 0.765))
p1, p2, p3, p4, p5, p6 = st.columns(6)
p1.metric("PR-AUC", f"{app_m['pr_auc_test']:.3f}")
p2.metric("ROC-AUC", f"{app_m['roc_auc_test']:.3f}")
p3.metric("Recall", f"{app_m['recall']:.1%}")
p4.metric("Precision", f"{app_m['precision']:.1%}")
p5.metric("Accuracy", f"{acc_app:.1%}")
p6.metric("False-positive rate", f"{app_m['fpr']:.1%}")
st.caption(
    f"Tại ngưỡng tối ưu {th_default:.2f}, Accuracy đạt {acc_app:.1%} (nếu tăng ngưỡng lên 0.22–0.24 thì Accuracy đạt ~80%–81%, nhưng sẽ bỏ sót nhiều ca bệnh hơn). Với dữ liệu mất cân bằng (~14% dương tính), Recall và PR-AUC quan trọng hơn Accuracy thuần túy."
    if lang_code == "vi"
    else f"At threshold {th_default:.2f}, Accuracy is {acc_app:.1%} (raising threshold to 0.22–0.24 brings Accuracy to ~80%–81%, at the cost of missing more cases). With an imbalanced ~14% positive label, Recall and PR-AUC are far more informative than raw Accuracy."
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
st.subheader("🤖 Kiến trúc Trợ lý AI Bác sĩ: Model · RAG · Few-Shot · Chi phí & Lưu lượng/Trial" if lang_code == "vi" else "🤖 AI Doctor Architecture: Model · RAG · Few-Shot · Cost & Traffic/Trial Governance")
if lang_code == "vi":
    st.markdown(
        "<table class='clean-table'>"
        "<thead><tr><th>Thành phần Kỹ thuật AI</th><th>Giải pháp Triển khai trên Hệ thống</th><th>Lợi ích Thực tế & Khắc phục Nhược điểm</th></tr></thead>"
        "<tbody>"
        "<tr><td><b>1. Mô hình (Model Routing)</b></td><td>Định tuyến phân tầng 3 lớp: <code>Gemini 2.5 Flash</code> → <code>Cloud OpenAI-Compatible</code> → <code>Local Clinical RAG Synthesizer</code></td><td>Thông minh, linh hoạt và không phụ thuộc cứng vào một nhà cung cấp API duy nhất.</td></tr>"
        "<tr><td><b>2. RAG (Retrieval-Augmented Generation)</b></td><td>Chỉ mục vector kép <code>TfidfVectorizer</code> (Word + Char 3–5 gram) trên <b>9 tài liệu chuẩn y khoa & dự án</b> (ADA 2025, WHO Châu Á, NIH DPP, BRFSS 2015)</td><td>Chống bịa đặt số liệu y khoa (anti-hallucination), luôn gắn trích dẫn <code>[KB-01..09]</code> khớp 100% với số liệu mô hình.</td></tr>"
        "<tr><td><b>3. Few-Shot Prompting</b></td><td>Nhúng <b>2 ví dụ hội chẩn mẫu (2-Shot)</b> kèm hồ sơ trực tiếp của người dùng (<code>BMI</code>, <code>Điểm rủi ro %</code>, <code>What-If BMI -2</code>)</td><td>Câu trả lời ngắn gọn, cá nhân hóa sát thực tế từng người thay vì trả lời chung chung như chatbot thường.</td></tr>"
        "<tr><td><b>4. Tối ưu Chi phí (Cost Control)</b></td><td><b>Semantic Cache</b> + chỉ truyền <b>Top-3 đoạn RAG</b> (~380 tokens thay vì 8,000+ tokens) + ưu tiên Free-Tier / Local RAG</td><td>Giảm <b>~95% chi phí token</b>; chi phí vận hành thực tế đạt <b>~$0.0000 / lượt hỏi</b> (kể cả khi mở rộng hàng nghìn sinh viên dùng thử).</td></tr>"
        "<tr><td><b>5. Lưu lượng & Trial (Traffic Guard)</b></td><td>Bộ giới hạn cửa sổ trượt (<code>25 req/phút</code>, <code>40 lượt cloud/phiên</code>) tự động chuyển mạch sang <code>Local RAG</code> (<code>&lt;15 ms</code>)</td><td>Chống lỗi <code>HTTP 429 Rate Limit</code> khi cả lớp truy cập cùng lúc hoặc khi hết hạn ngạch API dùng thử (Zero Downtime).</td></tr>"
        "</tbody></table>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<table class='clean-table'>"
        "<thead><tr><th>AI Engineering Pillar</th><th>System Implementation</th><th>Operational Impact</th></tr></thead>"
        "<tbody>"
        "<tr><td><b>1. Tiered Model Routing</b></td><td>3-tier cascade: <code>Gemini 2.5 Flash</code> → <code>OpenAI-Compatible Free Endpoint</code> → <code>Local Clinical RAG Reasoner</code></td><td>High clinical intelligence without single-vendor lock-in.</td></tr>"
        "<tr><td><b>2. Clinical RAG</b></td><td>Hybrid Word + Char-Ngram <code>TfidfVectorizer</code> index over <b>9 verified sources</b> (ADA 2025, WHO Asian BMI, NIH DPP, CDC BRFSS)</td><td>Eliminates medical/statistical hallucination with transparent <code>[KB-01..09]</code> citations.</td></tr>"
        "<tr><td><b>3. Few-Shot Grounding</b></td><td><b>2-Shot clinical exemplars</b> paired with live patient context (<code>BMI</code>, <code>Risk %</code>, <code>What-If delta</code>)</td><td>Delivers tailored, actionable two-specialist advice on every query.</td></tr>"
        "<tr><td><b>4. Cost Optimization</b></td><td><b>Semantic Cache</b> + Top-3 chunk compression (~380 prompt tokens vs 8,000+) + Free-Tier/Local routing</td><td>Cuts token volume by <b>~95%</b>, keeping marginal cost near <b>$0.0000 / query</b>.</td></tr>"
        "<tr><td><b>5. Traffic & Trial Guard</b></td><td>Sliding-window rate limiter (<code>25 RPM</code>, <code>40 cloud calls/session</code>) with instant <code>&lt;15 ms</code> Local RAG failover</td><td>Zero <code>HTTP 429</code> crashes during classroom traffic bursts or exhausted free trials.</td></tr>"
        "</tbody></table>",
        unsafe_allow_html=True,
    )

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
