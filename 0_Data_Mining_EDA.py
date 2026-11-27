"""
pages/0_Data_Mining_EDA.py — Data Mining, Dataset Status & Statistics, Interactive EDA Dashboard,
Feature Grouping -> Feature Selection (21 -> 7), and Algorithm Recommendation Rationale.
"""

import altair as alt
import pandas as pd
import streamlit as st
from i18n import FEATURE_LABELS_I18N, inject_friendly_css, load_artifacts, render_sidebar_and_get_lang

st.set_page_config(
    page_title="Data Mining & EDA | Diabetes Screening Assistant",
    page_icon="📊",
    layout="wide",
)

inject_friendly_css()
lang_code, t = render_sidebar_and_get_lang()
_, _, meta = load_artifacts()

eda = meta.get("eda_summary", {})
fname_map = FEATURE_LABELS_I18N.get(lang_code, FEATURE_LABELS_I18N["en"])

EDA_I18N = {
    "vi": {
        "title": "📊 Khai phá Dữ liệu (Data Mining), Thống kê Dataset & Chọn lọc Đặc trưng",
        "subtitle": "Quy trình Data Mining hoàn chỉnh trên bộ dữ liệu CDC BRFSS 2015: Tình trạng dữ liệu → Thống kê mô tả → Dashboard EDA → Nhóm biến & Feature Selection (21 → 7 biến) → Lý do đề xuất thuật toán.",
        "kpi1": "Tổng mẫu khảo sát (BRFSS)",
        "kpi2": "Giá trị khuyết thiếu (Missing)",
        "kpi3": "Tỷ lệ Dương tính (Mất cân bằng)",
        "kpi4": "Chọn lọc Đặc trưng (Feature Selection)",
        "sec1": "1. Tình trạng Dataset & Kiểm định Chất lượng Dữ liệu (Dataset Health)",
        "sec1_bullets": (
            "- **Nguồn gốc & Quy mô:** Bộ dữ liệu **CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015** (`UCI ID #891`) gồm **253,680** người trưởng thành tại Hoa Kỳ với **21 chỉ số sức khỏe gốc**.\n"
            "- **Độ sạch dữ liệu (`Missing = 0%`):** Toàn bộ **253,680 dòng × 11 cột** đều sạch hoàn toàn (`0` ô trống, không có ngoại lai âm hoặc lỗi định dạng). Trong `Pipeline`, nhóm vẫn tích hợp `SimpleImputer` + kẹp biên BMI `[12.0, 70.0]` để chống crash khi người dùng nhập số đo cực trị.\n"
            "- **Tình trạng mất cân bằng lớp nghiêm trọng (`13.93%` vs `86.07%`):** Có **35,346** ca tiểu đường/tiền tiểu đường (`13.93%`) so với **218,334** người khỏe mạnh (`86.07%` — tỷ lệ ~`1 : 6.2`). Vì vậy **Accuracy thuần túy bị sai lệch** (dự đoán ai cũng khỏe mạnh đã đạt `86.1%` Accuracy nhưng `Recall = 0%`!), buộc phải dùng **Recall, Precision, F1, PR-AUC** và **Ngưỡng cắt tối ưu chi phí (`0.18`)**.\n"
            "- **Vì sao giữ nguyên các cấu hình khảo sát trùng nhau (`119,726` mẫu rời rạc lặp lại) và loại bỏ file `5050split`?** Khi khảo sát `253,680` người bằng các câu hỏi nhị phân/thứ bậc, nhiều người khỏe mạnh có cùng tổ hợp trả lời. Giữ nguyên tần suất gốc giúp xác suất dự đoán phản ánh đúng tỷ lệ mắc bệnh thực tế trong cộng đồng (`Brier Score = 0.0979`)."
        ),
        "sec2": "2. Dashboard Khám phá Dữ liệu Trực quan (Interactive EDA Dashboard)",
        "dim_select": "Chọn chiều phân tích tỷ lệ mắc Tiểu đường / Tiền tiểu đường:",
        "dim_opts": {
            "genhlth": "Tự đánh giá sức khỏe tổng quát (GenHlth: 1 → 5)",
            "bmi": "Phân nhóm BMI (chuẩn Châu Á & WHO)",
            "age": "Nhóm tuổi 5 năm (Age: 18–24 → 80+)",
            "comorb": "Hiệu ứng cộng hưởng số bệnh nền (0 → 4 bệnh lý đi kèm)",
        },
        "chart1_title": "Tỷ lệ mắc Tiểu đường / Tiền tiểu đường theo nhóm (Đường đứt nét = TB toàn quần thể 13.9%)",
        "chart2_title": "Độ lớn Tương quan |r| với Tiểu đường (Xanh đậm = 7 biến được chọn vào App)",
        "sec3": "3. Bảng Số liệu Thống kê Mô tả & So sánh 2 Nhóm (Khỏe mạnh vs. Tiểu đường)",
        "sec4": "4. Phân nhóm 21 Biến Gốc & Quy trình Feature Selection (21 → 10 → 7 Biến)",
        "sec5": "5. Đề xuất Thuật toán & Lý do Lựa chọn (Why Histogram Gradient Boosting?)",
    },
    "en": {
        "title": "📊 Data Mining, Dataset Health, Interactive EDA & Feature Selection",
        "subtitle": "End-to-end Data Mining workflow on CDC BRFSS 2015: Dataset Health & Summary Statistics → Interactive EDA Dashboard → Feature Grouping & Selection (21 → 7) → Algorithm Recommendation Rationale.",
        "kpi1": "Total Survey Rows (BRFSS)",
        "kpi2": "Missing Values",
        "kpi3": "Positive Class (Imbalanced)",
        "kpi4": "Feature Selection Ratio",
        "sec1": "1. Dataset Status & Data Quality Audit (Dataset Health)",
        "sec1_bullets": (
            "- **Source & Scale:** **CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015** (`UCI ID #891`) covering **253,680** US adults across **21 raw health indicators**.\n"
            "- **Completeness (`0% Missing`):** All **253,680 rows** contain `0` missing cells. Even so, our production `Pipeline` embeds `SimpleImputer` and BMI boundary clamping (`[12.0, 70.0]`) to guarantee crash-free inference on extreme user inputs.\n"
            "- **Severe Class Imbalance (`13.93%` Positive vs `86.07%` Negative):** **35,346** positive cases vs **218,334** negative cases (~`1 : 6.2` ratio). Raw Accuracy is misleading (`86.1%` by predicting all negative with `0%` Recall), motivating **Recall, Precision, F1-Score, PR-AUC**, and **5:1 cost-sensitive threshold tuning (`0.18`)**.\n"
            "- **Why preserve natural survey profile frequencies & reject `5050split`?** Across 253,680 discrete questionnaire responses, repeated profiles reflect real population weights. Deduplicating discrete rows or using artificial `5050split` destroys probability calibration (`Brier = 0.0979`)."
        ),
        "sec2": "2. Interactive Exploratory Data Analysis (EDA) Dashboard",
        "dim_select": "Select clinical dimension to inspect Diabetes / Prediabetes prevalence:",
        "dim_opts": {
            "genhlth": "Self-Rated General Health (GenHlth: Excellent → Poor)",
            "bmi": "BMI Clinical Categories (Asian & WHO Cutoffs)",
            "age": "5-Year Age Groups (18–24 → 80+)",
            "comorb": "Non-Linear Comorbidity Synergy (0 → 4 Conditions)",
        },
        "chart1_title": "Empirical Diabetes/Prediabetes Prevalence (Dashed line = 13.9% Population Base Rate)",
        "chart2_title": "Absolute Correlation |r| with Target (Blue = 7 Selected App Features)",
        "sec3": "3. Summary Statistics & Cohort Comparison (Healthy vs. Diabetes/Prediabetes)",
        "sec4": "4. Grouping the 21 BRFSS Variables & Feature Selection Pipeline (21 → 10 → 7)",
        "sec5": "5. Proposed Algorithm & Justification (Why Histogram Gradient Boosting?)",
    },
}
cfg = EDA_I18N.get(lang_code, EDA_I18N["en"])

st.title(cfg["title"])
st.caption(cfg["subtitle"])

# Top KPI Cards
k1, k2, k3, k4 = st.columns(4)
k1.metric(cfg["kpi1"], f"{eda.get('n_rows', 253680):,}")
k2.metric(cfg["kpi2"], "0 (0.0%)")
k3.metric(cfg["kpi3"], f"{eda.get('positive_rate', 0.1393):.2%} (1 : 6.2)")
k4.metric(cfg["kpi4"], "21 → 7 (99.8% AUC)")

# Section 1: Dataset Health
st.subheader(cfg["sec1"])
with st.container(border=True):
    st.markdown(cfg["sec1_bullets"])

# Section 2: Interactive EDA Dashboard
st.subheader(cfg["sec2"])
d_left, d_right = st.columns([1.15, 1.0], gap="large")

with d_left:
    with st.container(border=True):
        chosen_dim = st.selectbox(
            cfg["dim_select"],
            options=list(cfg["dim_opts"].keys()),
            format_func=lambda k: cfg["dim_opts"][k],
        )
        dim_key_map = {
            "genhlth": "prevalence_by_genhlth",
            "bmi": "prevalence_by_bmi",
            "age": "prevalence_by_age",
            "comorb": "prevalence_by_comorbidity_count",
        }
        prev_df = pd.DataFrame(eda.get(dim_key_map[chosen_dim], []))
        if not prev_df.empty:
            prev_df["pct"] = prev_df["prevalence"] * 100.0
            base_rule = alt.Chart(pd.DataFrame({"y": [13.93]})).mark_rule(
                color="#dc2626", strokeDash=[6, 4], size=2
            ).encode(y="y:Q")
            bars = (
                alt.Chart(prev_df)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#2563eb")
                .encode(
                    x=alt.X("category:N", sort=alt.EncodingSortField(field="order", order="ascending"), title=""),
                    y=alt.Y("pct:Q", title="Prevalence (%)"),
                    tooltip=[
                        alt.Tooltip("category:N", title="Group"),
                        alt.Tooltip("count:Q", format=",", title="Respondents (n)"),
                        alt.Tooltip("pct:Q", format=".1f", title="Diabetes Rate (%)"),
                    ],
                )
            )
            txt = bars.mark_text(dy=-8, fontSize=11, fontWeight="bold", color="#0f172a").encode(
                text=alt.Text("pct:Q", format=".1f")
            )
            st.markdown(f"**{cfg['chart1_title']}**")
            st.altair_chart((bars + txt + base_rule).properties(height=300), use_container_width=True)

with d_right:
    with st.container(border=True):
        fstats_df = pd.DataFrame(eda.get("feature_stats", []))
        if not fstats_df.empty:
            fstats_df["label"] = fstats_df["feature"].map(lambda k: fname_map.get(k, k))
            fstats_df["status"] = fstats_df["selected_in_7q"].map(
                lambda s: ("Selected (7-Q App)" if lang_code != "vi" else "Được chọn (7 biến App)")
                if s
                else ("Dropped / Audit Only" if lang_code != "vi" else "Loại khỏi Form (Audit/Nhiễu)")
            )
            c_bars = (
                alt.Chart(fstats_df)
                .mark_bar(height=20, cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
                .encode(
                    y=alt.Y("label:N", sort="-x", title=""),
                    x=alt.X("abs_corr:Q", title="|r| Point-Biserial Correlation"),
                    color=alt.Color(
                        "selected_in_7q:N",
                        scale=alt.Scale(domain=[True, False], range=["#1d4ed8", "#94a3b8"]),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("label:N", title="Feature"),
                        alt.Tooltip("corr_with_target:Q", format="+.4f", title="Correlation r"),
                        alt.Tooltip("status:N", title="Selection Status"),
                    ],
                )
            )
            st.markdown(f"**{cfg['chart2_title']}**")
            st.altair_chart(c_bars.properties(height=352), use_container_width=True)

# Section 3: Summary Statistics Table
st.subheader(cfg["sec3"])
if fstats_df is not None and not fstats_df.empty:
    rows_html = ""
    for _, r in fstats_df.iterrows():
        sel_badge = (
            "<span style='background:#dbeafe; color:#1e40af; padding:2px 8px; border-radius:999px; font-weight:700; font-size:0.78rem;'>✓ Chọn (App)</span>"
            if r["selected_in_7q"] and lang_code == "vi"
            else (
                "<span style='background:#dbeafe; color:#1e40af; padding:2px 8px; border-radius:999px; font-weight:700; font-size:0.78rem;'>✓ Selected</span>"
                if r["selected_in_7q"]
                else "<span style='background:#f1f5f9; color:#475569; padding:2px 8px; border-radius:999px; font-size:0.78rem;'>Loại / Equity Audit</span>"
            )
        )
        rows_html += (
            f"<tr>"
            f"<td><b>{fname_map.get(r['feature'], r['feature'])}</b> (<code>{r['feature']}</code>)</td>"
            f"<td>{sel_badge}</td>"
            f"<td style='text-align:right;'>{r['mean_all']:.2f} ± {r['std_all']:.2f}</td>"
            f"<td style='text-align:right;'>{r['min']:.0f} – {r['max']:.0f} (med {r['median']:.0f})</td>"
            f"<td style='text-align:right; color:#0f766e; font-weight:600;'>{r['mean_healthy']:.2f}</td>"
            f"<td style='text-align:right; color:#b91c1c; font-weight:700;'>{r['mean_diabetes']:.2f}</td>"
            f"<td style='text-align:right; font-weight:700;'>{r['corr_with_target']:+.3f}</td>"
            f"</tr>"
        )
    if lang_code == "vi":
        th_html = "<tr><th>Đặc trưng (Feature)</th><th>Trạng thái</th><th style='text-align:right;'>TB ± Độ lệch chuẩn</th><th style='text-align:right;'>Min – Max (Trung vị)</th><th style='text-align:right;'>TB Nhóm Khỏe mạnh</th><th style='text-align:right;'>TB Nhóm Tiểu đường</th><th style='text-align:right;'>Tương quan (r)</th></tr>"
    else:
        th_html = "<tr><th>Feature</th><th>Status</th><th style='text-align:right;'>Mean ± SD</th><th style='text-align:right;'>Min – Max (Median)</th><th style='text-align:right;'>Healthy Mean</th><th style='text-align:right;'>Diabetes Mean</th><th style='text-align:right;'>Target Corr (r)</th></tr>"

    st.markdown(
        f"<table class='clean-table'><thead>{th_html}</thead><tbody>{rows_html}</tbody></table>",
        unsafe_allow_html=True,
    )

# Section 4: Feature Grouping & Selection (21 -> 7)
st.subheader(cfg["sec4"])
if lang_code == "vi":
    grp_table = (
        "<table class='clean-table'>"
        "<thead><tr><th>Nhóm Đặc trưng (21 Biến BRFSS Gốc)</th><th>Các biến thành phần</th><th>Quyết định Feature Selection</th><th>Lý do Y khoa & Khai phá Dữ liệu</th></tr></thead>"
        "<tbody>"
        "<tr><td><b>1. Chuyển hóa & Tim mạch</b> (6 biến)</td><td><code>BMI</code>, <code>HighBP</code>, <code>HighChol</code>, <code>HeartDiseaseorAttack</code>, <code>Stroke</code>, <code>CholCheck</code></td><td><b style='color:#1d4ed8;'>Giữ 4 biến cốt lõi</b> (<code>BMI</code>, <code>HighBP</code>, <code>HighChol</code>, <code>HeartDisease</code>)</td><td>Bộ 3 Hội chứng chuyển hóa + biến chứng mạch vành; đóng góp tín hiệu cao nhất (<code>BMI</code> Permutation = 0.0905, <code>HighBP</code> OR = 2.20×). <code>CholCheck</code> gây nhiễu hành vi đi khám; <code>Stroke</code> quá hiếm (4%) và trùng tín hiệu tim mạch.</td></tr>"
        "<tr><td><b>2. Nhân khẩu & Tổng trạng</b> (4 biến)</td><td><code>Age</code>, <code>GenHlth</code>, <code>DiffWalk</code>, <code>Sex</code></td><td><b style='color:#1d4ed8;'>Giữ 3 biến cốt lõi</b> (<code>Age</code>, <code>GenHlth</code>, <code>DiffWalk</code>)</td><td><code>GenHlth</code> có tương quan đơn biến mạnh nhất toàn bộ dataset (<code>r = +0.294</code>; tỷ lệ bệnh tăng gấp 15 lần từ 2.5% lên 37.9%). <code>Sex</code> có tương quan gần bằng 0 (<code>r = +0.031</code>).</td></tr>"
        "<tr><td><b>3. Lối sống & Đếm ngày tự nhớ</b> (7 biến)</td><td><code>PhysActivity</code>, <code>Smoker</code>, <code>Fruits</code>, <code>Veggies</code>, <code>HvyAlcoholConsump</code>, <code>MentHlth</code>, <code>PhysHlth</code></td><td><b style='color:#64748b;'>Loại khỏi Form 60 giây</b></td><td>Sai số nhớ lại (recall bias) cao khi hỏi số ngày trong 30 ngày qua; thử nghiệm giữ lại <code>PhysActivity</code> ở mô hình 10 biến cho thấy <b>PR-AUC không tăng</b> (<code>0.418</code> vs <code>0.418</code>).</td></tr>"
        "<tr><td><b>4. Kinh tế - Xã hội & Bảo hiểm</b> (4 biến)</td><td><code>Income</code>, <code>Education</code>, <code>AnyHealthcare</code>, <code>NoDocbcCost</code></td><td><b style='color:#0f766e;'>Chỉ dùng Kiểm toán Công bằng (Equity Audit)</b></td><td>Không hỏi thu nhập/bằng cấp trên form cá nhân để tôn trọng quyền riêng tư và tránh thiên kiến xã hội; giữ riêng ở tập Test để đánh giá sai số giữa các nhóm thu nhập.</td></tr>"
        "</tbody></table>"
    )
else:
    grp_table = (
        "<table class='clean-table'>"
        "<thead><tr><th>Domain Feature Group (21 Raw BRFSS Vars)</th><th>Variables</th><th>Feature Selection Decision</th><th>Clinical & Data Mining Justification</th></tr></thead>"
        "<tbody>"
        "<tr><td><b>1. Cardiometabolic History</b> (6 vars)</td><td><code>BMI</code>, <code>HighBP</code>, <code>HighChol</code>, <code>HeartDiseaseorAttack</code>, <code>Stroke</code>, <code>CholCheck</code></td><td><b style='color:#1d4ed8;'>Select 4 Core</b> (<code>BMI</code>, <code>HighBP</code>, <code>HighChol</code>, <code>HeartDisease</code>)</td><td>Core metabolic syndrome triad; top permutation signal (<code>BMI = 0.0905</code>, <code>HighBP OR = 2.20×</code>). Dropped <code>CholCheck</code> (healthcare-access confounder) and <code>Stroke</code> (collinear with CHD).</td></tr>"
        "<tr><td><b>2. Demographics & Functional Status</b> (4 vars)</td><td><code>Age</code>, <code>GenHlth</code>, <code>DiffWalk</code>, <code>Sex</code></td><td><b style='color:#1d4ed8;'>Select 3 Core</b> (<code>Age</code>, <code>GenHlth</code>, <code>DiffWalk</code>)</td><td><code>GenHlth</code> has the strongest correlation in the entire dataset (<code>r = +0.294</code>; prevalence jumps 15× from 2.5% to 37.9%). <code>Sex</code> adds negligible signal (<code>r = +0.031</code>).</td></tr>"
        "<tr><td><b>3. Lifestyle & 30-Day Recall Counts</b> (7 vars)</td><td><code>PhysActivity</code>, <code>Smoker</code>, <code>Fruits</code>, <code>Veggies</code>, <code>HvyAlcoholConsump</code>, <code>MentHlth</code>, <code>PhysHlth</code></td><td><b style='color:#64748b;'>Drop from 1-Min Form</b></td><td>High self-report recall noise; empirical benchmark (7 vs 10 vars) proves <b>zero loss in PR-AUC</b> (<code>0.418</code> vs <code>0.418</code>) and 99.8% ROC-AUC retention.</td></tr>"
        "<tr><td><b>4. Socioeconomic & Care Access</b> (4 vars)</td><td><code>Income</code>, <code>Education</code>, <code>AnyHealthcare</code>, <code>NoDocbcCost</code></td><td><b style='color:#0f766e;'>Held-Out Equity Audit Only</b></td><td>Excluded from personal intake for privacy & non-discrimination; preserved on the 50,736-row Test set to audit subgroup Recall & FPR fairness.</td></tr>"
        "</tbody></table>"
    )
st.markdown(grp_table, unsafe_allow_html=True)

# Section 5: Algorithm Proposal & Justification
st.subheader(cfg["sec5"])
if lang_code == "vi":
    algo_html = (
        "<table class='clean-table'>"
        "<thead><tr><th>Thuật toán xem xét</th><th>Cơ chế toán học</th><th>Ưu / Nhược điểm trên dữ liệu Y tế Bảng (253,680 dòng)</th><th>Quyết định</th></tr></thead>"
        "<tbody>"
        "<tr style='background:#eff6ff;'><td><b>⭐ Histogram Gradient Boosting</b><br>(<code>HistGradientBoostingClassifier</code>)</td><td>Ensemble cây quyết định tăng cường trên 255 thùng histogram (tương đương LightGBM)</td><td><b>Bắt trọn tương tác phi tuyến</b> (VD: khi có đồng thời 4 bệnh nền, tỷ lệ tiểu đường vọt lên <b>49.5%</b>), tốc độ huấn luyện <code>O(N)</code> cực nhanh trên 253k dòng, xác suất hiệu chỉnh tốt (<code>Brier = 0.0979</code>), đạt <b>F1 cao nhất (0.456)</b> và <b>PR-AUC cao nhất (0.418)</b>.</td><td><b style='color:#1d4ed8;'>Đề xuất Triển khai Chính</b></td></tr>"
        "<tr><td><b>Logistic Regression</b><br>(<code>LogisticRegression</code>)</td><td>Tổ hợp tuyến tính log-odds: <code>σ(wᵀx + b)</code></td><td>Giải thích trực tiếp được bằng <b>Odds Ratio</b> (như <code>HighBP = 2.20×</code>), nhưng giả định cộng tính tuyến tính nên bỏ lỡ hiệu ứng cộng hưởng giữa BMI + Huyết áp + Tuổi (<code>PR-AUC = 0.392</code>, thấp hơn HGB).</td><td><b>Giữ làm Baseline đối chiếu & Giải thích OR</b></td></tr>"
        "<tr><td><b>KNN / SVM / Clustering</b></td><td>Khoảng cách Euclid / Kernel bậc hai</td><td>Độ phức tạp <code>O(N²)</code> không khả thi cho 253,680 mẫu khi chạy trực tuyến; nhạy cảm với biến nhị phân rời rạc và không xuất xác suất hiệu chỉnh trực tiếp cho bài toán ngưỡng chi phí 5:1.</td><td><span style='color:#64748b;'>Loại bỏ</span></td></tr>"
        "</tbody></table>"
    )
else:
    algo_html = (
        "<table class='clean-table'>"
        "<thead><tr><th>Candidate Algorithm</th><th>Mathematical Mechanism</th><th>Suitability on 253,680 Imbalanced Tabular Survey Rows</th><th>Decision</th></tr></thead>"
        "<tbody>"
        "<tr style='background:#eff6ff;'><td><b>⭐ Histogram Gradient Boosting</b><br>(<code>HistGradientBoostingClassifier</code>)</td><td>Gradient-boosted decision trees over 255-bin histograms (LightGBM-inspired)</td><td><b>Captures non-linear clinical synergy</b> (4 comorbidities → <b>49.5%</b> diabetes rate), trains in <code>O(N)</code> time, delivers well-calibrated probabilities (<code>Brier = 0.0979</code>), and achieves <b>highest F1 (0.456)</b> & <b>PR-AUC (0.418)</b>.</td><td><b style='color:#1d4ed8;'>Selected Production Model</b></td></tr>"
        "<tr><td><b>Logistic Regression</b><br>(<code>LogisticRegression</code>)</td><td>Linear log-odds hyperplane: <code>σ(wᵀx + b)</code></td><td>Provides transparent <b>Odds Ratios</b> (<code>HighBP = 2.20×</code>), but additive linearity misses multi-factor metabolic interactions (<code>PR-AUC = 0.392</code> vs <code>0.418</code>).</td><td><b>Retained as Interpretable Baseline</b></td></tr>"
        "<tr><td><b>KNN / RBF-SVM / Deep MLP</b></td><td>Pairwise Euclidean distance / dense over-parameterized layers</td><td><code>O(N²)</code> memory/latency bottleneck on 253k rows or poorer calibration on low-dimensional tabular survey features compared to gradient-boosted trees.</td><td><span style='color:#64748b;'>Rejected</span></td></tr>"
        "</tbody></table>"
    )
st.markdown(algo_html, unsafe_allow_html=True)
