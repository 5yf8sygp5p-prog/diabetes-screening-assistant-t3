"""
i18n.py — 4-Language Translation Dictionary (English, Tiếng Việt, 中文, Français) & Shared Helpers
==================================================================================================
"""

import io
import json
import wave
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

LANGUAGES = {
    "🇬🇧 English": "en",
    "🇻🇳 Tiếng Việt": "vi",
    "🇨🇳 中文 (简体)": "zh",
    "🇫🇷 Français": "fr",
}

TRANSLATIONS = {
    "en": {
        "lang_label": "🌐 Language / Ngôn ngữ",
        "disclaimer_box": "Educational project — not for clinical use.",
        "disclaimer_long": "Educational project — not for clinical use. Results describe patterns in a 2015 US survey and are not medical advice.",
        "nav_app": "🩺 Screening Assistant",
        "nav_threshold": "⚖️ Threshold and Cost",
        "nav_equity": "🤝 Health Equity",
        "nav_about": "📖 About Model",
        # Main App Page
        "app_title": "Diabetes Screening Assistant",
        "who_for_bold": "Who is this for?",
        "who_for_text": "An adult answering a few health questions at home who wants to decide **whether to ask a health professional about a blood glucose test.**",
        "app_caption": "7 questions · takes about a minute · nothing you enter is stored. The tool estimates how similar your answers are to people who *reported* diabetes or prediabetes in a US health survey. It does not diagnose anything.",
        "q_age": "Age group (years)",
        "q_height": "Height (cm)",
        "q_weight": "Weight (kg)",
        "q_genhlth": "In general, how would you rate your health?",
        "q_highbp": "Has a health professional ever told you that you have **high blood pressure**?",
        "q_highchol": "Has a health professional ever told you that you have **high cholesterol**?",
        "q_heart": "Have you ever been told you have **coronary heart disease** or had a **heart attack**?",
        "q_diffwalk": "Do you have **serious difficulty walking or climbing stairs**?",
        "no": "No",
        "yes": "Yes",
        "genhlth_1": "Excellent",
        "genhlth_2": "Very good",
        "genhlth_3": "Good",
        "genhlth_4": "Fair",
        "genhlth_5": "Poor",
        "btn_analyze": "Analyze",
        "prompt_analyze": "Fill in the form and press **Analyze**.",
        # Results section
        "res_recommend_flag": "🩺 Recommended action: Ask a health professional about a blood glucose test",
        "res_recommend_safe": "✅ Low immediate risk: Routine healthy lifestyle — no urgent test flagged today",
        "res_score_label": "Estimated similarity score (probability)",
        "res_threshold_label": "Active decision threshold",
        "res_bmi_computed": "Calculated BMI from height & weight",
        "res_natural_freq": "In the 2015 US survey, about **{n} out of 100** adults with similar answers reported diabetes or prediabetes.",
        "res_top_factors_title": "What pushed your score up or down?",
        "res_top_factors_sub": "Top factors for your specific answers compared with the average survey respondent:",
        "pushes_up": "⬆️ Pushes risk UP",
        "pushes_down": "⬇️ Pushes risk DOWN",
        "neutral": "➖ Near average",
        "col_question": "Question / Factor",
        "col_your_answer": "Your Answer",
        "col_impact": "Effect on Your Score",
        "col_plain_meaning": "Plain-English Meaning",
        "what_if_title": "💡 Interactive What-If Check: What if BMI drops by 2 points?",
        "what_if_desc": "If your BMI decreased from **{bmi1}** to **{bmi2}** kg/m², your estimated score would change from **{p1}** to **{p2}** ({delta}).",
        # Threshold and Cost Page
        "tc_title": "Threshold and Cost Simulator",
        "tc_banner": "Costs below are **assumed numbers for simulation only**, not measured medical costs. They express how much worse a missed case is judged to be than an unnecessary test.",
        "tc_slider_label": "Cost of a missed case (FN) relative to an unnecessary test (FP = 1)",
        "tc_manual_check": "Pick the threshold manually",
        "tc_manual_slider": "Manual decision threshold",
        "tc_opt_metric": "Cost-minimizing threshold",
        "tc_default_note": "Default used by the app: {th} (cost ratio 5:1).",
        "tc_btn_use": "Use on Screening page",
        "tc_btn_reset": "Reset to default",
        "tc_applied_msg": "Active screening threshold updated to **{th}**!",
        "tc_kpi_recall": "Recall (cases caught)",
        "tc_kpi_precision": "Precision",
        "tc_kpi_fpr": "False-positive rate",
        "tc_kpi_flagged": "Flagged overall",
        "tc_per1000_title": "Per 1,000 adults screened (held-out test data, survey prevalence)",
        "tc_row_pos": "Reported diabetes/prediabetes",
        "tc_row_neg": "Did not report it",
        "tc_col_flag": "Model flags",
        "tc_col_noflag": "Model does not flag",
        "tc_per1000_caption": "Missed cases (false negatives): **{fn}** · Unnecessary flags (false positives): **{fp}**",
        "tc_chart1_title": "Expected cost vs. threshold",
        "tc_chart1_y": "Assumed cost per person (validation)",
        "tc_chart2_title": "Recall, precision and share flagged (test)",
        "tc_how_to_read": "**How to read this:** with a high cost ratio the best threshold moves down, the model flags more people and recall rises, while precision falls. Choosing a threshold is a value judgement about missed cases versus unnecessary tests and testing capacity, not something the data can decide alone.",
        # Health Equity Page
        "he_title": "Health equity check",
        "he_intro": "Does the screening tool behave similarly for people with different incomes and education levels? We evaluate on held-out test data, split by group, using the **same threshold for everyone**. Error bars are 95% confidence intervals: small groups are less certain.",
        "he_model_radio": "Model",
        "he_model_app": "App model (7 questions, no income/education inputs)",
        "he_model_full": "Full model (includes income, education & physical activity)",
        "he_group_radio": "Group by",
        "he_group_income": "Income",
        "he_group_edu": "Education",
        "he_chart_recall": "Recall: share of cases the model flags",
        "he_chart_fpr": "False-positive rate: non-cases flagged",
        "he_dashed_note": "Dashed line = overall rate.",
        "he_numbers_title": "What the numbers say",
        "he_numbers_b1": "**Cases caught (recall)** ranges from **{min_rec}** (High ($75k+)) to **{max_rec}** (Low (< $25k)); the confidence intervals do not overlap.",
        "he_numbers_b2": "**False alarms (false-positive rate)** range from **{min_fpr}** (High ($75k+)) to **{max_fpr}** (Low (< $25k)).",
        "he_numbers_b3": "The share of people in each group who reported diabetes/prediabetes also differs (**{min_prev}** to **{max_prev}**). With one shared threshold, groups with a higher underlying rate tend to be flagged more often, which raises both recall and false alarms for that group.",
        "he_exp1_title": "How to interpret this (and what it does not show)",
        "he_exp1_body": "Because lower-income and lower-education groups in the US survey have a higher prevalence of High Blood Pressure, High BMI, and Fair/Poor self-rated health, a model using one shared threshold flags a larger share of respondents in those groups. That catches more true cases (higher Recall) at the cost of more false alarms (higher FPR). Conversely, higher-income respondents have lower average risk scores, so more borderline cases fall below the cutoff.",
        "he_exp2_title": "Explore a different threshold (point estimates, no confidence intervals)",
        # About Model Page
        "ab_title": "About the model",
        "ab_decision_title": "Decision this tool supports",
        "ab_decision_text": "*An adult answering health-related questions at home uses the application to assess their likelihood of reporting diabetes or prediabetes and decide whether to seek a blood glucose test.*",
        "ab_data_title": "Data",
        "ab_models_title": "Models compared",
        "ab_perf_title": "Performance of the app model (test set)",
        "ab_thresh_title": "Threshold and cost assumptions",
        "ab_thresh_text": "Default threshold **{th}** minimizes an *assumed* cost where a missed case (false negative) counts **5×** an unnecessary test (false positive = 1). These weights are assumptions for decision simulation, not measured medical costs. Change them on the Threshold & Cost page.",
        "ab_matters_title": "Which answers matter most?",
        "ab_perm_title": "App model: permutation importance (Drop in PR-AUC)",
        "ab_or_title": "Logistic regression odds ratios (simple, readable baseline; per 1 unit, other answers fixed)",
        "ab_limits_title": "Limitations",
    },
    "vi": {
        "lang_label": "🌐 Ngôn ngữ / Language",
        "disclaimer_box": "Dự án học thuật — không dùng cho chẩn đoán lâm sàng. (Educational project — not for clinical use.)",
        "disclaimer_long": "Dự án học thuật — không dùng cho chẩn đoán lâm sàng. Kết quả chỉ mô tả quy luật thống kê từ khảo sát y tế Mỹ năm 2015 và không phải lời khuyên y khoa.",
        "nav_app": "🩺 Trợ lý Tầm soát",
        "nav_threshold": "⚖️ Ngưỡng cắt & Chi phí",
        "nav_equity": "🤝 Công bằng Y tế",
        "nav_about": "📖 Về Mô hình AI",
        # Main App Page
        "app_title": "Trợ lý Kiểm tra Nguy cơ Tiểu đường Tại nhà",
        "who_for_bold": "Ai nên dùng công cụ này?",
        "who_for_text": "Bất kỳ người trưởng thành nào ở nhà muốn biết **mình có nên đi bác sĩ thử máu kiểm tra đường huyết hay không.**",
        "app_caption": "Chỉ cần trả lời 7 câu hỏi đơn giản · Mất chưa tới 1 phút · Không lưu lại thông tin cá nhân của bạn.",
        "q_age": "Độ tuổi của bạn",
        "q_height": "Chiều cao (cm)",
        "q_weight": "Cân nặng (kg)",
        "q_genhlth": "Bạn tự thấy sức khỏe của mình dạo này thế nào?",
        "q_highbp": "Bác sĩ đã từng nói bạn bị **cao huyết áp** chưa?",
        "q_highchol": "Bác sĩ đã từng nói bạn bị **mỡ máu cao (cholesterol cao)** chưa?",
        "q_heart": "Bạn đã từng bị **bệnh tim** hoặc **đau tim** chưa?",
        "q_diffwalk": "Bạn có thấy **khó khăn nhiều khi đi bộ hoặc leo cầu thang** không?",
        "no": "Chưa / Không",
        "yes": "Có",
        "genhlth_1": "Rất khỏe mạnh",
        "genhlth_2": "Khỏe",
        "genhlth_3": "Bình thường",
        "genhlth_4": "Hơi yếu",
        "genhlth_5": "Kém / Hay mệt",
        "btn_analyze": "Xem Kết Quả Ngay",
        "prompt_analyze": "Bạn hãy chọn 7 câu trả lời ở trên rồi bấm nút **Xem Kết Quả Ngay** nhé.",
        # Results
        "res_recommend_flag": "🩺 Lời khuyên: Bạn nên đi khám và làm xét nghiệm máu kiểm tra đường huyết",
        "res_recommend_safe": "✅ Kết quả tốt: Nguy cơ hiện tại thấp — cứ giữ thói quen sống khỏe nhé!",
        "res_score_label": "Mức độ nguy cơ ước tính",
        "res_threshold_label": "Mốc cần đi kiểm tra",
        "res_bmi_computed": "Chỉ số cân nặng theo chiều cao (BMI)",
        "res_natural_freq": "Cứ **100 người** có câu trả lời giống bạn trong khảo sát, có khoảng **{n} người** bị tiểu đường hoặc tiền tiểu đường.",
        "res_top_factors_title": "Vì sao bạn có kết quả này?",
        "res_top_factors_sub": "Những câu trả lời dưới đây ảnh hưởng nhiều nhất đến kết quả của bạn:",
        "pushes_up": "⬆️ Làm tăng nguy cơ",
        "pushes_down": "⬇️ Giúp giảm nguy cơ (Tốt)",
        "neutral": "➖ Bình thường",
        "col_question": "Thông tin bạn chọn",
        "col_your_answer": "Câu trả lời của bạn",
        "col_impact": "Ảnh hưởng tới kết quả",
        "col_plain_meaning": "Ý nghĩa dễ hiểu",
        "what_if_title": "💡 Gợi ý nhỏ: Nếu bạn giảm nhẹ cân nặng (giảm 2 điểm BMI) thì sao?",
        "what_if_desc": "Nếu chỉ số BMI giảm từ **{bmi1}** xuống **{bmi2}**, mức nguy cơ của bạn sẽ giảm từ **{p1}** xuống còn **{p2}** ({delta}).",
        # Threshold and Cost Page
        "tc_title": "Mô phỏng Ngưỡng quyết định & Chi phí Sai số",
        "tc_banner": "Các mức chi phí dưới đây là **con số giả định phục vụ mô phỏng quyết định**, không phải hóa đơn viện phí cố định. Chúng thể hiện việc bỏ sót một ca bệnh nghiêm trọng gấp bao nhiêu lần so với một lần đi xét nghiệm kiểm tra.",
        "tc_slider_label": "Chi phí của một ca bỏ sót (FN) so với một lần xét nghiệm chưa cần thiết (FP = 1)",
        "tc_manual_check": "Tự chọn ngưỡng cắt thủ công",
        "tc_manual_slider": "Ngưỡng quyết định thủ công",
        "tc_opt_metric": "Ngưỡng tối thiểu hóa chi phí",
        "tc_default_note": "Ngưỡng mặc định của ứng dụng: {th} (tỷ lệ chi phí 5:1).",
        "tc_btn_use": "Áp dụng cho trang Tầm soát",
        "tc_btn_reset": "Khôi phục mặc định",
        "tc_applied_msg": "Đã cập nhật ngưỡng tầm soát sang **{th}**!",
        "tc_kpi_recall": "Độ nhạy — Recall (Tỷ lệ phát hiện bệnh)",
        "tc_kpi_precision": "Độ chuẩn xác — Precision",
        "tc_kpi_fpr": "Tỷ lệ báo động giả (FPR)",
        "tc_kpi_flagged": "Tổng tỷ lệ được khuyến nghị đi khám",
        "tc_per1000_title": "Trên mỗi 1.000 người lớn được tầm soát (dữ liệu tập Test độc lập)",
        "tc_row_pos": "Thực tế có tiểu đường / tiền tiểu đường",
        "tc_row_neg": "Thực tế không báo cáo mắc bệnh",
        "tc_col_flag": "Mô hình khuyến nghị đi xét nghiệm",
        "tc_col_noflag": "Mô hình báo nguy cơ thấp",
        "tc_per1000_caption": "Số ca bỏ sót (False Negatives): **{fn}** · Số ca báo động giả (False Positives): **{fp}**",
        "tc_chart1_title": "Chi phí kỳ vọng theo từng ngưỡng cắt (Validation)",
        "tc_chart1_y": "Chi phí giả định trên mỗi người",
        "tc_chart2_title": "Biến thiên Recall, Precision và Tỷ lệ cảnh báo (Test)",
        "tc_how_to_read": "**Cách đọc biểu đồ:** Khi tỷ lệ chi phí bỏ sót bệnh cao, ngưỡng cắt tối ưu sẽ hạ thấp xuống để bắt được nhiều ca bệnh hơn (Recall tăng), chấp nhận nhiều người khỏe mạnh đi xét nghiệm kiểm tra hơn (Precision giảm). Việc chọn ngưỡng là bài toán đánh đổi giá trị y tế chứ không chỉ thuần túy là con số thống kê.",
        # Health Equity Page
        "he_title": "Kiểm định Công bằng Y tế (Health Equity)",
        "he_intro": "Công cụ tầm soát hoạt động ra sao giữa các nhóm Thu nhập và Trình độ học vấn khác nhau? Chúng tôi đánh giá trên tập Test độc lập với **cùng một ngưỡng cắt chung cho tất cả mọi người** (kèm khoảng tin cậy 95%).",
        "he_model_radio": "Mô hình",
        "he_model_app": "Mô hình trên App (7 câu hỏi, không hỏi thu nhập/học vấn)",
        "he_model_full": "Mô hình đầy đủ (có bao gồm thu nhập, học vấn & vận động)",
        "he_group_radio": "Phân nhóm theo",
        "he_group_income": "Thu nhập (Income)",
        "he_group_edu": "Học vấn (Education)",
        "he_chart_recall": "Recall: Tỷ lệ ca bệnh thực tế được phát hiện",
        "he_chart_fpr": "False-positive rate: Tỷ lệ người không bệnh bị gắn cờ",
        "he_dashed_note": "Đường nét đứt = tỷ lệ trung bình toàn cục.",
        "he_numbers_title": "Những con số nói lên điều gì?",
        "he_numbers_b1": "**Tỷ lệ phát hiện bệnh (Recall)** dao động từ **{min_rec}** (Thu nhập cao $75k+) đến **{max_rec}** (Thu nhập thấp < $25k); các khoảng tin cậy không chồng lấn.",
        "he_numbers_b2": "**Tỷ lệ báo động giả (FPR)** dao động từ **{min_fpr}** (Thu nhập cao $75k+) đến **{max_fpr}** (Thu nhập thấp < $25k).",
        "he_numbers_b3": "Tỷ lệ mắc bệnh nền trong từng nhóm cũng rất khác biệt (**{min_prev}** đến **{max_prev}**). Khi dùng chung một ngưỡng cắt, nhóm có tỷ lệ bệnh nền cao hơn sẽ được khuyến nghị đi khám nhiều hơn, làm tăng cả Recall lẫn FPR.",
        "he_exp1_title": "Cách hiểu đúng kết quả này (và những điều nó không thể hiện)",
        "he_exp1_body": "Do nhóm thu nhập thấp và học vấn thấp có tỷ lệ huyết áp cao, BMI cao và sức khỏe tổng quát kém cao hơn, mô hình sử dụng một ngưỡng cắt chung sẽ gắn cờ cảnh báo nhiều hơn ở nhóm này — giúp giảm tối đa số ca bệnh bị bỏ sót ở nhóm yếu thế, nhưng đổi lại tỷ lệ báo động giả cao hơn.",
        "he_exp2_title": "Khám phá kết quả công bằng khi thay đổi ngưỡng cắt (ước lượng điểm)",
        # About Model Page
        "ab_title": "Thông tin chi tiết về Mô hình & Dữ liệu",
        "ab_decision_title": "Quyết định mà công cụ này hỗ trợ",
        "ab_decision_text": "*Một người trưởng thành trả lời các câu hỏi sức khỏe tại nhà sử dụng ứng dụng để đánh giá khả năng mắc tiểu đường/tiền tiểu đường và quyết định có nên đi xét nghiệm đường huyết hay không.*",
        "ab_data_title": "Dữ liệu huấn luyện (Data)",
        "ab_models_title": "Bảng so sánh các mô hình (Models compared)",
        "ab_perf_title": "Hiệu năng của mô hình trên App (Tập kiểm thử độc lập — Test set)",
        "ab_thresh_title": "Giả định về Ngưỡng quyết định và Chi phí",
        "ab_thresh_text": "Ngưỡng mặc định **{th}** tối thiểu hóa hàm chi phí giả định trong đó một ca bỏ sót bệnh (False Negative) nghiêm trọng gấp **5 lần** một xét nghiệm chưa cần thiết (False Positive = 1). Bạn có thể tùy chỉnh tỷ lệ này ở trang Ngưỡng cắt & Chi phí.",
        "ab_matters_title": "Câu hỏi nào quan trọng nhất đối với mô hình?",
        "ab_perm_title": "Mô hình chính (Gradient Boosting): Độ quan trọng hoán vị (Mức sụt giảm PR-AUC khi xáo trộn biến)",
        "ab_or_title": "Tỷ số chênh (Odds Ratios) của Hồi quy Logistic (mô hình cơ sở dễ đọc; tính trên mỗi 1 đơn vị tăng thêm)",
        "ab_limits_title": "Các hạn chế của mô hình (Limitations)",
    },
    "zh": {
        "lang_label": "🌐 语言 / Language",
        "disclaimer_box": "教学科研项目 — 不用于临床诊断。(Educational project — not for clinical use.)",
        "disclaimer_long": "教学科研项目 — 不用于临床诊断。结果仅反映2015年美国健康调查的统计规律，不构成医疗建议。",
        "nav_app": "🩺 糖尿病筛查助手",
        "nav_threshold": "⚖️ 阈值与成本模拟",
        "nav_equity": "🤝 健康公平性评估",
        "nav_about": "📖 关于模型与数据",
        "app_title": "居家糖尿病风险筛查助手",
        "who_for_bold": "适用人群：",
        "who_for_text": "在家回答几个简单健康问题的成年人，用于辅助决定**是否需要向医生咨询进行血糖检测。**",
        "app_caption": "仅需7个问题 · 约1分钟完成 · 不存储您的任何输入数据。本工具评估您的回答与美国CDC健康调查中*报告*患有糖尿病或糖尿病前期人群的相似程度，不提供任何医学诊断。",
        "q_age": "年龄组（岁）",
        "q_height": "身高（厘米 cm）",
        "q_weight": "体重（公斤 kg）",
        "q_genhlth": "总体而言，您如何评价自己的健康状况？",
        "q_highbp": "医生或医疗人员是否曾告知您患有**高血压**？",
        "q_highchol": "医生或医疗人员是否曾告知您患有**高胆固醇（高血脂）**？",
        "q_heart": "您是否曾被告知患有**冠心病**或曾发生**心肌梗死**？",
        "q_diffwalk": "您在**走路或爬楼梯时是否有严重困难**？",
        "no": "否",
        "yes": "是",
        "genhlth_1": "极好 (Excellent)",
        "genhlth_2": "很好 (Very good)",
        "genhlth_3": "良好 (Good)",
        "genhlth_4": "一般 (Fair)",
        "genhlth_5": "较差 (Poor)",
        "btn_analyze": "立即分析",
        "prompt_analyze": "请填写上方表格并点击 **立即分析**。",
        "res_recommend_flag": "🩺 建议行动：建议向医生咨询并进行空腹血糖或糖化血红蛋白（HbA1c）检测",
        "res_recommend_safe": "✅ 当前风险较低：请继续保持健康的生活方式与常规年度体检",
        "res_score_label": "模型估计风险概率得分",
        "res_threshold_label": "当前决策阈值",
        "res_bmi_computed": "根据身高与体重计算的 BMI 指数",
        "res_natural_freq": "在2015年美国调查中，每 **100位** 回答与您相似的成年人中，约有 **{n}位** 报告患有糖尿病或糖尿病前期。",
        "res_top_factors_title": "哪些因素推高或降低了您的风险得分？",
        "res_top_factors_sub": "与普通受访者相比，您的各项回答对风险分数的具体影响：",
        "pushes_up": "⬆️ 推高风险",
        "pushes_down": "⬇️ 降低风险（保护因素）",
        "neutral": "➖ 接近平均水平",
        "col_question": "健康问题 / 指标",
        "col_your_answer": "您的回答",
        "col_impact": "对得分的影响",
        "col_plain_meaning": "通俗解释",
        "what_if_title": "💡 情景模拟：如果 BMI 降低 2 个点会怎样？",
        "what_if_desc": "如果您的 BMI 从 **{bmi1}** 降至 **{bmi2}** kg/m²，您的估计风险得分将从 **{p1}** 降至 **{p2}** ({delta})。",
        "tc_title": "决策阈值与错误成本模拟器",
        "tc_banner": "以下成本为**仅用于决策模拟的假设权重**，并非真实医疗账单。它表达了“漏诊一名患者”相比“做一次不必要血液检查”的相对严重程度。",
        "tc_slider_label": "漏诊成本 (FN) 相对于一次不必要检查成本 (FP = 1) 的倍数",
        "tc_manual_check": "手动选择决策阈值",
        "tc_manual_slider": "手动阈值",
        "tc_opt_metric": "成本最小化最优阈值",
        "tc_default_note": "应用默认阈值：{th}（成本比 5:1）。",
        "tc_btn_use": "应用到筛查页面",
        "tc_btn_reset": "恢复默认值",
        "tc_applied_msg": "已将筛查阈值更新为 **{th}**！",
        "tc_kpi_recall": "召回率 / 敏感度 (Recall)",
        "tc_kpi_precision": "精确率 (Precision)",
        "tc_kpi_fpr": "假阳性率 / 误报率 (FPR)",
        "tc_kpi_flagged": "总建议检测比例",
        "tc_per1000_title": "每筛查 1,000 名成年人的预期结果（基于独立测试集）",
        "tc_row_pos": "实际报告患有糖尿病/前期",
        "tc_row_neg": "实际未报告患病",
        "tc_col_flag": "模型建议去检测",
        "tc_col_noflag": "模型判定低风险",
        "tc_per1000_caption": "漏诊人数（假阴性 FN）：**{fn}** · 不必要提醒人数（假阳性 FP）：**{fp}**",
        "tc_chart1_title": "不同阈值下的每位受访者预期成本（验证集）",
        "tc_chart1_y": "人均假设成本 (Validation)",
        "tc_chart2_title": "召回率、精确率与建议检测比例随阈值变化曲线（测试集）",
        "tc_how_to_read": "**如何解读：** 当漏诊成本权重较高时，最优阈值下移，模型会提醒更多人进行检查，从而提高召回率（捕获更多患者），但精确率会相应下降。",
        "he_title": "健康公平性检查 (Health Equity)",
        "he_intro": "筛查工具对不同收入和教育水平的人群表现是否一致？我们在独立测试集上对所有人使用**相同的统一阈值**进行了分组评估（误差棒代表 95% 置信区间）。",
        "he_model_radio": "选择模型",
        "he_model_app": "应用精简模型（仅7个问题，不含收入/教育输入）",
        "he_model_full": "完整模型（包含收入、教育及运动指标）",
        "he_group_radio": "分组维度",
        "he_group_income": "收入水平 (Income)",
        "he_group_edu": "教育程度 (Education)",
        "he_chart_recall": "召回率 (Recall)：实际患病者被成功提醒的比例",
        "he_chart_fpr": "假阳性率 (FPR)：健康人群被误报的比例",
        "he_dashed_note": "虚线 = 总体平均比率。",
        "he_numbers_title": "数据结果说明了什么",
        "he_numbers_b1": "**病例检出率（召回率 Recall）** 在 **{min_rec}**（高收入 $75k+）至 **{max_rec}**（低收入 < $25k）之间变化，置信区间不重叠。",
        "he_numbers_b2": "**误报率（假阳性率 FPR）** 在 **{min_fpr}**（高收入）至 **{max_fpr}**（低收入）之间变化。",
        "he_numbers_b3": "不同群体的基础患病率存在显著差异（**{min_prev}** 至 **{max_prev}**）。在统一阈值下，基础患病率较高的群体更容易被标记，从而同时提高了该群体的检出率和误报率。",
        "he_exp1_title": "如何解读这一结果（以及它未反映的内容）",
        "he_exp1_body": "由于低收入和低学历群体中高血压、高BMI和自评健康不佳的比例更高，统一阈值会在这些群体中发出更多提醒，从而捕获更多真实患者（高召回率），但也伴随更高的假阳性率。",
        "he_exp2_title": "探索不同阈值下的分组点估计表现",
        "ab_title": "关于模型与数据 (About the Model)",
        "ab_decision_title": "本工具支持的决策",
        "ab_decision_text": "*成年人在家中回答健康问题，使用本应用评估自身报告糖尿病或糖尿病前期的可能性，并决定是否寻求血糖检测。*",
        "ab_data_title": "数据集与清洗流程 (Data)",
        "ab_models_title": "模型对比 (Models compared)",
        "ab_perf_title": "应用主模型在独立测试集上的表现",
        "ab_thresh_title": "阈值与成本假设",
        "ab_thresh_text": "默认阈值 **{th}** 最小化了假设成本（其中漏诊1名患者的成本设为不必要检测的 **5倍**）。您可以在“阈值与成本模拟”页面调整该比例。",
        "ab_matters_title": "哪些问题的权重最大？",
        "ab_perm_title": "应用主模型：排列重要性（打乱该特征后 PR-AUC 的下降幅度）",
        "ab_or_title": "逻辑回归优势比 Odds Ratios（简单直观的基线模型；其他变量固定时每增加1个单位的倍数）",
        "ab_limits_title": "模型局限性 (Limitations)",
    },
    "fr": {
        "lang_label": "🌐 Langue / Language",
        "disclaimer_box": "Projet éducatif — ne pas utiliser à des fins cliniques. (Educational project — not for clinical use.)",
        "disclaimer_long": "Projet éducatif — ne pas utiliser à des fins cliniques. Les résultats décrivent les tendances d'une enquête américaine de 2015 et ne constituent pas un avis médical.",
        "nav_app": "🩺 Assistant de Dépistage",
        "nav_threshold": "⚖️ Seuil et Coûts",
        "nav_equity": "🤝 Équité en Santé",
        "nav_about": "📖 À propos du Modèle",
        "app_title": "Assistant de Dépistage du Diabète",
        "who_for_bold": "À qui s'adresse cet outil ?",
        "who_for_text": "Un adulte répondant à quelques questions de santé à domicile pour décider **s'il doit demander un test de glycémie à un professionnel de santé.**",
        "app_caption": "7 questions · prend environ une minute · aucune donnée saisie n'est enregistrée. L'outil estime à quel point vos réponses ressemblent à celles des personnes ayant *déclaré* un diabète ou un prédiabète dans l'enquête CDC BRFSS. Il ne pose aucun diagnostic médical.",
        "q_age": "Tranche d'âge (années)",
        "q_height": "Taille (cm)",
        "q_weight": "Poids (kg)",
        "q_genhlth": "En général, comment évalueriez-vous votre santé ?",
        "q_highbp": "Un professionnel de santé vous a-t-il déjà dit que vous aviez une **hypertension artérielle** ?",
        "q_highchol": "Un professionnel de santé vous a-t-il déjà dit que vous aviez un **taux de cholestérol élevé** ?",
        "q_heart": "Vous a-t-on déjà diagnostiqué une **maladie coronarienne** ou une **crise cardiaque** ?",
        "q_diffwalk": "Avez-vous de **graves difficultés à marcher ou à monter des escaliers** ?",
        "no": "Non",
        "yes": "Oui",
        "genhlth_1": "Excellente",
        "genhlth_2": "Très bonne",
        "genhlth_3": "Bonne",
        "genhlth_4": "Passable",
        "genhlth_5": "Mauvaise",
        "btn_analyze": "Analyser",
        "prompt_analyze": "Remplissez le formulaire et appuyez sur **Analyser**.",
        "res_recommend_flag": "🩺 Action recommandée : Consultez un professionnel de santé pour un test de glycémie",
        "res_recommend_safe": "✅ Risque immédiat faible : Maintenez un mode de vie sain — aucun test urgent signalé",
        "res_score_label": "Score de probabilité estimé",
        "res_threshold_label": "Seuil de décision actif",
        "res_bmi_computed": "IMC (BMI) calculé à partir de la taille et du poids",
        "res_natural_freq": "Dans l'enquête américaine de 2015, environ **{n} adultes sur 100** ayant des réponses similaires ont déclaré un diabète ou un prédiabète.",
        "res_top_factors_title": "Qu'est-ce qui a fait monter ou baisser votre score ?",
        "res_top_factors_sub": "Principaux facteurs influençant votre score par rapport à la moyenne des répondants :",
        "pushes_up": "⬆️ Augmente le risque",
        "pushes_down": "⬇️ Réduit le risque (Protecteur)",
        "neutral": "➖ Proche de la moyenne",
        "col_question": "Question / Facteur",
        "col_your_answer": "Votre réponse",
        "col_impact": "Effet sur votre score",
        "col_plain_meaning": "Explication en langage clair",
        "what_if_title": "💡 Simulation What-If : Et si votre IMC baissait de 2 points ?",
        "what_if_desc": "Si votre IMC passait de **{bmi1}** à **{bmi2}** kg/m², votre score passerait de **{p1}** à **{p2}** ({delta}).",
        "tc_title": "Simulateur de Seuil et de Coûts",
        "tc_banner": "Les coûts ci-dessous sont des **valeurs hypothétiques pour la simulation uniquement**, et non des frais médicaux réels. Ils expriment à quel point un cas manqué est jugé plus grave qu'un test inutile.",
        "tc_slider_label": "Coût d'un cas manqué (FN) par rapport à un test inutile (FP = 1)",
        "tc_manual_check": "Choisir le seuil manuellement",
        "tc_manual_slider": "Seuil manuel",
        "tc_opt_metric": "Seuil minimisant le coût",
        "tc_default_note": "Seuil par défaut de l'application : {th} (ratio de coût 5:1).",
        "tc_btn_use": "Utiliser sur la page de dépistage",
        "tc_btn_reset": "Réinitialiser",
        "tc_applied_msg": "Seuil de dépistage mis à jour à **{th}** !",
        "tc_kpi_recall": "Sensibilité / Rappel (cas détectés)",
        "tc_kpi_precision": "Précision",
        "tc_kpi_fpr": "Taux de faux positifs (FPR)",
        "tc_kpi_flagged": "Part globale signalée",
        "tc_per1000_title": "Pour 1 000 adultes dépistés (données de test indépendantes)",
        "tc_row_pos": "Diabète / prédiabète déclaré",
        "tc_row_neg": "Non déclaré",
        "tc_col_flag": "Signalé par le modèle",
        "tc_col_noflag": "Non signalé",
        "tc_per1000_caption": "Cas manqués (faux négatifs) : **{fn}** · Alertes inutiles (faux positifs) : **{fp}**",
        "tc_chart1_title": "Coût espéré en fonction du seuil",
        "tc_chart1_y": "Coût supposé par personne (validation)",
        "tc_chart2_title": "Rappel, précision et part signalée (test)",
        "tc_how_to_read": "**Comment lire ces graphiques :** avec un ratio de coût élevé, le meilleur seuil diminue, le modèle signale plus de personnes et le rappel augmente, tandis que la précision diminue.",
        "he_title": "Vérification de l'équité en santé",
        "he_intro": "L'outil de dépistage se comporte-t-il de manière similaire selon le revenu et le niveau d'éducation ? Évaluation sur le jeu de test avec le **même seuil pour tous** (intervalles de confiance à 95 %).",
        "he_model_radio": "Modèle",
        "he_model_app": "Modèle de l'application (7 questions, sans revenu/éducation)",
        "he_model_full": "Modèle complet (inclut revenu, éducation et activité physique)",
        "he_group_radio": "Regrouper par",
        "he_group_income": "Revenu (Income)",
        "he_group_edu": "Éducation (Education)",
        "he_chart_recall": "Rappel : part des vrais cas signalés par le modèle",
        "he_chart_fpr": "Taux de faux positifs : non-cas signalés",
        "he_dashed_note": "Ligne pointillée = taux global.",
        "he_numbers_title": "Ce que disent les chiffres",
        "he_numbers_b1": "**Les cas détectés (rappel)** varient de **{min_rec}** (Revenu élevé $75k+) à **{max_rec}** (Revenu faible < $25k) ; les intervalles de confiance ne se chevauchent pas.",
        "he_numbers_b2": "**Les fausses alertes (FPR)** varient de **{min_fpr}** (Revenu élevé) à **{max_fpr}** (Revenu faible).",
        "he_numbers_b3": "La prévalence sous-jacente diffère fortement selon le groupe (**{min_prev}** à **{max_prev}**). Avec un seuil partagé unique, les groupes ayant une prévalence plus élevée sont signalés plus souvent.",
        "he_exp1_title": "Comment interpréter cela (et ce que cela ne montre pas)",
        "he_exp1_body": "Les groupes à faible revenu ayant une prévalence plus élevée d'hypertension et d'IMC élevé dans l'enquête, un seuil unique signale davantage de personnes dans ces groupes, augmentant à la fois la détection des vrais cas et les fausses alertes.",
        "he_exp2_title": "Explorer un autre seuil (estimations ponctuelles)",
        "ab_title": "À propos du modèle",
        "ab_decision_title": "Décision soutenue par cet outil",
        "ab_decision_text": "*Un adulte répondant à des questions de santé à domicile utilise l'application pour évaluer sa probabilité de déclarer un diabète ou un prédiabète et décider s'il doit demander un test sanguin.*",
        "ab_data_title": "Données (Data)",
        "ab_models_title": "Modèles comparés",
        "ab_perf_title": "Performance du modèle de l'application (jeu de test)",
        "ab_thresh_title": "Hypothèses de seuil et de coût",
        "ab_thresh_text": "Le seuil par défaut **{th}** minimise un coût supposé où un cas manqué (faux négatif) compte **5×** plus qu'un test inutile (faux positif = 1).",
        "ab_matters_title": "Quelles réponses comptent le plus ?",
        "ab_perm_title": "Modèle principal : importance par permutation (Baisse du PR-AUC)",
        "ab_or_title": "Rapports de cotes (Odds Ratios) de la régression logistique (par unité supplémentaire)",
        "ab_limits_title": "Limites (Limitations)",
    },
}

AGE_OPTIONS = {
    1: "18–24",
    2: "25–29",
    3: "30–34",
    4: "35–39",
    5: "40–44",
    6: "45–49",
    7: "50–54",
    8: "55–59",
    9: "60–64",
    10: "65–69",
    11: "70–74",
    12: "75–79",
    13: "80+",
}

FEATURE_LABELS_I18N = {
    "en": {
        "BMI": "Body mass index (BMI)",
        "GenHlth": "Self-rated general health",
        "Age": "Age group",
        "HighBP": "High blood pressure",
        "HighChol": "High cholesterol",
        "HeartDiseaseorAttack": "Heart disease or heart attack",
        "DiffWalk": "Difficulty walking / climbing stairs",
    },
    "vi": {
        "BMI": "Chỉ số khối cơ thể (BMI)",
        "GenHlth": "Đánh giá sức khỏe tổng quát",
        "Age": "Nhóm tuổi",
        "HighBP": "Huyết áp cao",
        "HighChol": "Mỡ máu / Cholesterol cao",
        "HeartDiseaseorAttack": "Bệnh tim mạch vành / Nhồi máu cơ tim",
        "DiffWalk": "Khó khăn khi đi bộ / leo cầu thang",
    },
    "zh": {
        "BMI": "身体质量指数 (BMI)",
        "GenHlth": "自评总体健康状况",
        "Age": "年龄组",
        "HighBP": "高血压",
        "HighChol": "高胆固醇",
        "HeartDiseaseorAttack": "冠心病或心肌梗死史",
        "DiffWalk": "走路或爬楼梯严重困难",
    },
    "fr": {
        "BMI": "Indice de masse corporelle (IMC / BMI)",
        "GenHlth": "Santé générale auto-évaluée",
        "Age": "Tranche d'âge",
        "HighBP": "Hypertension artérielle",
        "HighChol": "Cholestérol élevé",
        "HeartDiseaseorAttack": "Maladie cardiaque ou crise cardiaque",
        "DiffWalk": "Difficulté à marcher / monter des escaliers",
    },
}


EIU_LOGO_B64 = "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"
DOC_MALE_B64 = "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"
DOC_FEMALE_B64 = "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"


def inject_friendly_css():
    """Injects high-contrast, dark-mode-proof CSS with Be Vietnam Pro & soothing medical colors."""
    st.markdown(
        "<style>"
        "@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');"
        ":root { color-scheme: light !important; }"
        "html, body, .stApp, [data-testid='stAppViewContainer'], [data-testid='stHeader'], main {"
        "background-color: #f4f7fb !important;"
        "color: #1e293b !important;"
        "accent-color: #1d4ed8 !important;"
        "}"
        "html, body, .stApp, h1, h2, h3, h4, h5, p, label, table, th, td, button, input, textarea {"
        "font-family: 'Be Vietnam Pro', -apple-system, BlinkMacSystemFont, sans-serif !important;"
        "}"
        "span[data-testid='stIconMaterial'], .material-symbols-rounded {"
        "font-family: 'Material Symbols Rounded' !important;"
        "}"
        "section[data-testid='stSidebar'], section[data-testid='stSidebar'] > div {"
        "background-color: #f8fafc !important;"
        "border-right: 1.5px solid #e2e8f0 !important;"
        "}"
        "section[data-testid='stSidebar'] p, section[data-testid='stSidebar'] span, section[data-testid='stSidebar'] label, section[data-testid='stSidebar'] div {"
        "color: #1e293b !important;"
        "}"
        "/* Transform Sidebar Radio into sleek modern SaaS Navigation Pill Tabs (hide ugly radio circles in sidebar) */"
        "section[data-testid='stSidebar'] div[role='radiogroup'] {"
        "gap: 4px !important;"
        "width: 100% !important;"
        "}"
        "section[data-testid='stSidebar'] div[role='radiogroup'] > label {"
        "background-color: transparent !important;"
        "border: 1px solid transparent !important;"
        "border-radius: 10px !important;"
        "padding: 9px 14px !important;"
        "margin-bottom: 2px !important;"
        "width: 100% !important;"
        "cursor: pointer !important;"
        "transition: all 0.15s ease !important;"
        "}"
        "section[data-testid='stSidebar'] div[role='radiogroup'] > label:hover {"
        "background-color: #e2e8f0 !important;"
        "}"
        "section[data-testid='stSidebar'] div[role='radiogroup'] > label > div:first-child {"
        "display: none !important;"
        "}"
        "section[data-testid='stSidebar'] div[role='radiogroup'] > label:has(input:checked) {"
        "background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%) !important;"
        "border-color: #1d4ed8 !important;"
        "box-shadow: 0 4px 10px rgba(37, 99, 235, 0.2) !important;"
        "}"
        "section[data-testid='stSidebar'] div[role='radiogroup'] > label:has(input:checked) p, "
        "section[data-testid='stSidebar'] div[role='radiogroup'] > label:has(input:checked) span {"
        "color: #ffffff !important;"
        "font-weight: 700 !important;"
        "}"
        "/* Force readable dark text & light background on all containers and widgets even when browser is in Dark Mode */"
        "[data-testid='stVerticalBlockBorderWrapper'] {"
        "background-color: #ffffff !important;"
        "border: 1.5px solid #cbd5e1 !important;"
        "border-radius: 16px !important;"
        "padding: 20px !important;"
        "box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04) !important;"
        "}"
        "label, div[data-testid='stWidgetLabel'] p, div[data-testid='stMarkdownContainer'] p, "
        "div[data-testid='stMarkdownContainer'] li, div[data-testid='stCaptionContainer'] p {"
        "color: #1e293b !important;"
        "font-size: 0.96rem !important;"
        "line-height: 1.55 !important;"
        "}"
        "h1, h2, h3, h4, h5 {"
        "color: #0f172a !important;"
        "}"
        "div[data-testid='stWidgetLabel'] p {"
        "font-weight: 600 !important;"
        "color: #0f172a !important;"
        "}"
        "div[data-baseweb='select'] > div, div[data-baseweb='input'], div[data-baseweb='base-input'], input, textarea {"
        "background-color: #ffffff !important;"
        "color: #0f172a !important;"
        "border: 1.5px solid #94a3b8 !important;"
        "border-radius: 8px !important;"
        "font-weight: 500 !important;"
        "}"
        "div[data-baseweb='popover'] *, ul[role='listbox'], li[role='option'] {"
        "background-color: #ffffff !important;"
        "color: #0f172a !important;"
        "}"
        "div[data-testid='stNumberInput'] button {"
        "background-color: #f1f5f9 !important;"
        "color: #0f172a !important;"
        "border-color: #cbd5e1 !important;"
        "}"
        "div[role='radiogroup'] label span, div[role='radiogroup'] p {"
        "color: #0f172a !important;"
        "font-weight: 600 !important;"
        "}"
        "div[data-testid='stMetric'] {"
        "background-color: #ffffff !important;"
        "border: 1.5px solid #cbd5e1 !important;"
        "border-radius: 12px !important;"
        "padding: 14px 18px !important;"
        "}"
        "div[data-testid='stMetricValue'], div[data-testid='stMetricLabel'] {"
        "color: #0f172a !important;"
        "}"
        "div[data-testid='stForm'] {"
        "background-color: #ffffff !important;"
        "border: 1.5px solid #cbd5e1 !important;"
        "border-radius: 16px !important;"
        "padding: 24px 28px !important;"
        "box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05) !important;"
        "}"
        "/* Secondary buttons (Quick-ask buttons) — light background with crisp dark text */"
        ".stButton > button {"
        "background-color: #f1f5f9 !important;"
        "color: #0f172a !important;"
        "border: 1.5px solid #94a3b8 !important;"
        "border-radius: 10px !important;"
        "font-weight: 600 !important;"
        "padding: 0.5rem 1rem !important;"
        "}"
        ".stButton > button * {"
        "color: #0f172a !important;"
        "}"
        ".stButton > button:hover {"
        "background-color: #e2e8f0 !important;"
        "border-color: #0284c7 !important;"
        "}"
        "/* Primary buttons (Analyze & Send) — rich clinical blue with white text */"
        "div[data-testid='stFormSubmitButton'] > button, .stButton > button[kind='primary'] {"
        "background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;"
        "color: #ffffff !important;"
        "font-weight: 700 !important;"
        "font-size: 1.0rem !important;"
        "border-radius: 10px !important;"
        "padding: 0.6rem 1.6rem !important;"
        "border: none !important;"
        "box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25) !important;"
        "}"
        "div[data-testid='stFormSubmitButton'] > button *, .stButton > button[kind='primary'] * {"
        "color: #ffffff !important;"
        "}"
        ".edu-warning-box {"
        "background-color: #f0f9ff !important;"
        "border: 1.5px solid #7dd3fc !important;"
        "border-left: 5px solid #0284c7 !important;"
        "color: #0c4a6e !important;"
        "padding: 13px 15px !important;"
        "border-radius: 10px !important;"
        "font-size: 0.88rem !important;"
        "font-weight: 600 !important;"
        "line-height: 1.45 !important;"
        "margin-top: 12px !important;"
        "margin-bottom: 12px !important;"
        "}"
        ".clean-card {"
        "background: #ffffff !important;"
        "border: 1.5px solid #cbd5e1 !important;"
        "border-radius: 14px !important;"
        "padding: 18px 22px !important;"
        "box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04) !important;"
        "margin-bottom: 16px !important;"
        "color: #0f172a !important;"
        "}"
        ".clean-table {"
        "width: 100% !important;"
        "border-collapse: collapse !important;"
        "background: #ffffff !important;"
        "border: 1.5px solid #cbd5e1 !important;"
        "border-radius: 12px !important;"
        "overflow: hidden !important;"
        "margin-top: 10px !important;"
        "margin-bottom: 10px !important;"
        "}"
        ".clean-table th {"
        "background: #f1f5f9 !important;"
        "color: #0f172a !important;"
        "font-weight: 700 !important;"
        "font-size: 0.9rem !important;"
        "padding: 12px 16px !important;"
        "text-align: left !important;"
        "border-bottom: 1.5px solid #cbd5e1 !important;"
        "}"
        ".clean-table td {"
        "padding: 12px 16px !important;"
        "border-bottom: 1px solid #e2e8f0 !important;"
        "color: #1e293b !important;"
        "font-size: 0.92rem !important;"
        "font-weight: 500 !important;"
        "}"
        ".clean-table tr:nth-child(even) td {"
        "background: #f8fafc !important;"
        "}"
        "</style>",
        unsafe_allow_html=True,
    )


LOGO_TEXT_I18N = {
    "en": (
        "EASTERN INTERNATIONAL UNIVERSITY",
        "Becamex Business School",
        "🏛️ Eastern International University (EIU) • Healthcare Decision-Support System",
        "🌐 Display Language",
        "📌 Menu Navigation",
    ),
    "vi": (
        "ĐẠI HỌC QUỐC TẾ MIỀN ĐÔNG (EIU)",
        "Trường Kinh doanh Becamex",
        "🏛️ Đại học Quốc tế Miền Đông (EIU) • Hệ thống Hỗ trợ Quyết định Y tế",
        "🌐 Ngôn ngữ hiển thị",
        "📌 Danh mục Trang",
    ),
    "zh": (
        "东方国际大学 (EIU)",
        "Becamex 商学院",
        "🏛️ 东方国际大学 (EIU) • 医疗决策支持系统",
        "🌐 显示语言 (Language)",
        "📌 页面导航",
    ),
    "fr": (
        "UNIVERSITÉ INTERNATIONALE DE L'EST",
        "École de Commerce Becamex",
        "🏛️ Université Internationale de l'Est (EIU) • Système d'Aide à la Décision",
        "🌐 Langue d'affichage",
        "📌 Navigation",
    ),
}

NAV_TABS_I18N = {
    "en": {
        "app": "🩺 Screening Assistant",
        "Threshold and Cost": "⚖️ Threshold & Cost",
        "Health Equity": "🤝 Health Equity",
        "About Model": "📖 About Model",
        "Batch Screening": "📂 Batch Screening",
    },
    "vi": {
        "app": "🩺 Trợ lý Sàng lọc",
        "Threshold and Cost": "⚖️ Ngưỡng cắt & Chi phí",
        "Health Equity": "🤝 Công bằng Y tế",
        "About Model": "📖 Về Mô hình AI",
        "Batch Screening": "📂 Tầm soát Hàng loạt",
    },
    "zh": {
        "app": "🩺 糖尿病筛查助手",
        "Threshold and Cost": "⚖️ 阈值与成本模拟",
        "Health Equity": "🤝 健康公平性评估",
        "About Model": "📖 关于模型与数据",
        "Batch Screening": "📂 批量筛查 (CSV)",
    },
    "fr": {
        "app": "🩺 Assistant de Dépistage",
        "Threshold and Cost": "⚖️ Seuil et Coûts",
        "Health Equity": "🤝 Équité en Santé",
        "About Model": "📖 À propos du Modèle",
        "Batch Screening": "📂 Dépistage par Lot",
    },
}


THEME_I18N = {
    "vi": {
        "label": "🎨 Chủ đề giao diện",
        "christmas": "🎄 Giáng sinh",
        "tet": "🧧 Tết Nguyên Đán",
        "medical": "🩺 Y tế Chuẩn",
        "xmas_side": "🎅 Giáng sinh An lành • Sức khỏe Trọn vẹn ❄️",
        "xmas_banner": "✨ Merry Christmas • Chúc bạn mùa lễ hội thật ấm áp & khỏe mạnh! 🎁",
        "xmas_music": "🎵 Nhạc Giáng sinh (Không lời)",
        "tet_side": "🧧 Chúc Mừng Năm Mới • Vạn Sự An Khang 🌸",
        "tet_banner": "🏮 Chúc Mừng Năm Mới • Sức Khỏe Dồi Dào, An Khang Thịnh Vượng! 🌸",
        "tet_music": "🎵 Nhạc Tết Xuân Về (Không lời)",
    },
    "en": {
        "label": "🎨 Theme",
        "christmas": "🎄 Christmas",
        "tet": "🧧 Lunar New Year (Tết)",
        "medical": "🩺 Standard Medical",
        "xmas_side": "🎅 Merry Christmas • Stay Healthy ❄️",
        "xmas_banner": "✨ Merry Christmas • Wishing you a warm & healthy holiday season! 🎁",
        "xmas_music": "🎵 Christmas Instrumental",
        "tet_side": "🧧 Happy Lunar New Year • Good Health 🌸",
        "tet_banner": "🏮 Happy Lunar New Year • Wishing you health, joy & prosperity! 🌸",
        "tet_music": "🎵 Lunar New Year Instrumental",
    },
    "zh": {
        "label": "🎨 界面主题",
        "christmas": "🎄 圣诞节",
        "tet": "🧧 农历新年 (春节)",
        "medical": "🩺 标准医疗",
        "xmas_side": "🎅 圣诞快乐 • 健康平安 ❄️",
        "xmas_banner": "✨ Merry Christmas • 祝您圣诞温暖、健康相伴！🎁",
        "xmas_music": "🎵 圣诞纯音乐",
        "tet_side": "🧧 新春快乐 • 万事如意 🌸",
        "tet_banner": "🏮 新春快乐 • 祝您身体健康、阖家幸福！🌸",
        "tet_music": "🎵 新春纯音乐",
    },
    "fr": {
        "label": "🎨 Thème",
        "christmas": "🎄 Noël",
        "tet": "🧧 Nouvel An Lunaire (Tết)",
        "medical": "🩺 Médical Standard",
        "xmas_side": "🎅 Joyeux Noël • Santé & Douceur ❄️",
        "xmas_banner": "✨ Joyeux Noël • Passez de merveilleuses fêtes en pleine santé ! 🎁",
        "xmas_music": "🎵 Musique de Noël (Instrumentale)",
        "tet_side": "🧧 Bonne Année Lunaire • Santé 🌸",
        "tet_banner": "🏮 Bonne Année Lunaire • Santé, bonheur et prospérité ! 🌸",
        "tet_music": "🎵 Musique du Tết (Instrumentale)",
    },
}


@st.cache_data(show_spinner=False)
def build_holiday_music_wav(theme_key: str) -> bytes:
    """
    Synthesizes a warm, pleasant, instrumental-only music-box / đàn tranh WAV melody
    for Christmas ('Jingle Bells') and Lunar New Year / Tết ('Ngày Tết Quê Em' pentatonic).
    """
    sr = 22050
    note_hz = {
        "REST": 0.0,
        "G2": 98.00, "A2": 110.00, "C3": 130.81, "D3": 146.83, "E3": 164.81, "F3": 174.61, "G3": 196.00, "A3": 220.00,
        "C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23, "G4": 392.00, "A4": 440.00,
        "C5": 523.25, "D5": 587.33, "E5": 659.25, "F5": 698.46, "G5": 783.99, "A5": 880.00, "C6": 1046.50,
    }

    if theme_key == "christmas":
        # Jingle Bells Chorus — Warm Celesta / Music Box
        beat = 0.27
        melody = [
            ("E5", 1, "C3"), ("E5", 1, "C3"), ("E5", 2, "C3"),
            ("E5", 1, "C3"), ("E5", 1, "C3"), ("E5", 2, "C3"),
            ("E5", 1, "C3"), ("G5", 1, "C3"), ("C5", 1.5, "C3"), ("D5", 0.5, "C3"), ("E5", 4, "C3"),
            ("F5", 1, "F3"), ("F5", 1, "F3"), ("F5", 1.5, "F3"), ("F5", 0.5, "F3"),
            ("F5", 1, "C3"), ("E5", 1, "C3"), ("E5", 1, "C3"), ("E5", 1, "C3"),
            ("E5", 1, "G3"), ("D5", 1, "G3"), ("D5", 1, "G3"), ("E5", 1, "G3"), ("D5", 2, "G3"), ("G5", 2, "G3"),
            ("E5", 1, "C3"), ("E5", 1, "C3"), ("E5", 2, "C3"),
            ("E5", 1, "C3"), ("E5", 1, "C3"), ("E5", 2, "C3"),
            ("E5", 1, "C3"), ("G5", 1, "C3"), ("C5", 1.5, "C3"), ("D5", 0.5, "C3"), ("E5", 4, "C3"),
            ("F5", 1, "F3"), ("F5", 1, "F3"), ("F5", 1.5, "F3"), ("F5", 0.5, "F3"),
            ("F5", 1, "C3"), ("E5", 1, "C3"), ("E5", 1, "C3"), ("E5", 1, "C3"),
            ("G5", 1, "G3"), ("G5", 1, "G3"), ("F5", 1, "G3"), ("D5", 1, "G3"), ("C5", 4, "C3"),
        ]
        decay_rate = 3.2
    else:
        # Joyful Vietnamese Pentatonic Spring Melody ("Ngày Tết Quê Em" style) — Đàn Tranh / Music Box
        beat = 0.28
        melody = [
            # "Tết, Tết, Tết, Tết đến rồi!"
            ("G5", 0.5, "C3"), ("G5", 0.5, "C3"), ("G5", 0.5, "C3"), ("A5", 0.5, "C3"), ("G5", 0.5, "C3"), ("E5", 1.5, "C3"),
            ("G5", 0.5, "G3"), ("G5", 0.5, "G3"), ("G5", 0.5, "G3"), ("A5", 0.5, "G3"), ("G5", 0.5, "G3"), ("E5", 1.5, "G3"),
            # "Tết, Tết, Tết, Tết đến rồi, Tết đến trong tim mọi người"
            ("G5", 0.5, "A2"), ("G5", 0.5, "A2"), ("G5", 0.5, "A2"), ("A5", 0.5, "A2"), ("C6", 1.0, "F3"),
            ("A5", 0.5, "F3"), ("G5", 0.5, "G3"), ("E5", 0.5, "G3"), ("D5", 0.5, "G3"), ("C5", 2.0, "C3"),
            # "Mừng ngày Tết trên khắp quê tôi"
            ("C5", 0.5, "C3"), ("D5", 0.5, "C3"), ("E5", 0.5, "C3"), ("G5", 0.5, "C3"), ("E5", 0.5, "C3"), ("D5", 0.5, "C3"), ("C5", 1.0, "C3"),
            # "Ngàn hoa thơm khoe sắc xinh tươi"
            ("D5", 0.5, "G3"), ("E5", 0.5, "G3"), ("G5", 0.5, "G3"), ("A5", 0.5, "G3"), ("G5", 0.5, "G3"), ("E5", 0.5, "G3"), ("D5", 1.0, "G3"),
            # "Đàn em thơ khoe áo mới, chạy tung tăng vui pháo hoa"
            ("E5", 0.5, "A2"), ("G5", 0.5, "A2"), ("A5", 0.5, "F3"), ("C6", 1.0, "F3"),
            ("A5", 0.5, "G3"), ("G5", 0.5, "G3"), ("E5", 0.5, "G3"), ("D5", 0.5, "G3"), ("C5", 2.5, "C3"),
        ]
        decay_rate = 4.2

    chunks = []
    for n_name, dur_beats, bass_name in melody:
        n_samples = int(sr * dur_beats * beat)
        t_arr = np.linspace(0.0, dur_beats * beat, n_samples, endpoint=False)
        freq = note_hz.get(n_name, 0.0)
        bass_f = note_hz.get(bass_name, 0.0)

        attack = 1.0 - np.exp(-t_arr / 0.012)
        env = attack * np.exp(-decay_rate * t_arr)
        bass_env = attack * np.exp(-1.8 * t_arr)

        sig = np.zeros(n_samples, dtype=np.float64)
        if freq > 0:
            sig += (
                0.55 * np.sin(2.0 * np.pi * freq * t_arr)
                + 0.22 * np.sin(2.0 * np.pi * (2.0 * freq) * t_arr)
                + 0.08 * np.sin(2.0 * np.pi * (3.0 * freq) * t_arr)
            ) * env
        if bass_f > 0:
            sig += (
                0.20 * np.sin(2.0 * np.pi * bass_f * t_arr)
                + 0.10 * np.sin(2.0 * np.pi * (1.5 * bass_f) * t_arr)
            ) * bass_env

        # Gentle fade at note end to prevent clicks
        fade_len = min(120, n_samples // 4)
        if fade_len > 0:
            sig[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)
        chunks.append(sig)

    full = np.concatenate(chunks)
    peak = np.max(np.abs(full)) + 1e-9
    pcm = np.int16((full / peak) * 18000)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def inject_tet_theme():
    """Injects a festive Vietnamese Lunar New Year (Tết) theme with falling hoa mai & hoa đào petals."""
    petals_html = (
        "<div class='tet-petal-container' aria-hidden='true'>"
        "<div class='tet-petal p1'>🌼</div>"
        "<div class='tet-petal p2'>🌸</div>"
        "<div class='tet-petal p3'>🧧</div>"
        "<div class='tet-petal p4'>🌼</div>"
        "<div class='tet-petal p5'>✨</div>"
        "<div class='tet-petal p6'>🌸</div>"
        "<div class='tet-petal p7'>🌼</div>"
        "<div class='tet-petal p8'>🌸</div>"
        "<div class='tet-petal p9'>🏮</div>"
        "<div class='tet-petal p10'>🌼</div>"
        "<div class='tet-petal p11'>✨</div>"
        "<div class='tet-petal p12'>🌸</div>"
        "</div>"
        "<div class='tet-lantern-bar' aria-hidden='true'>"
        "<span class='t-lamp'>🏮</span><span class='t-lamp'>🌼</span><span class='t-lamp'>🧧</span><span class='t-lamp'>🌸</span>"
        "<span class='t-lamp'>🏮</span><span class='t-lamp'>🌼</span><span class='t-lamp'>🧧</span><span class='t-lamp'>🌸</span>"
        "<span class='t-lamp'>🏮</span><span class='t-lamp'>🌼</span><span class='t-lamp'>🧧</span><span class='t-lamp'>🌸</span>"
        "</div>"
    )
    tet_css = (
        "<style>"
        "html, body, .stApp, [data-testid='stAppViewContainer'], [data-testid='stHeader'], main {"
        "background: linear-gradient(180deg, #fffbeb 0%, #fff1f2 52%, #fef9c3 100%) !important;"
        "background-attachment: fixed !important;"
        "color: #1e293b !important;"
        "accent-color: #dc2626 !important;"
        "}"
        "section[data-testid='stSidebar'], section[data-testid='stSidebar'] > div {"
        "background: linear-gradient(180deg, #fef2f2 0%, #fffbeb 55%, #fff7ed 100%) !important;"
        "border-right: 2px solid #fcd34d !important;"
        "}"
        "section[data-testid='stSidebar'] div[role='radiogroup'] > label {"
        "background-color: rgba(255,255,255,0.88) !important;"
        "border: 1px solid #fde68a !important;"
        "}"
        "section[data-testid='stSidebar'] div[role='radiogroup'] > label:hover {"
        "background-color: #fef3c7 !important;"
        "border-color: #f59e0b !important;"
        "}"
        "section[data-testid='stSidebar'] div[role='radiogroup'] > label:has(input:checked) {"
        "background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;"
        "border-color: #f59e0b !important;"
        "box-shadow: 0 4px 12px rgba(220, 38, 38, 0.28) !important;"
        "}"
        "[data-testid='stVerticalBlockBorderWrapper'], div[data-testid='stForm'], .clean-card {"
        "background-color: rgba(255, 255, 255, 0.95) !important;"
        "border: 1.5px solid #fcd34d !important;"
        "border-radius: 18px !important;"
        "box-shadow: 0 6px 18px rgba(217, 119, 6, 0.08), 0 2px 6px rgba(220, 38, 38, 0.05) !important;"
        "}"
        "div[data-testid='stMetric'] {"
        "background: linear-gradient(180deg, #ffffff 0%, #fffbeb 100%) !important;"
        "border: 1.5px solid #fcd34d !important;"
        "border-radius: 14px !important;"
        "}"
        "div[data-testid='stFormSubmitButton'] > button, .stButton > button[kind='primary'] {"
        "background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;"
        "color: #ffffff !important;"
        "border: 1.5px solid #facc15 !important;"
        "border-radius: 12px !important;"
        "box-shadow: 0 5px 14px rgba(220, 38, 38, 0.28) !important;"
        "}"
        "div[data-testid='stFormSubmitButton'] > button:hover, .stButton > button[kind='primary']:hover {"
        "background: linear-gradient(135deg, #b91c1c 0%, #d97706 100%) !important;"
        "transform: translateY(-2px) !important;"
        "}"
        ".stButton > button {"
        "background-color: #ffffff !important;"
        "border: 1.5px solid #fde68a !important;"
        "border-radius: 12px !important;"
        "}"
        ".stButton > button:hover {"
        "background-color: #fef3c7 !important;"
        "border-color: #f59e0b !important;"
        "}"
        ".edu-warning-box {"
        "background: linear-gradient(135deg, #fffbeb 0%, #fef2f2 100%) !important;"
        "border: 1.5px solid #fcd34d !important;"
        "border-left: 5px solid #dc2626 !important;"
        "color: #7f1d1d !important;"
        "}"
        ".tet-gold-strip {"
        "height: 6px; width: 100%; border-radius: 999px; margin-bottom: 10px;"
        "background: repeating-linear-gradient(45deg, #dc2626, #dc2626 12px, #facc15 12px, #facc15 24px);"
        "}"
        ".tet-lantern-bar {"
        "position: fixed; top: 0; left: 0; width: 100%; height: 26px;"
        "display: flex; justify-content: space-around; align-items: center;"
        "pointer-events: none; z-index: 9998; font-size: 0.9rem; opacity: 0.88;"
        "}"
        ".t-lamp { display: inline-block; animation: tetSway 3s ease-in-out infinite; }"
        "@keyframes tetSway {"
        "  0%, 100% { transform: rotate(-6deg); }"
        "  50% { transform: rotate(6deg); }"
        "}"
        ".tet-petal-container {"
        "position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;"
        "pointer-events: none; overflow: hidden; z-index: 9997;"
        "}"
        ".tet-petal {"
        "position: fixed; top: -10%; opacity: 0.78; user-select: none;"
        "animation-name: tetFall, tetDrift;"
        "animation-timing-function: linear, ease-in-out;"
        "animation-iteration-count: infinite, infinite;"
        "}"
        "@keyframes tetFall {"
        "  0% { top: -8%; }"
        "  100% { top: 105%; }"
        "}"
        "@keyframes tetDrift {"
        "  0%, 100% { transform: translateX(0px) rotate(0deg); }"
        "  50% { transform: translateX(26px) rotate(28deg); }"
        "}"
        ".p1 { left: 5%; font-size: 1.05rem; animation-duration: 11s, 3.6s; animation-delay: 0s, 0s; }"
        ".p2 { left: 14%; font-size: 0.95rem; animation-duration: 13.5s, 4.1s; animation-delay: 2s, 0.8s; }"
        ".p3 { left: 23%; font-size: 0.9rem; animation-duration: 15s, 4.4s; animation-delay: 4s, 1.2s; }"
        ".p4 { left: 32%; font-size: 1.1rem; animation-duration: 12s, 3.8s; animation-delay: 1s, 1.8s; }"
        ".p5 { left: 41%; font-size: 0.85rem; animation-duration: 10.5s, 3.2s; animation-delay: 5s, 0.3s; }"
        ".p6 { left: 50%; font-size: 1.05rem; animation-duration: 13s, 4.0s; animation-delay: 3s, 1.5s; }"
        ".p7 { left: 59%; font-size: 0.95rem; animation-duration: 11.5s, 3.5s; animation-delay: 6s, 0.9s; }"
        ".p8 { left: 68%; font-size: 1.15rem; animation-duration: 14s, 4.3s; animation-delay: 2.2s, 1.1s; }"
        ".p9 { left: 77%; font-size: 0.95rem; animation-duration: 15.5s, 4.6s; animation-delay: 0.6s, 2.0s; }"
        ".p10 { left: 85%; font-size: 1.0rem; animation-duration: 12.2s, 3.7s; animation-delay: 4.2s, 0.5s; }"
        ".p11 { left: 91%; font-size: 0.85rem; animation-duration: 10.8s, 3.3s; animation-delay: 1.7s, 1.4s; }"
        ".p12 { left: 96%; font-size: 1.05rem; animation-duration: 13.2s, 4.0s; animation-delay: 3.1s, 0.7s; }"
        "</style>"
    )
    st.markdown(tet_css + petals_html, unsafe_allow_html=True)


def inject_christmas_theme():
    """Injects a cute, cozy, high-contrast Christmas theme with gentle falling snow & twinkling lights."""
    snowflakes_html = (
        "<div class='xmas-snow-container' aria-hidden='true'>"
        "<div class='xmas-flake f1'>❄️</div>"
        "<div class='xmas-flake f2'>❅</div>"
        "<div class='xmas-flake f3'>❆</div>"
        "<div class='xmas-flake f4'>❄️</div>"
        "<div class='xmas-flake f5'>✦</div>"
        "<div class='xmas-flake f6'>❅</div>"
        "<div class='xmas-flake f7'>❄️</div>"
        "<div class='xmas-flake f8'>❆</div>"
        "<div class='xmas-flake f9'>❄️</div>"
        "<div class='xmas-flake f10'>❅</div>"
        "<div class='xmas-flake f11'>✦</div>"
        "<div class='xmas-flake f12'>❄️</div>"
        "</div>"
        "<div class='xmas-lights-bar' aria-hidden='true'>"
        "<span class='xbulb b-red'></span><span class='xbulb b-gold'></span><span class='xbulb b-green'></span><span class='xbulb b-blue'></span>"
        "<span class='xbulb b-red'></span><span class='xbulb b-gold'></span><span class='xbulb b-green'></span><span class='xbulb b-blue'></span>"
        "<span class='xbulb b-red'></span><span class='xbulb b-gold'></span><span class='xbulb b-green'></span><span class='xbulb b-blue'></span>"
        "<span class='xbulb b-red'></span><span class='xbulb b-gold'></span><span class='xbulb b-green'></span><span class='xbulb b-blue'></span>"
        "</div>"
    )
    xmas_css = (
        "<style>"
        "/* Cozy Winter Christmas Background */"
        "html, body, .stApp, [data-testid='stAppViewContainer'], [data-testid='stHeader'], main {"
        "background: linear-gradient(180deg, #eff6ff 0%, #fff5f5 48%, #f0fdf4 100%) !important;"
        "background-attachment: fixed !important;"
        "color: #1e293b !important;"
        "accent-color: #dc2626 !important;"
        "}"
        "section[data-testid='stSidebar'], section[data-testid='stSidebar'] > div {"
        "background: linear-gradient(180deg, #fff1f2 0%, #ffffff 55%, #ecfdf5 100%) !important;"
        "border-right: 2px solid #fecdd3 !important;"
        "}"
        "/* Festive Sidebar Pill Tabs */"
        "section[data-testid='stSidebar'] div[role='radiogroup'] > label {"
        "background-color: rgba(255,255,255,0.85) !important;"
        "border: 1px solid #fecdd3 !important;"
        "}"
        "section[data-testid='stSidebar'] div[role='radiogroup'] > label:hover {"
        "background-color: #ffe4e6 !important;"
        "border-color: #fb7185 !important;"
        "}"
        "section[data-testid='stSidebar'] div[role='radiogroup'] > label:has(input:checked) {"
        "background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;"
        "border-color: #991b1b !important;"
        "box-shadow: 0 4px 12px rgba(220, 38, 38, 0.25) !important;"
        "}"
        "/* Frosted Winter Cards */"
        "[data-testid='stVerticalBlockBorderWrapper'], div[data-testid='stForm'], .clean-card {"
        "background-color: rgba(255, 255, 255, 0.94) !important;"
        "border: 1.5px solid #fecdd3 !important;"
        "border-radius: 18px !important;"
        "box-shadow: 0 6px 18px rgba(220, 38, 38, 0.06), 0 2px 6px rgba(22, 163, 74, 0.04) !important;"
        "}"
        "div[data-testid='stMetric'] {"
        "background: linear-gradient(180deg, #ffffff 0%, #fff7f7 100%) !important;"
        "border: 1.5px solid #fecdd3 !important;"
        "border-radius: 14px !important;"
        "}"
        "/* Cute Holly-Red Primary Buttons */"
        "div[data-testid='stFormSubmitButton'] > button, .stButton > button[kind='primary'] {"
        "background: linear-gradient(135deg, #ef4444 0%, #dc2626 55%, #b91c1c 100%) !important;"
        "color: #ffffff !important;"
        "border: 1.5px solid #fca5a5 !important;"
        "border-radius: 12px !important;"
        "box-shadow: 0 5px 14px rgba(220, 38, 38, 0.28) !important;"
        "}"
        "div[data-testid='stFormSubmitButton'] > button:hover, .stButton > button[kind='primary']:hover {"
        "background: linear-gradient(135deg, #dc2626 0%, #16a34a 100%) !important;"
        "transform: translateY(-2px) !important;"
        "}"
        "/* Secondary Buttons with Cute Pine-Green Hover */"
        ".stButton > button {"
        "background-color: #ffffff !important;"
        "border: 1.5px solid #bbf7d0 !important;"
        "border-radius: 12px !important;"
        "}"
        ".stButton > button:hover {"
        "background-color: #f0fdf4 !important;"
        "border-color: #16a34a !important;"
        "}"
        "/* Festive Disclaimer Box */"
        ".edu-warning-box {"
        "background: linear-gradient(135deg, #f0fdf4 0%, #fff1f2 100%) !important;"
        "border: 1.5px solid #86efac !important;"
        "border-left: 5px solid #16a34a !important;"
        "color: #14532d !important;"
        "}"
        "/* Candy-Cane Strip Utility */"
        ".xmas-candy-strip {"
        "height: 6px; width: 100%; border-radius: 999px; margin-bottom: 10px;"
        "background: repeating-linear-gradient(45deg, #ef4444, #ef4444 10px, #ffffff 10px, #ffffff 18px, #16a34a 18px, #16a34a 28px, #ffffff 28px, #ffffff 36px);"
        "}"
        "/* Twinkling Fairy Lights Bar */"
        ".xmas-lights-bar {"
        "position: fixed; top: 0; left: 0; width: 100%; height: 22px;"
        "display: flex; justify-content: space-around; align-items: flex-start;"
        "pointer-events: none; z-index: 9998; padding-top: 2px;"
        "}"
        ".xbulb {"
        "width: 9px; height: 13px; border-radius: 50% 50% 55% 55%; display: inline-block;"
        "animation: xmasTwinkle 2.2s ease-in-out infinite;"
        "}"
        ".b-red { background: #ef4444; box-shadow: 0 0 8px #ef4444; animation-delay: 0s; }"
        ".b-gold { background: #f59e0b; box-shadow: 0 0 8px #fbbf24; animation-delay: 0.55s; }"
        ".b-green { background: #22c55e; box-shadow: 0 0 8px #22c55e; animation-delay: 1.1s; }"
        ".b-blue { background: #38bdf8; box-shadow: 0 0 8px #38bdf8; animation-delay: 1.65s; }"
        "@keyframes xmasTwinkle {"
        "  0%, 100% { opacity: 0.95; transform: scale(1); }"
        "  50% { opacity: 0.35; transform: scale(0.82); }"
        "}"
        "/* Gentle Falling Snowflakes */"
        ".xmas-snow-container {"
        "position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;"
        "pointer-events: none; overflow: hidden; z-index: 9997;"
        "}"
        ".xmas-flake {"
        "position: fixed; top: -10%; color: #93c5fd; opacity: 0.72; user-select: none;"
        "animation-name: xmasFall, xmasSway;"
        "animation-timing-function: linear, ease-in-out;"
        "animation-iteration-count: infinite, infinite;"
        "text-shadow: 0 0 5px rgba(255,255,255,0.9);"
        "}"
        "@keyframes xmasFall {"
        "  0% { top: -8%; }"
        "  100% { top: 105%; }"
        "}"
        "@keyframes xmasSway {"
        "  0%, 100% { transform: translateX(0px) rotate(0deg); }"
        "  50% { transform: translateX(22px) rotate(18deg); }"
        "}"
        ".f1 { left: 4%; font-size: 1.05rem; animation-duration: 11s, 3.5s; animation-delay: 0s, 0s; }"
        ".f2 { left: 12%; font-size: 0.85rem; animation-duration: 14s, 4.2s; animation-delay: 2s, 1s; color: #f472b6; }"
        ".f3 { left: 21%; font-size: 1.15rem; animation-duration: 12.5s, 3.8s; animation-delay: 4s, 0.5s; }"
        ".f4 { left: 30%; font-size: 0.95rem; animation-duration: 15s, 4.5s; animation-delay: 1s, 2s; }"
        ".f5 { left: 39%; font-size: 0.85rem; animation-duration: 10.5s, 3.2s; animation-delay: 5s, 0s; color: #fbbf24; }"
        ".f6 { left: 48%; font-size: 1.1rem; animation-duration: 13.5s, 4.0s; animation-delay: 3s, 1.5s; }"
        ".f7 { left: 57%; font-size: 0.9rem; animation-duration: 11.8s, 3.6s; animation-delay: 6s, 0.8s; }"
        ".f8 { left: 66%; font-size: 1.2rem; animation-duration: 14.5s, 4.4s; animation-delay: 2.5s, 1.2s; }"
        ".f9 { left: 75%; font-size: 0.95rem; animation-duration: 12s, 3.7s; animation-delay: 0.7s, 2.1s; }"
        ".f10 { left: 83%; font-size: 0.85rem; animation-duration: 15.5s, 4.6s; animation-delay: 4.5s, 0.4s; color: #f472b6; }"
        ".f11 { left: 90%; font-size: 0.9rem; animation-duration: 11.2s, 3.3s; animation-delay: 1.8s, 1.6s; color: #fbbf24; }"
        ".f12 { left: 96%; font-size: 1.1rem; animation-duration: 13s, 4.1s; animation-delay: 3.4s, 0.9s; }"
        "</style>"
    )
    st.markdown(xmas_css + snowflakes_html, unsafe_allow_html=True)


def render_sidebar_and_get_lang() -> tuple:
    """Renders the translated EIU logo, smart language selector, holiday theme + instrumental music, and disclaimer."""
    if st.session_state.get("_sidebar_done", False):
        lang_code = LANGUAGES[st.session_state.get("lang_choice", "🇬🇧 English")]
        return lang_code, TRANSLATIONS[lang_code]

    if "lang_choice" not in st.session_state:
        st.session_state["lang_choice"] = "🇻🇳 Tiếng Việt"
    if "ui_theme" not in st.session_state:
        st.session_state["ui_theme"] = "christmas"

    cur_lang = LANGUAGES[st.session_state["lang_choice"]]
    uni_title, uni_sub, _, lang_lbl, _ = LOGO_TEXT_I18N.get(cur_lang, LOGO_TEXT_I18N["en"])

    # 1. EIU Logo Card at top of Sidebar (Translated!)
    logo_slot = st.sidebar.empty()

    # 2. Smart Compact Language Switcher
    selected_label = st.sidebar.selectbox(
        lang_lbl,
        options=list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(st.session_state["lang_choice"]),
        key="_lang_select_widget",
    )
    st.session_state["lang_choice"] = selected_label
    lang_code = LANGUAGES[selected_label]
    t = TRANSLATIONS[lang_code]
    uni_title, uni_sub, _, _, _ = LOGO_TEXT_I18N.get(lang_code, LOGO_TEXT_I18N["en"])
    th_cfg = THEME_I18N.get(lang_code, THEME_I18N["en"])

    # 2b. Holiday Theme Selector (Christmas / Tết / Standard Medical)
    theme_options = ["christmas", "tet", "medical"]
    selected_theme = st.sidebar.selectbox(
        th_cfg["label"],
        options=theme_options,
        index=theme_options.index(st.session_state["ui_theme"]),
        format_func=lambda k: th_cfg[k],
        key="_ui_theme_widget",
    )
    st.session_state["ui_theme"] = selected_theme

    # 2c. Theme Styling + Instrumental Music Player for Holiday Themes
    if selected_theme == "christmas":
        inject_christmas_theme()
        st.sidebar.caption(th_cfg["xmas_music"])
        st.sidebar.audio(build_holiday_music_wav("christmas"), format="audio/wav", loop=True, autoplay=True)
        logo_slot.markdown(
            f"<div style='background:linear-gradient(180deg,#ffffff 0%,#fff5f5 100%); padding:12px 14px 12px 14px; "
            f"border-radius:16px; border:1.5px solid #fecdd3; text-align:center; margin-bottom:10px; "
            f"box-shadow:0 4px 12px rgba(220,38,38,0.08);'>"
            f"<div class='xmas-candy-strip'></div>"
            f"<img src='data:image/png;base64,{EIU_LOGO_B64}' style='max-width:158px; height:auto; margin-bottom:6px;'/>"
            f"<div style='font-size:0.74rem; font-weight:800; color:#b91c1c; letter-spacing:0.3px; line-height:1.3;'>{uni_title}</div>"
            f"<div style='font-size:0.7rem; font-weight:600; color:#15803d; margin-top:2px;'>{uni_sub}</div>"
            f"<div style='margin-top:8px; display:inline-block; background:#fef2f2; border:1px solid #fca5a5; "
            f"color:#b91c1c; font-size:0.68rem; font-weight:700; padding:3px 9px; border-radius:999px;'>"
            f"{th_cfg['xmas_side']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    elif selected_theme == "tet":
        inject_tet_theme()
        st.sidebar.caption(th_cfg["tet_music"])
        st.sidebar.audio(build_holiday_music_wav("tet"), format="audio/wav", loop=True, autoplay=True)
        logo_slot.markdown(
            f"<div style='background:linear-gradient(180deg,#ffffff 0%,#fffbeb 100%); padding:12px 14px 12px 14px; "
            f"border-radius:16px; border:1.5px solid #fcd34d; text-align:center; margin-bottom:10px; "
            f"box-shadow:0 4px 12px rgba(217,119,6,0.10);'>"
            f"<div class='tet-gold-strip'></div>"
            f"<img src='data:image/png;base64,{EIU_LOGO_B64}' style='max-width:158px; height:auto; margin-bottom:6px;'/>"
            f"<div style='font-size:0.74rem; font-weight:800; color:#b91c1c; letter-spacing:0.3px; line-height:1.3;'>{uni_title}</div>"
            f"<div style='font-size:0.7rem; font-weight:600; color:#b45309; margin-top:2px;'>{uni_sub}</div>"
            f"<div style='margin-top:8px; display:inline-block; background:#fef2f2; border:1px solid #fcd34d; "
            f"color:#b91c1c; font-size:0.68rem; font-weight:700; padding:3px 9px; border-radius:999px;'>"
            f"{th_cfg['tet_side']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        logo_slot.markdown(
            f"<div style='background:#ffffff; padding:14px 14px 12px 14px; border-radius:14px; border:1.5px solid #cbd5e1; "
            f"text-align:center; margin-bottom:10px; box-shadow:0 3px 10px rgba(15,23,42,0.04);'>"
            f"<img src='data:image/png;base64,{EIU_LOGO_B64}' style='max-width:165px; height:auto; margin-bottom:6px;'/>"
            f"<div style='font-size:0.74rem; font-weight:800; color:#1e3a8a; letter-spacing:0.3px; line-height:1.3;'>{uni_title}</div>"
            f"<div style='font-size:0.7rem; font-weight:600; color:#475569; margin-top:2px;'>{uni_sub}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # 3. Placeholder for Navigation Menu (populated by app.py if in flat mode)
    st.session_state["_nav_placeholder"] = st.sidebar.container()

    # 4. Educational Disclaimer Box at bottom
    st.sidebar.markdown(
        f"<div class='edu-warning-box'>{t['disclaimer_box']}</div>",
        unsafe_allow_html=True,
    )
    st.session_state["_sidebar_done"] = True
    return lang_code, t


@st.cache_resource
def load_artifacts():
    base_dir = Path(__file__).resolve().parent

    def find_file(subpath: str, filename: str) -> Path:
        for candidate in [
            base_dir / subpath / filename,
            base_dir / filename,
            base_dir.parent / subpath / filename,
            base_dir.parent / filename,
        ]:
            if candidate.exists():
                return candidate
        return base_dir / subpath / filename

    meta_path = find_file("models", "model_meta.json")
    inydb_path = find_file("models", "inydb")
    pipeline_path = find_file("models", "pipeline.joblib")
    baseline_path = find_file("models", "baseline_logreg.joblib")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    main_pipe = None
    base_pipe = None
    try:
        import joblib
        if inydb_path.exists():
            main_pipe = joblib.load(inydb_path)
        elif pipeline_path.exists():
            main_pipe = joblib.load(pipeline_path)
        if baseline_path.exists():
            base_pipe = joblib.load(baseline_path)
    except Exception:
        main_pipe = None
        base_pipe = None

    return main_pipe, base_pipe, meta


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
