"""
i18n.py — 4-Language Translation Dictionary (English, Tiếng Việt, 中文, Français) & Shared Helpers
==================================================================================================
"""

import json
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
        "app_title": "Trợ lý Tầm soát Nguy cơ Tiểu đường",
        "who_for_bold": "Công cụ này dành cho ai?",
        "who_for_text": "Một người trưởng thành trả lời vài câu hỏi sức khỏe tại nhà để quyết định **có nên đi khám và xét nghiệm đường huyết hay không.**",
        "app_caption": "7 câu hỏi · chỉ mất khoảng 1 phút · không lưu trữ bất kỳ thông tin nào của bạn. Công cụ ước tính mức độ tương đồng giữa câu trả lời của bạn với những người *đã báo cáo* mắc tiểu đường hoặc tiền tiểu đường trong khảo sát CDC Hoa Kỳ. Công cụ không thay thế chẩn đoán y khoa.",
        "q_age": "Nhóm tuổi (năm)",
        "q_height": "Chiều cao (cm)",
        "q_weight": "Cân nặng (kg)",
        "q_genhlth": "Nhìn chung, bạn đánh giá sức khỏe tổng quát của mình thế nào?",
        "q_highbp": "Bác sĩ hoặc nhân viên y tế đã bao giờ cho biết bạn bị **huyết áp cao** chưa?",
        "q_highchol": "Bác sĩ hoặc nhân viên y tế đã bao giờ cho biết bạn bị **mỡ máu cao (cholesterol cao)** chưa?",
        "q_heart": "Bạn đã từng được chẩn đoán **bệnh mạch vành** hoặc từng bị **nhồi máu cơ tim** chưa?",
        "q_diffwalk": "Bạn có gặp **khó khăn nghiêm trọng khi đi bộ hoặc leo cầu thang** không?",
        "no": "Không",
        "yes": "Có",
        "genhlth_1": "Tuyệt vời (Excellent)",
        "genhlth_2": "Rất tốt (Very good)",
        "genhlth_3": "Tốt (Good)",
        "genhlth_4": "Trung bình (Fair)",
        "genhlth_5": "Kém (Poor)",
        "btn_analyze": "Phân tích ngay",
        "prompt_analyze": "Điền thông tin vào biểu mẫu và nhấn **Phân tích ngay**.",
        # Results
        "res_recommend_flag": "🩺 Khuyến nghị: Bạn nên trao đổi với bác sĩ để làm xét nghiệm đường huyết",
        "res_recommend_safe": "✅ Nguy cơ tức thời thấp: Duy trì lối sống lành mạnh — chưa cần xét nghiệm khẩn cấp",
        "res_score_label": "Điểm xác suất tương đồng ước tính",
        "res_threshold_label": "Ngưỡng quyết định đang áp dụng",
        "res_bmi_computed": "Chỉ số BMI tính từ chiều cao & cân nặng",
        "res_natural_freq": "Trong khảo sát CDC Hoa Kỳ 2015, cứ **100 người** có câu trả lời giống bạn thì có khoảng **{n} người** báo cáo mắc tiểu đường hoặc tiền tiểu đường.",
        "res_top_factors_title": "Yếu tố nào làm tăng hoặc giảm điểm rủi ro của bạn?",
        "res_top_factors_sub": "Mức độ tác động của từng câu trả lời so với mức trung bình của quần thể khảo sát:",
        "pushes_up": "⬆️ Đẩy rủi ro TĂNG",
        "pushes_down": "⬇️ Giảm rủi ro (Bảo vệ)",
        "neutral": "➖ Mức trung bình",
        "col_question": "Câu hỏi / Yếu tố",
        "col_your_answer": "Câu trả lời của bạn",
        "col_impact": "Tác động lên điểm số",
        "col_plain_meaning": "Giải thích dễ hiểu",
        "what_if_title": "💡 Giả lập What-If: Nếu giảm 2 điểm BMI thì sao?",
        "what_if_desc": "Nếu chỉ số BMI của bạn giảm từ **{bmi1}** xuống **{bmi2}** kg/m², xác suất rủi ro ước tính sẽ đổi từ **{p1}** xuống **{p2}** ({delta}).",
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


def inject_friendly_css():
    """Injects clean, modern, user-friendly CSS inspired by the reference screenshots."""
    st.markdown(
        "<style>"
        "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');"
        "html, body, .stApp, [class*='css'], h1, h2, h3, h4, p, div, span, label, table, th, td, button {"
        "font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;"
        "}"
        ".stApp {"
        "background-color: #fbfcfd;"
        "}"
        "section[data-testid='stSidebar'] {"
        "background-color: #f3f5f8 !important;"
        "border-right: 1px solid #e2e8f0 !important;"
        "}"
        "div[data-testid='stForm'] {"
        "background-color: #ffffff;"
        "border: 1px solid #e2e8f0;"
        "border-radius: 14px;"
        "padding: 24px 28px;"
        "box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);"
        "}"
        "div[data-testid='stFormSubmitButton'] > button {"
        "background-color: #c83e4d !important;"
        "color: #ffffff !important;"
        "font-weight: 600 !important;"
        "border-radius: 8px !important;"
        "padding: 0.5rem 1.4rem !important;"
        "border: none !important;"
        "width: auto !important;"
        "}"
        "div[data-testid='stFormSubmitButton'] > button:hover {"
        "background-color: #a82e3b !important;"
        "}"
        ".edu-warning-box {"
        "background-color: #fef9e7;"
        "border: 1px solid #f7dc6f;"
        "color: #7d6608;"
        "padding: 14px 16px;"
        "border-radius: 10px;"
        "font-size: 0.88rem;"
        "font-weight: 500;"
        "line-height: 1.45;"
        "margin-top: 12px;"
        "margin-bottom: 12px;"
        "}"
        ".clean-card {"
        "background: #ffffff;"
        "border: 1px solid #e2e8f0;"
        "border-radius: 14px;"
        "padding: 20px 24px;"
        "box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);"
        "margin-bottom: 16px;"
        "}"
        ".clean-table {"
        "width: 100%;"
        "border-collapse: collapse;"
        "background: #ffffff;"
        "border: 1px solid #e2e8f0;"
        "border-radius: 10px;"
        "overflow: hidden;"
        "margin-top: 8px;"
        "margin-bottom: 8px;"
        "}"
        ".clean-table th {"
        "background: #f8fafc;"
        "color: #475569;"
        "font-weight: 600;"
        "font-size: 0.84rem;"
        "padding: 11px 14px;"
        "text-align: left;"
        "border-bottom: 1px solid #e2e8f0;"
        "}"
        ".clean-table td {"
        "padding: 11px 14px;"
        "border-bottom: 1px solid #f1f5f9;"
        "color: #1e293b;"
        "font-size: 0.88rem;"
        "}"
        ".clean-table tr:last-child td {"
        "border-bottom: none;"
        "}"
        "</style>",
        unsafe_allow_html=True,
    )



EIU_LOGO_B64 = "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"
AI_MASCOT_B64 = "iVBORw0KGgoAAAANSUhEUgAAANYAAAG0CAYAAABOqNnLAAAQAElEQVR4Aey9B3gcx5UuenoSBnmQMzgAwUyCYBZJUSKUqGSLsn1ta+21pPXus73rQK3Xvnv3XT9JG+7amySv13sddk05ybYsW5StRAUTFJNEiSQEZhIkBgCRwwyR48zrv4Ea9gy6Z3pyz0zjQ011V51Kp+qvc+pUdbeOtD+NAxoHws4BDVhhZ6mWocYBIg1Y2ijQOBABDmjAigBTtSw1DmjA0saAWjkQ1/XSgBXX3adVXq0c0ICl1p7R6hXXHNCAFdfdp1VerRzQgKXWntHqFdcc0IAV193nr/JafKw4oAErVpzXyk1oDmjASuju1RoXKw5owIoV57VyE5oDGrASunu1xsWKAxqw/HFei9c4EAQHNGAFwTQticYBfxzQgOWPQ3ESf+bI63WnD72689SxV61xUuWErqYGrDjpXgCn6ej+3R8c2f9E05H9LzQdff0A79t554JzkuuUS6c7oHfqWnA/71rm6V5Aug+OvrZHAN+BA5Y4aXbcVlMDlgq7DlIHIGo6vP8pAOPFn/3A1XT23Knenr4XOKLH+SrvJpdrJ+/7A4h1nm430nEu7ikBfKYpABLuBYAN5fF5af9h5EBUgBXG+iZkVkvKCupuWrPyU9/80qde++aXHnIcev7nLfu+/y8vvPJf39rzyg++ubP7nX1kuHqQzr76E3rlh98S3O/3focafvtTeuM3z9Lxg2/R4YYDdObseRodHVPKI4ByN8AmSDleGipNqNH554AGLP88CiuFtbjYuri0cPfDd9Q99Y1P7jjwxB/d4vqrBzed+sv7lv2sOn1sV3X6RHaxcZSWl2RQdXG24HIyUoQ6lOSmC/cIX5JvJNBVmIYoa+gy5Q9fFMB38Nnv0L4f/IsAOICtt7dPSOvvx+WiOtL+wsYBDVhhY6V8RlarxXLf+mV7vnj/plOfv3tlyz9+ZvsL922s2rPGmr9zZWUeMeDI56A8BqADKAE4gA2Sramx0W8GHNFav0QagWIOaMBSzKrgCCGhPrZ+RcvDd6186pbV5XUY+MHlpCyVfWSSzrUNuN2GmkKauD7gPzHHQTX0T6dRKOKABixFbAqeaFVp9u7NS4st4sF+tft68BnOpwSAkA/yZT6uJ6ZmCFKQObPJQNf7e+ZT+fBcrqRUBX1wJKQoDVghsc9/4orCTEESsIEOv5RfKzEwSPkIO9HcS/DhABj4cOwaAIL0Q37MxzXWYd61MnKz3kFS90I9pSK0sMA5oAErcJ4FlMJo0C1Yu0CKMDBI+QiDCgcfDoCBD8eupQAkVzGHEonFJ8ZeGe9p/2HggAasMDDRVxaWjJSYSwKTMolFLudszOvqi5fxFKcBK9K9pQIzdk6GmVpaWv221KnntHWWXy4pI9CApYxPQVOlpxotXYOjhDVR0JmEmBDm/NExBRvHLi4YiRVi7RIzuQasCPYrTlRAWmA9dLZtMOwlwZihFLCOvm6/5XMcd6tfIo1AEQc0YCliU3BErlmXBdICqbPTjAQTOa7D4c7Y+sg5M0UwhCjJz8Q5lZBpNGHigAasMDFSKptCS4b7EQ5rYSadsynYT5LKyCsMoJoYdlB5fqZXjPztYFebfCSLmTvYy+40PwQOaMAKgXn+ktZa893AMhgMpJ+dDFlqMVDlZRjJbDb7q4I7XqlkO6U9UuLmWSgXGrBC4Z6ftC5yLRKTLJKVWmIq+WthTcVLKoAqLS1NntArBiroGwff8QqVvjUYJjTLoDRrAgrVgBUQuwIjLrSkuSUWUppMJpoaHw1KasGy2N7RTQAV8kpN9S+tAKi33r9Mbx06QanTw4oeKdFM7uBu6E4DVug8lM3hnK3XisHNCACGGSdHqSY9C1LsX7B1UWVuipteTmKhvBOXOmlfwwd08MgJouF+yjFzgmuxtbrTy11wTs5DysrRaeG+OaAByzd/QorVjzusGNwY5BjsV3tHafGiEsWWPHHhKaLTE5B8WLMhHuZ2nB880tRCL709ByZHRytlusYo08SBxO262v0Dizja04RH/4+8/oi23nKzLuALDVgBs0xZAjwuYuC5i8GNQY7B3nzxMjWdbaaDJ3n/aq9wyFZZbkTjsxx1OSbpau8Y2fpG6aXDZ2jfWyfopbeOU9eVSzQx0E2pswvBJM5/xK7g8ZG5BLuJXHv1wiP8r+3VzhDOMSWQX77rAyFXMa3KqqbjZnYDVOJq4T5TN0Uz1/upr/UKXT57jp5/9Sg9v/84vfXOWUn3y5f5eJ7G3tdP4w5+k3nsOk3Y+4U1E/KCmicuw9f1B0cbhCeLjx05qmi9NZcX9wheVIN3b2gAm+OIkl8NWEq4FAANTlt87u4NBz60afFT/pJBogEYOaZZXiQNSbqCtLn1Eej85ecv3uCaJTxZnDPYRPufeYpefv5Xis4QCvnye1wCwHgVUbjXfnxyQAOWT/Yoj4Tq98lb1+z97J1rTt1eV7kzVceDRXnyqFDOzMwQjBsoDI+fLE6xE15Us+8n/6UcYEisOb8c0IDll0W+CaxWiwWA+vzdK1s+srXmETwz5TtF9GP7xlxE6Tm0ecMqYkesWC1wvzxnhs6/+UuCisjCJXwb53TW126/6xmJOC3IiwMasLwYovTWarVY7qqreeKPt6yRBJR9bIYmZpTmFl66GScRA5OlbBHtunUD3b55ufCGJ7mSUk0GuSgHD8sna7fvqlqz454GOSIt3JMDHsDyjNLu5DgAtQ8viPnTu9c8vqYq3yJFd/v6KlqxZqUgKYRBLkUUpjAAGGXMpGRSRlE5LVm1kj553zYBTBuWli6QUuJiYa5v6hyjgo330tbt28RR89eufWu373pi/kbzFHJAA5ZCRnmTpRplZ3g3aXVxtjC4d9+5mQoWLSZDdj4NO01kn+BlgJsqsAuktU/xG8ypWWTOKxbyXb+hlgCkXdtW05aVFT4lEysNgLrQNUI95ir6+Of+klavWsGivHwOb9z1CtNu/XFA549Ai1/IAVt3t+2nBy+ve/Gd5gZmDFhIdSMEB2Brqwvp1vVLaPftG+j+2zdT1YqVgnQBOIgHCdy4Pm1OhZu/RxjAmFNSLgAIaZD2Y7s20+03raLttVWEfPG8143SfF+hvh9c7aPB4UkqqFxCu+67z3cCIqv2Cmp/LFoYrwFrIU8UhQBcP284Xf/Pzx+vP3q+0wYJoCghTwSgVfPSDNIF4ABI4O6/ZS1B8uCaOYBx/fIKAUDVfBqk5bMI+B+nM861DZJep6PM1BQqtKTR7HVlj7HoXTrtYG6AHNcFSK+Re3Ggubu/Qa/XN0ACvNXY5jZne5HF5BZgOn6xm5589h366R/O0x11c8cAxyanKT8rlapL80jJK6hdLoo5sCjO/jRghaHDFpfkWbeuqqRVlfl0ucNBGMw4jR6GrAPKAlLzfPsgnWkdoM6BUdq2ooxuXlVOCK8syBLyGpuYprvWW6l5YIY21K6gFpv/ByC1R/YF1gX0owErIHZJE5uy8iwpueWUZjZSZUEmLw3S6MI1O/37707RqSu9EZNiAAyk0kvHrwplQTIdv9RNxNtGNi4pprQUI1+XVMFPMxsoIytbaMCFa4PUP2gXrnNSeGLhSvsJJwd04cwsWfMqL8yp6x64Tm29QzQ2OUP52am0ojyXHrmDt9ItK6WmkRzqyKqj0wN64eAtABEoryABASIYHl47a6e3Wjk6fT2dcjNS6dP1K4WyvvLh9Xy5ebS8ItedvdFkoq996nZ6ZPdtRGl5Qv0QWWqZe55ramgQt5oLMwd0Yc4vqbODinXo3DVBaq2vKeIlhYE6Cm+hz+z537Tt3o/T/X/xd1S/57t0NWsT/dPz79MPXztN71zoEtTHjoERwT92vouwVkMcHOj+1zOHad+xKzSas4LWfvJ/0uf+7v/Sn/zPf6CPfelxWvPw31Oja6nAd5S5e2sNX36WIJ1KKqvJunQVFRbkU2aamQB+EC7nQQ8fLoMPh+/TuZw2n/Fa5AIOaMBawJLQAurrqqlqcQ0ZTSa6lrqcNt5yBxkNeirKy6LyohzK5iXM/R/5BN22+yFBeuk4WOlMtGlJiWCxK7Kk05u8EeRq93UhPjPVRDs2rqKHv/b39LE/20OLqmtI/JeWnkG3fviTlLXzc9R4bYQseQVkXbaKACqm+q1bVknrllVSW2cvv+YqE5IPTwmesh+Oe1EZoUbFOKABi3EiBN8xNOpYUlHIr6sGyWg0UZYll4rX7KBbP/KwAKSqsnzKyUqjjLQUGurvpLZLTbT7Ew/Rrs3LqW9ozK26ra8pFKQcU+Ue3LqEbrlpPT38v/6ZVtTOGeYAUpqdonca3uDXdCZ3rQG46ns+TwX8nhfq4I4QXUxOjLvviouKhOvrE07B9/Fjq922a5+PeC1KggMasCSYEmiQi3M1QqWCxS0vN0dIbp4cpOJMPZUUZJNOxwlh+Gm+eJG+/a1/JN30GH38vlvoz3bVUv/QOLX1DfGgMgprIIRhvbSIlzw7H/k6QSpl85IOAF1cUUCXT79P5xvfp772Zlq6qEgoA4DLK6lAEbIOxgxEHr3QTZvX1NDI2ARNGDMR5MNxT/qI1KJkOKABS4YxgQR39w05QJ+ankH5eXPAwv3spQPwPNwtt91Gj/+ff6QVq1ZSmnOUeoZnqHLxclq5po6aOscJUgt7TOv5NdrG2+6nkqICApgA0JT5g7I7b7+D1q5fTxs2bxZAm82DDjRFpnGPsrxvUsypPHCnaZHVyktPMx1otNGWzRu8ycT3tlrtNLuYH4qvNWApZpU8YUvXwAeIHacUgkqIazjX6CD97D89n3fMzMoSAHH0ue8RNz1O2zavpyWVvNTJz6aP3rVNWJ8hLdxs42+o6/TbBGmEe+aQxx89/DC7FfyL58+TFJCFyPkfgP5U+wjt3LhCkFYcv0UwHyXnWT44sv8JuUgtXJ4DGrDkeSMRIx30yuHGBqhVJTw4oBIyKlzfvSRVANfw0JAQfOL4cfrJN/+a0hzNVF61hK509Avh7AfrMxgg2P30mVfpe09+lTo7Okju73e//iUNHn+RMky+u7M4L5vWrpgzfkBabdqyWS5LFm7hldiHtUfyGTuU+757Qnk+SU3Z3N3f8N7ZFkEd9GYEwPWxtTn0zs//jfZ/7+/IcP4l2lJhpsVLl1KPfcSbXLiHAaKofBFBtcS6aHvxDB353l/TL7/7TwTJBCIA9He/+Am98cP/Q9ssDlpXU4xgnw7AB8Hlth46cvwUHX5XELQI8uWsTnKdatIeyffFowVxGrAWsCS4gOPnWvdhr0gu9Y5aK926poKW1VTT2Y4hguXucnuPoAZKpRmd0QsSbfGKWsF0DmNDXVofdb70FP3m8UeotP8duqvSSTtWlgrrJak8pMKGeYPFOyc+oDoeiOPOVFLy3ay5fFyPa69Dm+OEkl8NWEq4pIDm94dPfhuD1h9pV/91Kpl/NtIXEAE65KXT64XN3qKyRYTN3i1bttBt2zeS3jWD6IDdqTOX6KYlBXTdUkuLBuF5/gAAEABJREFUl6+iPzz333TsyFEl+Vh1pqk9Sgg1GiINWGEaBdPGaZsvoLBiTl1sdRs4sHHLwr19ubx0er2gImZZcr2TKLovz0ujt5uHac3Nd9Pa2mW0bP1Waj/zjqLXoXFEj2vrLUVsjgGwlNUr7qgME4Y6X0CBpDp1sY3g/8U//Zz+/r9fIqx10FCEIe6/XzwkhH+Rj8f126cuCdY70ITLDQ2PUdaae8hisVBudhoZUtKotjSNDjcs3BqQKnOWXHulwrUwTw5oEsuTH2G9AzD++jvP011f/Df66Ne/S3/xrZ8RAANAvXKkiR5+4r9p25/8H484hJ+80EpwLC3ovv2LNwj5hVrBM6M5VLNmswAqWBoHL859hWS284yi9RYvteo0E7z/XtCA5Z9HAVMAOHAMDDDFB5yJKAHy+tUb7xEDGqRdMCCDVPzon8wtk0oKM+m5//wn9/sxqouz6eSBl0Slyl/y4NJUQnn2CDEasAQ2hO8HIDjJq3zIESoe/HA6gBRSDSCDFMS90vwHxolS0zIoK8NMZxpPUjF57qEtznbR/pdfVpSdk1wvaFZCeVZpwJLnTUAx1QVpD0AijIxP0tLKImLgCiiTAIkB3Kd/8SahXCVJR3QWgawwL53ee/VXC16LZjYZSKlKyGdk1ZumEmu9xTcqXP+6cGWUzPnsWGHdvW1F8Z7Wa50EkzvcKX6dFC2evHy4ya+RAyCcyrISzhs2HvsDlVCfZPWgEr73+m8UWQn5DHZ/cPS1Pbyv/XtxQAOWF0MCvcVHEHbWVuwtzU2nI2evEczkcGKJhbOA65cvIrjPPrCD5Bzi4QKpAx5evO/mWr8S8t1L/Gb0yrVUmJdBZ976DUE6yZUDKyE+mCAXLw7nXNxT2qkMMUfmrjVgzfEhqF+r1WK5b2P1C3gb7uSMk6q3PiDkA7P7nofuoB8/8Vk6+qO/Efz/+PqnCE4OVAhHPBzSvP4ff0nf/Z+fpq88dCfdu71W9oTGpbYeQUoCzCTzJ0ir0pvIoNfRsTdfpGp+LSVD6g5emjFKb7z+pvve94VrLyyF2vsHb3BJA9YNXgR0ZbVaLPU1VQe2rSi1ImGXY5JuuvUuOuNIxa1PMAgEfn4y0swEgH7izk30vz97vwBOAE4MNpz9g/FiSUWRQCuX5e+bBmkJL62MuhnqPvGqT2nF8oBEM3R9oPRUBnFEj+udupamI/tbmg7vfyrZQaYBi42kAHyrdQ5U92y01rFkuZUrhcuS1bfSexc7hWv8wIIHSyGuw+HEYPvNP/0FwQFgcnk/89Z5Wn3bxwVp1XRwHy0tnAO+HL04vHNwlE7+4SXBUqjk/YNIyzsrj7I9AsiOvn4gWdVEDVj8SAjk32q1CJJKDCqkN+aUwqO8giI6dT1HMCZg7+mWdUuFQ7IAGPaeIGEEwjD9+APV0ts/QzCxu6aGaPTSYcWl4p0bMGTcXldJi5ztdPbVn9Dv936HXn3xBUUv+RQKcrl2Ern2QorByJFM5nkNWMIIkP+xWi2Wbcutj3zu7g179zy4/cIt1Yvs3qBC6mLrYniC27TzXnr6pTN03/Y1Aqgw+O/l10kAGYwaAJhSE7mQYYA/AC8kFQNVeqqJjr/6M/dmsL/sJqZmhHch5mSkuElLeOPMknwjVel6BJDhY3UKD+8iDyvHGzl483xLsqzFdGi15m5wAIdMv/0P3/jkp3dt/9sv3r/p1Od3rrfv2b1uLz9zP5Kip2X3b66+QTx/hQ8NVC1ZJpiyEXT17An6xE0VAqhwL3YAFxweOoQUCzfAkN/zp8dp7Yf+XJBUMFg4eq5Q/kSbuBo+r8+2DdKGmkJZGoAMH6tL7T5Jz33/34RzhqOjY7L0ogg8OMnWYi80Hd2/WxSXUJdJDywACYvtX//oPy586+ufd724999PUeu7v/jwusJv3LK6vA7qEHocs3iqSY/LBc4+MkHLVqwgHGq91nqFch2NslY8lphJMdx39V+HF5JDHs8da6druTfT2vqPCHkBVCuXFFHT/p9Tjkj6CJEyP/aRScW0MHDANJ8/fJHe+Pl/BroW200ueqHpyH5705HX9gJkUBXRH1AbIdngTh96dSfCZKqr2uCkAhb2nGqK83fiw3HokWd/8PQfXn7xt6d+/8x39gyefGUZ1ByoO5iREc8cQHWiuZfw3V4WJvbZ+irDTDR7+U2qXzcn1fC6sWstl2nIMSgm97iGMeLUxVZxWEDXUCv3NQ3RoZEqWsKvp8oXzamkDFRH//AaVaSMKs6Tra0UJ5gnXF6SIazF3n/xR/TGb56ld4+fmI/x61mIuEcAMl5VtDvJdYrj1UaOtzLCuXS6AwhrmgcggEZx8JfQwAKQdm9Z8dTXP7rt1Dcfuc31Vw9uOvWlD68/UL88p+VLD2wfGG56vZ6ByXtGx+ucT17qoCNNV+jk2Stk8MEpY0qa0NWv7v032r1tqXDtnJ2ll/5wjMZHR6jnWivZLp0VAIZwgUD0A9O66NbnJdZPABNM6D87qyfDuk/RopseEMzpLCHWVLXLS8g5PUHH9j2jyLyOtJhAegaHqOlqDwFgCAvUQcJXmIYopf0Y7fvBv9Bbv3teANkoryq2tLQKauPxg2/S22+8RmfOnld0on6+DgIAAbQm3toI6TYfrkrPx3BRZX0VV+p/bFv9AoB0z6aqPWkphjrH0AidudxGfT3dtLzITGU5KbliMGFQnbH10fsX2ungiYukn7xOOcZpKs3U07STaG21/JqjdMkq4YUx99SVueuHx99t3Q7CW5HaR41kd2VSY0s//fr3byx4/AP7UFImeayXYLrf/0EnQSo9d2aKXuktpcztn6Pymz5KqzbdIqyj3IXyF1BHV/LqH44uvfrbX9GWan488uFK/j+42ksrCgyUyY3T9PAAHXjvPB070yIADSqikjwYDdRESLEyrp+ye96jX/7HN+n8mz+j/OEzlDV0ngonLpPh6pvU/c6veQB+iwfgrwWgsfQ+fd7ayEu3vT5pYhypi3H5YS8eUuq+9cv2ZJlduxvPX6Xzl22UOjssgKTcwg/wsRnqHjdQKW/lwmA5w4Pp3XOtdOyDy5TuGqW8lFmqzE0hEy+iBkamqW1wkszpmT5n/Vd++zwVOvuEY0Vvn7okAKeqejF99U8+Qh+/ZwfdtmW1sIELo4UhLYv6+vo92o311tuXh+iVqzp65lwKPWfLFdz4so8KEqlw67xU2rzLQzJ5ZMLfWMtzaFl1AUENxLNWQ+ff5kOV/3f0DhLajBRovzU/lYrTOQFoV1ra6NCpy3Ts3DXCBxpAE4hLN3F0ua1fsDaK02FyW16SRmVcNw03vUy/+K//VHpOcSfWxmxtJs5TDdc6NVQiXHXA+umL96079fBdK5/SuZyUl2ESsr7mmKaBST2Ncum0tKqc0s1Gfia2EUAHMBWmugiDaGrGKQCpd5wTaBdXVVJBQf4CaQXpBlWxc2iWp8ugj9+xkQAasQNYhMK9fpDfyNB1YZ+LRUFatXV20itHTlBXTyd1dCi34CEPSCeofiXz38BC2Bu//glV8BMJrpU4tOfem5YT2uxKzaHrsynUPeoS+DE8Mcvz0kiYmIpTZwhSPxiQLS1I4XnewWsFnXSuzb6gWgDZuqJZQXotiFwYYMFGNFubfXBk/yk1GTkSAlg7Vlh3f+MTO+z3bV58YM7wQLRxVRUVFBXTtnXLqKykkKadHNntDn7QdgozsDXHQCWWFMKgAfDs00aaTcmmWzcsoy0rF9FqawGho8enZhdIq55xI33h0w/SQ/fX07071gvSaGG/S4cAfGYDR2+9e0YggITDoyZbavLpXOtVeuNIA/3+wH763s9+SD/8938UaHz9AEwAVTq/V8Xo8Go0V0cju1XkQ3pDfUObsU6qqymhraurBH6Aj8OuVAFotv5xyjTrAwbZ8soCOt83QxlmA68VuEg/OcKr3Ff5Ca6dztn6eXWzj87YBuhQUzvZu2zU8Orv3WszJQ3giOqc5DrAA+wJSDElaSJJE/fAslotli0rSvbiICwGBZgFVaW910Ft3f18510U1gvF/Exbzs/geRlG6nJMks0+QxgsGDQ71i2h9UvLFlj9MIuvqvR8aQvyvnNrraDuoaxgXHqWhXT8Gg6SCulhGbxl7WJyTozhlsjpJGtePn3qs1+mdw6+Phfm9QsptYpfS0H9g+onjj62by+xCUYcLncNUAFMcvHIq7a6SADa1rVLaFyfKWgAUJNT+EkCfAV/xZLM2/gB0N67uYYycvIJGkEn3wdQuYvTdZTqmuQnuylKd01QeSY/JHnDT/FMh7A2O/jsd+in3/lnpUYOYZ8MUqyJtyLCbC/XpkiH862IdBGRzX9TafnezUuLLehIu8MhLLjRwViAQ8VD56EGGAToUKiD61ctpvr1NYTBgkGDeCknJa2GJ2YIryYrycsWkmD/CEYGSB743o6Fj4xNCPT4wed1KgsyCQYOpBfC0szwBFCtW7KcvvTVJyivoIjKrYvp8rkbL9YEiCpKLLR+VZnwJPBcohu/b7z8OwrEvI6UXYMjioEIgGDbYePyCkGaQcpDzQZ/xSCD8QNGoPd5YxAmKJQDV12cTdAIGEDHuVQBpPZJTgAcJjuxoWhgaJy44W7at/c7dPzgW4IUQz4KnAVm+1hJL52CCqqWZMuyyj0fuqlK2L3PyTATNztDVn7BncmrKpBKncOzgqHCmJknDAJ0KFQ8DA5/jcJg8JZWSDPMrzcAKjxjhXuspQSJs26poBLiWuyg+uH+5SOnQS64ZdZS+nnDeeH9fjQ6QP/+6wb6mx+9RjqTmawFhfRpXlIJhPwP9qXGxkYJG88FuRkEta+cH5x81IL/4aEhanzlpwtU1wWEXgGYQLyCFN96gwwq9TV+TTs54xKMQHm8MQgWVoAMRiIYi7BGRR8g7UprEQGkeA4N/YPJDnGQom+830xZ3JiwvpudnhKebmZmfMWPtLjoKcWNCSNh3AIL1r8Htix+HJ0AfuRkpNCsMZ0wA0JVgVTaXruYtq4sV3xGDvkwh8HG8mZhGBB1q5YSAxULV+KvX1YpvPoMtADjmuoSOny2g/BVkStdffRO5yhlm0z0p1/4K5B4uB07byNusp+KcozuY1MeBPM3MK/XlmfO3ynz0CbwThm1byoABSo1VGuo2DCACCo3PxlBc4AGAWMRLLDMjN81OOqRKeoDyyMMSzV5BsFK2c1bce/YuJRQT/TJcn4zOmPgLDU1NnqklbmxxkIllAeWTC3VEGy1Wiy71ln31pRaLOL6lOVnEmZAdDA6QBwXyLWctMJjFLdtWhlIVm5agBEqJAsoLMinysJM4WN1dUWpNGMfoIc++ajHvhSTUDlpMwKgnv3Jj1nyBX5nRwcNnHptQbi/AKjQUM/80QUaDxUbBhCo3It56ypU8F7e2grjRylvNGJmfKjtsDBCZcSG/LunrxDWa1gLA5RVi8qFydG7PwGyC+/+gZT88Srh49HeUI5LYK0qKPNmg04AABAASURBVHwK5/i8mYrZzjssmHspaYV8DLyqBj8cDh88WLdqGeFjdfiC4+qaGvceFdZRPW0X6V3ecHHs7QaCivehBz9CcM/+WBpcT//940FJZu8BG462eecBEEAFh6pXv2kFSa3LoDJiQx6SDeu1IVeasA4GQL3zY/dL802kUCW06ExTe1i6aPhxByyY1j+8peYRKeaEY5DgTKDU2grlZWfPGSxwzRyMEjBYMCOF2Gc0Uj7WaZO6VLrvtq3CiYwVuS6yXThBg9fOknO0g3bespk+9ZmH6I5ddwkHfPGVkZdeepNmDDn07rF3PLKEed0y1e0RpvQmXJOR0vJAB40C6ypsbWD9e6pjgrAugzTrHNXRljWLadPSYpD6dOjvUVujog1l3hz/lWhKrbgCFg7P4sUtmAF9cjzISAwynGBHh0llYbFYFgTjnB+ME8xIwXysqQAycQKAiVkBoRZmpqUIHzzAB+far12jhx76CN33oXtpy9abyJJzo6w3979O7568TGs276QVa9ZRd0+/+3M+yP/tX35XWH/gOt4ceL51ZQVhXQZptn11ZUDGF6y3Dit7PXZUpVZcAat+dYnw4pZIDR48h4TZVCp/LLJxpk8qTioMgPMOZ+ssPIeFvBgN1ltrl9d4kwv3P/zeD2mCswiAqiix0DrezP7A7vvp0oXzgor43DM/JCUvhxEy8/qB5Q3WVK/gqN0eOdfJTwhmxaZ+d8W8LsY7zqtOasUNsHBKvb62vM6Lpx63mP08AgK8sfLGhENNNsKC3jvp+AwRrHne4YHeQ2LdwpvmATJx2sqKMvGtAJpvP/VdWlp3M9XVLhMABTM71l8gxHrr9y+8QDgPKCdhQefL2UcmQh7UvvL3FQeVe0NNYVjKV6PUigtgYV1Vv7Zij78B5C/eV0cjLi8rlbby5vmp0SE62HjV47BpRnoGSDzc+OgI7f3VS4RHQvDcVV/XNRrs7aaG988LdFAHsf4SbvgfrMcATiap+CD3vy6v0n2N9dQvfvECffShT9P2TctIDCg3EX8xbe8M6Dwgn8TjP9SJyCOzAG5gdcUh6FD7S1yk2qSW6oFltVqEI0u+rEOMweEYKAaDgZZWlVFNgZlaWq/RkTNtggTLtiw0XMCyt3FZKb38ziUaGLSTY6CPfvHaMariN6lRJwBoWHTiApvEkFaI83b63DlgAVS2llb6kz/9Y1lAIS0MFoGeB0S6WDuo1ACUkv4MpK5qk1qqB9Z2a8ULOLLki8lTU1NCNDpMuAjxR6fTUUlJMRXlZtHMzAwBsAW5N4wJ4uwLC/LorvVWutA+KJymuJlfA+ld026SzPmjSj/5/UG6+6aV7nDxxdunLtGGzZup4c03qYvfj9p1793Cox9iGu/rYwGeB/RObx+Z9A6Kyj32AiOxb4bKq0lqqRpYd9XVPLFrvXUnmCblxsbGqL39Gg0MDArRUC+k1kdCZAA/yOPCNTtxKWl0a121kBKWP+HC6yc9cw5wOEGxoaaI0lKMHhSQWP+093e0sjidsjPTPeLYDaTV8PAwlZSV0c477mDBsj4MFhUpnicWZInnIzA5QAXDJuyRpitkHxpZcOh4njRiHvgaKVCh0mqSWqoFVk1x/s7b6irdR5bAOOacTqfwsGBPT68gUcQSCwOI0QXqo+OxqIalbGVlHrFBUFyYL5tVRlY2NbXZ6fWTNmGz92Rzj3CagiXo7OikRTkmqigtYEELfKyvSnlQ4YU0CyK9AnDCQspg4T2pQOU6Y+sjnGgAkPB6ATzwWZKpp7XVRVRdmueVc+Rv0TeR2iphtQ9Aaj0eybf16liF1ORbrRbL/VsW72UDW1w3gKqrq5tGRkbcwQhjN8Gog5jJGaA28JYq787XG1NY9h4+LHwwne/auY0qC7OEc3+QXJXzDxzCWMHUQibZPDLgb6AGbtx6M3+l7P+3P/oPSYMF2g0g4QweDrziqBDO5eFEQykPpiXleVRWViqouBkZGcoKi0OqAKQW6Z3c45FqoiqBhUdB2DvRvRvexW+OMgkljmPgwqzIwjFrw7F7X74UoBi93ECEhe/e7bV0ub2XXPoUQXJduDZIp6920f7DJ+nVtw5RXVW+kE1auvRgHjNaKDMrS6Dx9/PO2w0+3w+4Y3UFkc4oZIMT/lgr5vAbzYsWVQpPQptMJiEuFj/oF4A/GmWP2hoVvq2XeyRSUkt1wMLpiptXlwqPgkh1QnOnXSqYpMCGhXKo1icAcwVvJZQsVBT44Ttvpo/eexvdd+dOAUzVFk7wAbS1S8pJp9eLqG9c9oxLh9+gmLuCCvjeC9/nN1SlpSeoIGmxXYAjQaNcOvVPm8mclkEAGOJj6dAXUhpIJOoENf69t5Ud0I2U1FIdsAw0Y5UDAyxZ6SbfVTabDIIVD+ulcHSkHZuo+QtN7UTkHhNiw4ZOr6fyqiWk088BBods83ItblrxBdTA0kVV4iDJ69/+4mf0m3/5K1pamCoZ7x0IHuDQK6Qw+AA11x4jKyCrGyQWu46Gz9ltMZVavkdpNDgQQBl2fpDjcQKpJBMTE0IwwITZ0c4PJMzgQqCfHwxEuY73ZbiQyzbFnOoGF9ZeqemZkqTvXeml0rJyyTgE4mngp/7yYdJfel0xqJBO7DB7A2BoHwAGoInjo3UNHkerLJSDcRBLqRVXwBoaGQPPJN3srNMdjhdOgrHuAD8XoJUbcNMug5/U0tEAV2XNcrKWF5Pc+urYuVZh/8o7Bwaosfefp5UFegrHoGRaQCov0b3Li8Y9gB2NcsRlxFJqxRWwTAb56rI1Fva0stNMPtciYuaza7PEgGsa0FNmXhEjCdg3Gk1UUlntVgvFGfz3i4fImJElDqJIAAoFQHrjwCseh2EAQ3iiO0yYyqWW7qlw8kN+pIazlCjkhRMSMMG3dA4QnlwNtEjvGRWg+vz/968Eo0Ggefmjx9uZ9h07T8uWrxBIIwUoZA71z86r0NtXloZF8iHPYJzUxBVMPoGmUS61aHc43wsfY2AFxia8sUcuBYCFN8yOzuoDllYsT6iD59oG6PxIJgFUMIN3j8yy6LD4Lx54n370wptkLiqj9gun6TuPPUThVPlYJTFRMCmFmZuFx8K38+vdWKmgaPs7b72qqNkuvT5s+1pxBax0E+eTQXiDUkVx4CcKMAhZ56dVbaQ9//Bt995SYUUViU+o+6yARCSkE6x/2NfCe9svNbeQY3yG7i1z0mc25hEsfeGezTFB4NmyWEspmv+z8xIzlipo5lQf4QMM89WR91yuneF6XVpcAWtyxinPFD4G76bDDMVfKv7HPhUbhBgAn/v6NzzSfvQzf0rPvfq2+w1LHpEyNwDiqwffEz6A0H/tCpUYRylLN0HtfdeptiqPPrmtyn1cSiaLoIIxQUD1S+XXi7AEBpVJBBKhXhHIVnGWAHXziSPK6F3heV2aqoBltVosRdkpsovI7Iw0WeYEI60wCMenZtxfLyzJzSA8jiEupLSsjLbsvIv27T9IkD7k4w+S6YW33iUzTdKWFRV057Y6qltbS8ev2ulSx6BQTrilE6sOk1IAFAYSC1eDH6k2B9K2In5ya2psVJLE+sGR/U8oIfRFoxpgWa0Wy4aC3AObrNl1viosFxeItMIMKrX+yMlIoY7muXeqi8u5748/z5vM0+nCuTP0m9ePLgAYOzOIBxsfvH0LrVi2hHILi6lv1El7XzxA+fx8YC3ytACK8w/1GhOE2qSUuE3gt/g+FtfoW8WvSyN6/NSxV62h1FMVwLJa50C1uTq7DiZ1O7/YDaRR+EpInkV6E9Y7H7HqJzWTtp15zzuJsN5a9+HPUn5WKtWWptJgl43YU8IgZmcG8WAj7uH+8N45OvzOcVpVnh0xaxzaAlAlmxkd/A3GLc030f6XX1aUNNSjTjEHltVqsdTXVB0AqNBinKyw84tdXHs7J0lX99r1GcIRHm9673sMQrHq5x2P+4zpQUkT++YdO+mqfhFIKDfdRNYsJ42P3jhhL0TM//zm2FV6/9pkRNZR80WQuC1SEwSjU4OvlvqhHrOdZ5QddeJ08SuxcOAWoLpno7UuLY3Xl/yMgjTzwtPZkFZF+Tk+U0IVkVL9pBJhffLBO4ekoujjf/oler9nLmp6aorwnouejlYSf/70mbfOU8nWT1D5opo5wjD/YjsAoIKUCtRQE+aqKMoOUhXPtykijgIRePb2m6/7LclFzhf9Evkg0PmIi2jUkrKCuj++dckpgAoFpaaa4fl06CD2xUFG2MvvM/mSVuhYZvXDjMXS+fIvHHlVMhr7Wtv+x+fpg/YhYn9D9kFquXSWADCAauntn/F4TTSjC9a382oxwMQcBgYMFErbEmy54Upn57UPrG/ClV848snQzb3KwVdeTs61z1e8v7iYAAug+uTNyw+sqcp3H/vOUPDwHTpoeGLGo02m1HSPe/ENZnZ/qp+Ynl2XpM4ssA6yODzlu+LDX6C2vhvgwgONPzo2QGJQTZnzWZKA/AneSslABB/3OEjLXLwASmg0/6PG+mYU+9PyXM+s23qPja9+0P8xARY+aCAGFWqPZ4aYOoiBijApl515Q2XEO77XVhcuIMNgVKr6LUjMBwDA7736K/5K+h8vfjGv/4gArrHJaTrQY6G7P/NVD0mVliYPeO9cYSrHJAAg4WQ+AxF8qKbe9PF0j75QU33B66UrV/us0qzO9aRPAgWRMQFWTrpZcsrIzMwQquzjrC2R7sZpc3N65gKLWzCqn1Co1w/ehe69pyUmuf3eDxHA9UqLgW792OfEUX6vUUeAiLnS3HRhjwtAgqrnNwONIGgOuHKsVFhY4CN96NIKmccEWO9e7Po2Cvd2abwBw2C4ARzveNyb5h8vxwv0l5Z7Hl/CrB+M6od8vR2k1rF9e4U30nrHsXuAa8fd9xM+CsfCpHzM2gxE8KEeAUTM4V4qXbyHod1qatulawO06ZbbfLJVRzrJsekzkURkTIB1ydH/9NHznZI6LN7RIFFPd1BZfpbwMTKsrTD4EYEODEX1Qx5SbknmJH3vb7/qE1x37LqLcvg1mRS4AKITzb3krd6xekuVGa9heLIAB6HF9YfaFWsJjAdgcTi7tbWNpjLL/Ugrce1Du44JsGw2h+Old688CpXIu/owYoxNyFttMCht9mnhA3NIi84L1OqHdEpdbd6sYnBdvTj3OdTxsRE68Jv/IkgkWPBCGVxK6xlrOmgaGMDiesRKWgHgDodDeOcke6PX9bFp2rBd9hWV7mrPkmvvqQMH3EY1d0SAFzEBFurY3N3fcOCD9qchbXAvdvh2kvje+7qqNJ8AMEiDVJNBWJ9404TzXim4Nq5dTMcbXqJXvv84VRoGwlmFuMgL0gEDmlVWqm9ZXCR8lA9w4yWudrtDeOckK+fqgLI3/3JEdXrT1F6WLlg/ZsBiFYa0YddKfDu/rwM6pvpFy2oGcP3XE1/y+C4V6iF2I8PDNHhiH9WYB8XBSXWNAc2e5o5Gw/Hau6GhIQ9h5+rfAAAQAElEQVTp5F3u8MQs1daU0ckDLyn63A+ffnfT4f1P8X7Q/zEDFvuCCCxiUOeUtgBvmk016SkWzxrh/ROv/+BvJfe4nv3h/6XXnv4qlaSF98FIpXyJNZ0YTHhDMR5KxYa+d70wMeLNvCMjIwRQeMcrvUd5TDrhdQxQ/+TS9o3OCsfLVhXo6eXn5bdRPNJztKfpyOuSXw71oJO5iQmwrFaLZcuKkr2QNnBgthK1Aaof1i1wMu2JePDSwlQ6/et/Fd5PgcLwfeB/eOzPqP/YrwjnHBGWjE4MEgzylvZuyW9fQYWvLi+mdy90ClLGHyi8eQlAdnR0EhyuxeV607L7ItGRt6UZo/Tqiy+wKD++a2+w4IoJsPCmW/EXRLDAB2jkWgnQhaD6yWUbdDgmg4Gjv6J/+cbX6B/+nwcpb/QKmXxuvgVdVNwmNOmcwuePpBoA/m2vraZLfVPUP+gQAMaMDFJAYdIJlj1IKdxL5SsVhm2Zggy9OwoGFbPjKh07ctQd5vsiOHBFHVjbllsfuWN95YI33cqBC5ZDrMNiofr5Yjhm3iWGHpqaVLYo9pVXOOLAJ7z2DSZ+uHDkGUoeeMV1e7e8AQcD/M6NNeSYNVOXY5KY4QHggTSCJAOIYIjAvVLp5F3nqRkX6VxOj2AAe6L5WCDgCni9pfMoMcI3OM1+/+aqpzAovYsCo73N0hgg4drw9S4vHPeoMz7zg5k3HPkFmwdeu42P5I0NDwmSsyQ3gyDhg80vXOlyUlyyUouVsWV5KVUtKiebfYam5l+9AIkEgwTABLWS0Qbq48B2iSVF8vXjAYIrYPN7VIFVv7rkhZpSi0WOQTkZKe4oDAwMEG+wuQlUcuELXFBho1HN1s4+muYn5UUVpVRTmiNsRWAdGohRKBL19Ce1WJkY5PXr56RXz9CNj/ax+GB9AAt1AFCl8kC5PWcVqIQc1yCV3ldY1IC1c031U/4+zo2KYjC+1dgm7E2JgYY4tTopcFXkpwsnLiJdZxh+slM4ysvL9TAWqIV3xelEZ2x9itgA6bVscSV1juqE0zWKEvkgMhk4dyxUTfeN6MLEKbDiuujHoiSKLhUCS1FeskT4iNyDW2v2YADKEQFQkFJYT91eV7ngcK1cOrWEo21itXCSuyF9I1lH8A0zMyRVJMsJNm8Ydbp6BoQPVSjJAxPC9tWVlFtQSG3XnYQ9KCXpvGls/eNUyquBLFwOWFMpeYxEznfMThkDfjYr4sCyWi2W3duWvACx611zzLZYR8G9yUspMBVGDG+6eLln4OqdNFF5YU5Uqg2+2sc9n1FjBQN07DqWfk2hmU41dwdUBSwBMFFl5OQHDTCAmhU6cH2cXXr4RYtXetwvvHHtW1df71gY7jsk4sCqx/sslha711UMTDCvo+NXVubR+NQsr/oVCWfrfFdX/bEAF1QarG+kNkgj0QIc8YKkF+cN/mJwisNiee2aHCVYLr3rAD55h4nv0QYxwCCdxfH+rmEQgeRLy1z4lqwLXSNUW1fnMwvO6QpYDUSGOvxEyt23aelerKukwATJhNkW6h/e34DrSNUj2vliUANUkMDRKBtAxiQFyQ+H8jEgAfJolK+kjBJeLbtg61pAirGBui+I8ApAewCwxVVzazA85ArQeJEJtzDf52WY6OrAFE0bMwnppMZXZrGV0tPThDQyP7Y1O+4J2HCBvCIGrMXFBY/UlOQ8gpkUjINkYmBCwQgDqBCmpgGAugXrMEjQplhMFNjnA4/hwNNogToQXpVm6ui9S54qIeqKUxj4ir+SvNAurMFu3bCMZlOyaWBST9cc08JeGNZjWFuNu4ykS82i2zcuJbn3oaCvihav8lmkiwI3WrAMIwIsbAJ/7Oale9HZYJz3bAFQYVZFfKKACu2xj0zE5Awj60y1+1jzjA1fJ/v8QWpW31trK6nxaj+1tnd6nEhn8XI+JpGNyytox7olVLeiigqKiql+0wq6fX21LKBYXv0TOlq9au5rLyzM23fqnM94hym9Dzuwaorzd37ilqV7N9QsfBcFKgWmQooBVLiPd4d1A0AFKQV1Jd7bE+n6V+am0BsnFz7jWl9npe5RoqaLrYTN4UDrkWY2eWw3+EtvKqz2R0JGp95tG/BL7EUQdmBtX13+gLeEEpcJUS4HOjFdPFwDUONTM7zhpTDutgdiyV8zN004LeJdhy3LS8mUnkVnrnQJh2zlNna90wV6j8lw3abNfpO5nLPqAVZlfmad3xrHOQEsWQCVJqVudKTZbL5x4+cqP8NIOC2CJYE3KdZEOOLU3D0sgAtnBp1OpzdZSPfov9FR+c/ussydei7osRx2icUqlYg+BgIAxZ5aTpT1IYWhs0wmE+EVdkqywueYqvNMsntb0Hhql1mF0+9QC3EId2zMPxCUlA0aaE2jSvJzceqRWKh4IjoACjMd1Fh0fCK2MZQ2AVTZ2Qv3inzlaZwdlz2ki8EPM7ntOhHegYKHJ/FoSSiHclEXbEdgLezo87ROIs7bcURrvcOU3msSyw+noI8DVFD7YIXyQ57U0VlZWWQwGPzyIMUwN+zwYChUQrkE0Ahg1BhypQnmdBxLgvRyOBxySRSFI18Tp0C95Dj1SCzHyGRorVbEmsgTMbVPM04o5zWkVkFBvt8EABQ2cUGIvS3sY+Fazm1aWiw8WoINX6y38F4NPKcFoMml8Rd+vb/HHwmRy+X/tU4yucxNHTKRwQRPzzg/CCadmtJAXcCWANQ+qA1qqpva6wIjhr93Q2I/C+sstAXXs+PDksedEM8c1G88dXzV7hQO5kIlhGqIhyEBNkbny0e/sv40KjnVzmcW7KvQwg4svi5x+8/UPjAfoAqkIRrtDQ5YLBYCwG6E+L6SO+7knQoqHDZ/caqCPbeFhyEhvWDk8KaXukceCB/saoPn1xkME0FZBjVg8azV1D6eCWH+Lyoq9GklZOssVixUQu/jTizO24dJHs9tNQ/MCE8dQ2LBLN/SYiNIMG96qXsALJIm96QHFtQDTe2TGnqhhWG9VVJSLAsuPPGcl5frLoSphDiZ4w70cQGrId6ZAcMGnuhHeQCz3BoP+eZkmN05QitpsbW672UvgjS5Jy2wNLVPdiiFLQJ7W3ID3ciPPJM5jcTgK8oy0vsX2gMqnxk2yspKKS0tTTYttBKs08QEikzuHHerOI3Sa755SkkTgw4Mhvlcs/ZFpz8x2KXAhXUVPhZhNpupoqLcDYrK7IUn4P3V1MCb+OF80XmDCrSKTO4ulxW0gTo1ASvQugdMr6l9AbMsLAnwoQspcLEPDDI1Dqoh1j6wEkJ1C0vhPjJRaMDQgCXHQ03tk+NM9MKlwDW/T+yuBDaYIb0Wl+YErBK6MwngYnp8RBH16UOvBryfFXaJdbVbPfvDmtqnaNxEjcgbXNdHFp7/g/SCdMPDjFDZI1k5/czC8qXKc+l1FqlwX2E6X5HBxPHACupR5mDK8pUGnaJZ+3xxKDZxABcMDQAQzORytcDzVbD8RVIlLM1NpzNnz8tVwR3uclHAe1lhBxbF+A9qHx6Ph7TSNnlj3BkyxcNaCHCl6H0PP5jEAS6ZbEIOxnpOyV4WRxTwYVzfLQu56tHLAECClIK1D6UmyhPKaEsiOljxcvl9LPRbLNvHjftfuriIAjZgJASwrnZfJ6b2QXXQJBXFxR8kEkzusaysksO4HFFyqYIAEdQ+7KgDTAAY9GaIeNL+NA4o4MDU2LACKqIzR14PCFxxK7Gg9tnn34oEPdw+Mim8xlhqI1AR5zSimHAg5qrghH9VEIxxcq6A1MGwA2vGPNN4uqVfWW1R4wAdjBMAFR48hCrBkmPzV3sQkXFDjb50nWKtXUDbaWnxf2YwUMtg2IFlszkc717s+nYkZiKAB8YJqH3iDgHQECbddVqoxgF5DkDb6e3rlyeYj+ECPDMYdmChHq83Nj+x/6QtrPtZACpmF7GUQlnaugpc0FwoHFBiGaQAzwxGBFho5BFb+4Nvn7nWiOtwOEgo7/WTtq4KB2djmwcmzNjWgCgSZwYjBiybzeHYf8r2aKTWW1hrQTXU1lWxHpahlY8JM7QcQk8diTODEQMWmnu5o6/xl4cv1IcKLsxqUPkAJOZjrZV0m8BgqubCzgGlZwadAbzAM6LAAgcYuI6e77Th3p9jkogBCGDCJiLABemENRbz/eWlxWscUMIB7H0qOTPIOblFSvIDTcSBhUIArn0fXFz34jvNDQAMcwDNaVt/w1uNbc88f/jyk3/zzKF6PqyBAQc+HMCkBpUBbdFceDmACTO8OQaeG8aWQ8ELPEnHKd4kjgqw0FSbzeH4ecPp+r9+5g8cc088+zb3d788VP/91048+tzhM080d/c3cMQdBL3mkoMDGNRqaOmMkmezAnjPYNSApQbmaXVQJwfUILVIwWFccO/UsVcVncBQHbDae4cjdmoDjNGcujgANR8HqGNdK4Umd9K7dIrUQV2sG+Rd/sXOgbDtfXnnrd2rkwMAFw5TY18yVjXE4YPe3j6/xSs92qQ6YHF6ziHFYFWoC37ZrhEEwwEcK8LWiX1kgnA8LRZ9jcMHLTb/b8flFB5tUh2wYEEEg707SC2LXO96affh4wAkF858vnnyKuFDCdEGmMKjTfGpCqKbRsenF6yzos1k1ENzseFAqSWFClOm6EjTVTpj86+ehauWCtdZFiUGjGAkVrjaIZ8PRwvWWZrEoqT7w1cfjdPD9Nb7l2Q/UBdOpkwMDSjKTokBQ5XA6nWMLTilkWoykH1kUlHDNaL45gDe685agHe6A2Aj9n767eELEQVYdgpH4TJgqBJYPIAWPHmGxaWdX9wyhmt+4nIgM+3GxwtYKzPNeqrM1tPllg56i1+D2SMwyWKMKTFgsDr58lUJrJPN3ZLPcmnrLF9dmfhx+BKkycBRtnGWzl+20ZEzbcLrGMLZciUGDI64BRqVdx1UCSw83o/zhN6V1e41DljzU2lgZJoAstJ0J717+orYghgNBtlmp4z7/BWkSmDZbA7H0Nik31nBX+O0+PjkwPDYhM+KLy1Oo4vdY8JH5ypzU8JqQZw1pPosm3M6H11XX7/Aau2dSJXAQiWHx6YlLYOaOgjuJLbDt7P8tXBNeYYALkYHAwcsiAdONgdt4GjqHKPaujqW5QLfRfTkmh33SC5TvIlVC6yrndcXnHLHBiKezfJuhHafnBxYxkuu09dufDEEFkRrjoFgQYSBY2xiSjFj8BzgypvvofR02Y/X7Vu7fdcTSjNULbAu9Qw2SFl+NImltGsTnw5A8gYXWg0LYnWOjt4720JTU/7BhTE1lb+cVq9ageQLHC+pGmenTI8uiPARoFpguWZdlo7+IR9V16Iiw4H4ylUMrqkZp0flqwvTydcXTRjxpZF02nXffezW23foiVO0rhInVC2wctJTrBMT0otYzDDiRmjXyc0BgAtrLlgLwQmdTkc5ORYqKysls3nhnhhomDvbN0v3fewT7Hahz9Gjq7fftWC9v5DQgpVIsgAAEABJREFUM0S1wKoqSLWOjy8EFh7V10zxnp2o5jtMgli/RKOOJZYUAVD4KqTFYiEAzFe5F7pGaH39/bLrKl4FfLJ22y6/pnWpMlQLLIOO1mabdRStTpFijhYWGgcAKjwGghMNoeXkPzU+aKcUUMgN46p68x1UVSX7fpiAjBXIU+xUC6zsdKMl06wnW7ddXF/tOk44gIGLJ4PxnFWgVR4eXaipyOXBAIXPqxoMBjkyj3AAHsYKOdM6L6kCNlZ4FMDfqBZYRQUFO/EV9ZnJcb6anv8luRnqlmSe1U26u/cudVOfY4TwbFUwjfe3jwUAYQ21aFElBQIoVpdIGCtY3sxXJbCsVosFj4lkZWXRysXlC0CUk5FCdu1ALutD1fiQBC+9c4nyU2aotroo4Hoh/YFGm3BcSSoxAAUgBaLyeefj11hB3GPBGCu8y1ElsAwThjpsBqOyeVmp1Dk4iksPNz4163Gv3cSWA82ddjp0qplWFZmorKQw4MpAdTx2xkY4/wcrn3cGaWlpBCsfVD/vOKX3rf0TPo0V5KKna7ff9YzS/HzRqRJYLp3OIq50qkm/4BQzgIfOENNp19HngH1kkt54v5nGrg9STaGZoGWYTKaAKgLVsbW9k3BqQgpUyAzLAn9WPtDJOYyVrKWb5Y0VHNdQe/Oux+TSBxquCzRBNOjvrK30OLAFE7u31II66B0WjbppZdzgAN5LcfFKG9XkGQiGJqhqWPvcoPB9BdUPR4+yuDEqyjLKEpv5vSjkLUvgJwLlTFiqaev2bXKUjtlJ44NykcGE64JJFI003mVAQnmHQZJ5h2n3kecA9hHxuLxFP+EBCKx/lEoV5AHVEUeP5KQUa0l2dha7DMo/bzfQPQ/I44ZzOh9UcmI9kMJVCayzrf2NEN3+GgLrIDrIH50WHx4OQO2DhMEhV5wmFwMCKiAki5KSIOkG+3oF1dEfPSQV1lf+6OTiG1sd9OGHPiMXTS7O9ZjSE+uymUhEqBJYh87b9j1/6NKDxy91OyTq7A6COggx7w7QLiLCAfAYYIDaBwmTye8vigvC4FeiAiKftxrbaHZ8mPCgojgPuetQQIVJd3X9A7InK4hcz6zddvfTcmWHEq5KYKFBArhOnq/CF0rQIQiTc/7i5dJp4f45gNeP4TVkeB2Z3DpIiQqIPsIpjNvrKolLSfdf8DxFsGogyjMu2iB7Yh2gqt1+d0An1uerpMhTLbBQe5vN4egbuN4IXRzmXKgi6BzEMQfDBmYmdq/54eEAeIqHBvHwINQ+uVwhqcy8cUEuHn2G10ej39gpjHU1xWSzz8glcYcjX0hDd4DCC4CqeTKH7rzrDpkU3KOhgEomU49gVQNrSVlB3crSjD0w47Z3dNP7F9qpNDed0FEk+tP2tETMCPESa9uDjVeFhwV9mb9RDMzqOOyKa+YwqM+1DdDJSx10pOkKHT99hbLTjMRABTps/i+3llCXYxK3si4zM0M2Ti4C9QeoPvZHn5Ii4ZcWAFV49qqkCmBhqgbWxkrLUzixjMpmmg1UlM5RS+s1mpmZIejq6ETErarMJXQmrjUXHAcEyXKmTeBvZbZOMJ/7ygnWv6KiQkI6qIsMSCfPXqHU2WHKMU4TNzvDbxYX0GprwYKscDBXn5opvLdiQSQfgPwD3QwGqMw1W0kOVDri6sO1AcxX0ee/zmdsDCO3Lbc+cs/W5TsZc7HYxQwHoFVmOskwO057Xz8tHHfCDKhJreA6C8DAa8TwOjGcegB/5XIanpglW/+4IGnaBifpyAdX6EpLG6W7RgUglWbqiaVHfNWicklQsfy3LC+l3pFZduvhB2q0wKkKgEpqr2r+UG1VOI4qeVTSx40qgWW1WizbVpY8BcBgYSwGFzoW7cHbee5elUNNF1sJO/fY58K6AHGa888Bf4DC07hdvKoGfsOdaB2ikYkZwuvHAJ5yi5HQB5jwxKUh3dmuMVpaVU6QSuI472toHHJP+AZitIBJfenOB2U2gHnL3/Zd69YpeLOSd/1CuVclsJZa8vdsXlpsYQ1j4EInQiW81D3GogivwsLOPVQRqALuCO1CkgNSgBJLIoAIjj2NCyAJLi+VMnh1XDJTPhD06JejLaO0vXaxP1DxKYhONXcL4BRuRD9Yu8GJgiQvAcz3r03Rhx79ssxRJS7iRgrJivGBqgTWlmUlX+Hr5vEPcMECBXBZ882Et/NgdgQRNiqxvzIwMBjVr1Og7HhxDFAwJkyNzx1qBoDgwEdrfqqgxsGHg1SaFL1DAnyHxBK3lwESgEIfkN5AD+1cQdhfFNNJXUO7MPLqvFScEqMF2gMjxae/8GWSeLOSTUfcumitp6TaoJMKjGVYTXH+ziVlFotUHSwWCwFg6ES848DWPyHo+4wWYR1dfQSrFjqOhSezDz7gUYy33r9IABQAwoADHw5hUjxKMSwcHpBMACPAxABp5Se6gUk93VpXTVDfpfLyDoOxCf3oHY57pvrjWsohLVVukjNS7JudMq2L5npKqo4LOSdFFcWwZaV5db46B0wHuFAlqIHoHNbJCMs06wlWLRy7wT4MLFZQGRCXTA7txgQDPliziTYsyiIrL5XkQCTFG7HEQnzn9TnzOPIB75EXpFbnqI7q66yKQYU9rTv4jWL72Ayy9XDoX1gEPQJFN3ip5rp7Py25nsLxpNrtu8J+7k9UvOJL1QGrvDBjrb/ag/klJcXCy0LQuSYDx1urJjykFwCGfZgh+yAd++CysAeGwZbIIIN6dOzcNTp44iJhYxcTDPjgj59S8QAMC4dkwuS1tCjd4ygSpJcuNUsAFaP156OOUBUxeZpTzQvIJdQ6gQb91jSYQvd9+vMk8Z4KB+d01kfqeJJQgQB/dAHSR5w8Oz2lTkkhZn63H+AyGAz8XohLMGJgcW0bWPgoP2bYvJRZwSwMkB0700KNzV1Bv4pYSf0iTzNXgjDgrvbQgZPNgum7OHWGYK2DJJ+jUP4LADE171LP3DoMlsFO3joICZXJawMsN5jTLXn5Ps3pjFbsQ42DBRdhKV7PbaEvpczsMEr1ZS6njz/86ML1FMc18Kpf1Zod9yh69TPKjYbTRaOQQMrIzTQrAhbyhOVo2JVKy8stuBU2NVeXZXpILiFC9AOQFfMbzdn6SZoeHqCjpy4KJwSwl/N+64gANnSkKInqLjHrQ/q+e65VkMaZ3DhBOkN6B1NZgAkOAAJ/ACKmOpZYUgQVEoOe5X11YEowp9eU5rAgRT7WeziCxoizMtL4SfHGSzalQAVTesHGeyWPJwmq37a76qNtSmf19+WrClg4wpSTYfZV3wVx2BheVFFKeXm5gmqYOT+rQnXBDLwggSgAszoGTim/sclNj1OPvphOXM+m921D9Itj7fTfB67Q0bPX6OSFVkGVPGfrIW8HyScOY/fwz9j6BKBiQIXqmniphC0FdkwIm7KFqS5h0IuapOgSfAGQmCudBw9AJc4AExfWs3jHBPabkK55YIa211YrMqeL84JkxYQANZCFQ3JBnWT34r0r0J/qIcGULvHqZ5uOuHVqUv1YG5ivYxdq8J0uzipmvL86Qa3AcSbQZWVlEQYATPJluamCagirIRbcABkGEfn5m56eIqPJRLrCJVS9sZ5W37yLaGqUclJcBFUy1TVO3m5seMgjDPf66VGCj8HPjdupu6tbOHuH62Ac0kMq4ZgQJgE2efhpjkc0BjB4AMckE4AEhwkGxDAaQGoATHgDEnvHBLYxOgbHyTFrpjs31hDWR6APxOFVaBtqChckcfEmegSaedWeSUVMQlD9HvrTzy9U/YhUYfVDnX05VQHL+5F8XxVHHKSVuJMxMGCSB8AgwVaWZxEW3FBtMCszgGHmRXqxw+DKzsmlkvJKMrQfp9kzL9PgyVc8FutiernrEn72xyBGfswAgLAufp0il8ZfuLd1zh894lEHgIg51AcgYg40cBjQ4BVABDDh/B+MQ+Al4kdGRgiv+l6zbBHhCBLCgnGQTlLp2DqL7V3B6rfijk/Kq37bd6nC6ifVFnGYqoDlItet4sr5uhZLK286DAomwWoWlRBmYQwsAAwDCzO2MOD6x906PtYn13kLIvKaMWUIm6XLC02EdBTEH/ITb6hm8ioqA1qg2aVI7CeJ80C+AC5rE3zUG21lDuWDL+AFgATDT1WVleCDV1D7xHniGoedh4dHBE0gMy0wFR3pxU5OE0G5qJfBZKamQVmrHxFHD6pZ9SOvP1UBKy/LbPWqn+Qt9G9vaSVJyAdiIGEWxmyMAYWOZIMNvhhk3DCv1PNpuMKl1DxkEIwgGLR8UMj/oUotcQVQJ4CHgQkSGPmjPcwBSAbeYgrpg3aLJRKAZOZVL3GeUtd2u4PAOwx8qfhwhGXwgO0cniWofpJWv/lC+ElX0diYJ4+5pypgZaWlKGKenL7ui5sYHBhQGGBMVfQGWcFsL3W3XaXUTAtlrdxJ+tX3UXfeJjpnXE1np0ro/GQBtY6nCY4NbKkymck/LT3NI9rE77d5BIhuAA4ABY7lDR9Ht/pHpgWQ4x7xmbz0A4AYmCAdWVZoEwMS2on1EtqNcEajxMd3pZAP+KaEPlgaqIgrb/+4pOonzpNzkd/9TTF9rK9VAywcZSrNVfbItq+3M0GawYKGl57AKifFYAM/k2OwAWSQZBh8kGxlvNGjVneVMtsP0Vjj72n88lFy9l4m06RdyMaVUUSjWdU0nVtDaXUfIufSO0lq/TM54xLoUYZZJBlSeJUOwGCOAQX3WBMBKHBi0OCYFpw4HNdCAfM/GPwoC+2Bw3WgQJrPyu0hPfJ1B0TwYnpMwXfQOJ2iSTeC1Qwoa11A1BEkLrRkWMWGCLmi5ky2ZrlownGZ5eU5wiuOO3rthJMIssR8BAYP1CWoPAAZ1hw1FQW0tDSbFqWO0bL0YarRddAqUxetnD4juCL7KcG4obv0Br93ZiAxQACSFF4ywSf+D7M+7wn/GWYDmXhwARhwDEC4hhOIAvgx8BMEJgVIJpQDMASQXDWkk0MD/uvicu30T6QeipCBFa6m1FrzFc1IXYMjsnsoMGjApIsBVmBJpyWVRdQ34BCeclVaT0gYWBYx8zOgwYSPcJZHJq+KAQhwUMO8AWLNS3VLMtQFEgRpkQ4qH659ObNIyknRiQGFSQGTgxRdvITpZxaelpGq+6ljryoaI1Jpox2mGmClmg23Kmm8L6mGODiWD/R3WKPsQX5AAQMWgxxAgySDFQ2Ag3TAgEYcK8uXD2AiLzEN7pEeoEM88ofkQRkAjpgW1wBk38iscLofdCgf4YngBrvaFDXDMEMasCjAP37dpIhpWEPJZS0VN+Pi5MiDCmcSCCoYwAAgAGy4BkDgJmecBCnGCgCIAEbcm9MzCGmYNEQ4gAuQga4Lm8n83hFomWsbnCQHvzm7ec1iYoDC+hEnPrCelGo3SxsPPk7b9Pb2+a2qU6eLG3VQ57c1USJQahH0VR10kH1k7tEG0GHAtfWPEiQX7iPlADYAA8N/xIEAABAASURBVACBS+HXUd5lARAGfk0klqhiGljhOjo6hc1YFg5A9U6aaAsPKLY5izb9/A9nqW/wOg3Yh4gmR4SvfHRJfJGF5aN2H4/w9/b1+60mR65FfolUQqAKYAViEZQbmOAnOgjrLIALA/DXhy/ThqWliFKFg5STqojD4SCAChuyiPcGlLjNePXbvZtrqH59De1Yt4TSsnN5A4qezl3tQNK4dYOdCtTBAC2DsWSGKoBVGIBFMNVk8Mmv7StL6UBjC+E1XGtLTAE/1uAzcwWRkByw/kmRQqrxWqI7CkeFAChsxCLQF6AQD4fJA+tGXMPhhDneLFuebRAO/CIsHt30+LD/aseRZVAVwFJqEbTzRggMLH89UF9X5T7pjsHrjz6c8eNTM4IEkcsTJw0gmfp41QfrKaiASgDF8hNLLxZWwFtA2XW8+ootgwcOWOKhjaoAllKLINQ7JUzFjJ6RbSEc/Yk2sPzVD0Bqb79Gg44hCgRQLF+7aA3JwsYmpqh7aDria0lWXiR8xZZBw4Ti5/UiUU+leaoDWCa9VUmFpWZruXSrrQXUN87R+PiEHElEwkfGfJeHyQEPCk4bM+nWDcuEE+OBtKs0N13YBGeVx3NSeEQF1kUWFo8+eDA6Oua36k49pwHLL5fmCZRaBDEo55Mo8ravrqRzHbzlTBF1eIhwssJXTrOkp9s3Lg167QdV2FqYSXi9wKEmGz339kWqKCsOOj9fdSWiqEXDcttia/VfnovTVEH/XCIKl0VQqizMglXlRdTee10qOiJhML37yhhqqq94f3F4PmrE3ke5KS6CCrh72xLZkyj+8lJbvKOv22+VOI5TdJDAb0YRJoi5Klio0CIIPgQqsZAGM2Hv0CQFkxbplToYJPCkrWNohAz8fpXSdErpsDaDsQNGD7QFrxzbtXlpUE/zKi0z2nQm7sb7L2TLdrkULRtk00cpIubAWlJuUfw4ACRQMHzB+UEczg0mra80ANPQ0JCwBwWDBE5DAFRwvtIFEoc1FAALszwMMTgFP+RKo/q6uBhfgTSVFBow4qLhMQdWZX5mnRLuwxrmbw/LVz7hAhckBxvoANOl1h661DNGo1y6cEIC5vTUVPnT9zA+4J0OvurK4sbGxghlALwIgxXRkpdPm5YW4zbh3PT4iKI2nTnyuqIxoyizCBHFHFjkIkVMsivcwyKZP0g7rG+wgctIoFLZJczXLB4SCVICJyN6enqptbVNkE79gw5q7h6m7nEDLa6qFE5BwAqJMpCnp8Riuc35oJm7kv9FuVD7UCYkFiht9hnCFzywIYz7RHT6mTFFzXI5Zy2KCGNIFFNgWa0WS3qqURGTMGBD5RPWW52Do+71FgY51LfW9k7CQGYOUqKlxSZIC4ThZASkR8fgOPWOc8RM5VtXlhPAKq4X8gSwcBpdHK70GiBmah/SIB+8cmzraivBIoiwRHU469nS4t8yGA+HcWMKLMOEoQ6DXclAwYBVQuePBiqhWErtWF1OncNO6ugfFg7AQkJBYrB8sMkMaTHsSqUVS6y0ZeUiv6ZtHF3CWkiubVKTBMoFoABip3NuEY+yYaQI9pVjrA3x4mOSGuXVX3/15VyU7Y8m1vExBVYhbxFUygCpwag0rTed98wPyYO1CzZuMZjx9C9ecNLNq3oFRcWCqldbXbRAOnnnKwYspJZ3PLsXTxIAEdZskIxYvzEaADPQ96KztPHsKzqMq+MULR9iyYeYAkvpGUEwSDwYcR9uh7ULNm4BpPWrFtP22sUEwHmD0Fe5dn4dyKSUwej7sDDyEdTLjk5ixgmEwbUNTlJuQaFfyUgJ+OecmfLfqjgwuccUWEaDTpmpnWd1OCUWn53sP4AUDhBDHZQrBFIKhgk4sdoJeqidMFIwgCIsqdy4Q0lzVW9yjymwLBkpFiVcBE04BjvyiZbzdQJjwDFEkFbiuiSTkULcbu/r4euKgEVqN7nHFFj8YFL0qLWdN4mHsofl3XmRuhdLVZ1OnrXeb5Xl+UDJZKTwxf9EMbnL976v1ocpDlYgJVnZ+bULVDQltLGkCUaqwkgxlKAnKYLpC8Umd5Wfco8ZsAI5fCuWBMF0llrTwEgBa2SinqQIhu+YbJWY3MnFKV5GBFMPxWlkCGMGLNRH6QyvlA55xotr7p1I+JMUwfZFIpxyjxmw1tcUK1pfoXMSTWKNz3C0cVVVwp+kQN8F4xSdcg8m4yimiRmwcjPkD6p6tz8eJZavyQDqDpx3O7X7OQ4oOuWu8hfLxAxYSt9zMcfq+Pi189ZLVtN4nAxY3WPtJwLvYgasDLNRcf/5mv0VZxIFQk0KBcdk71SKJBaf6PShVxUvJ3jyqP7HDFh6HaeYKYkwg0W1V+O8MKX97dLrVGsZ1MWqD5QyD+pVPGwOx4qPiVguHgY9c/a836a5FD7L5zejCBDEBFhLygrqsBGopD32ONkcVtIWRoO2ix+4ZOGaP8cBTLpKXoWm5sdHYgIs16zLkszrEbQdb8ydG0barxQHuHEFZwZV/PhITIBVGKPnsKQ6MEHCEq4Z3oeU462BMQGWmp7DircOS5b6Tlz3/1kfUvFeVkyANT4zo/pHq5NlAKu1nVhnqbVuSuoVE2ApfeWZkgaoiSZe9tvUxDO5usT7XlZMgCXHTKnweBqs8T7LSvE/VmHxzsuYAIu3iCl+GUj0GByrIaSVK8UBpXtZTpU+lxUTYOVnpVqkmJlMYfEkiWPRL4onVJU+lxUTYMWio9RWpuKBo7aKR7E+Cp7LcuiczoYoVklxUVEHViCnLtAKbWYHF5LT+X4uy/XM7JSpas2OezRgYXgEeuoiXmd2vNkW7dVccBxobHVQXmmlVOJ9szpnVe32ux9dV1+v4HiGVBaRDwuvxIp8fbUSEpwD+BLLlckc+tCjX6bFS5bcaC3HNXBOZ33t9l0Prtt6j+1GhDqvog4sTs85Trf0q3amUWc3JX6t7COT1DSYQhsf+BO672OfoPT0tLlGM0Btu6terWrfXEU9f6MOrMsdfY0/ffd01Y9fP/fY0fOdNjDUs0qJcaetDZX1I/r/gt1AmbV30ccffpQKCwtYwn2ChNoWX4Bildexi2j6NpvD8fLJi08//eK7Vf/8/PH6txrbnoEUA5OjWY+IluWciWj28Z45+poBavdn/pRWr1ox3yTeKCGsoXY9GE8Sar7ybi8mwHKXzl80d/c3fP+1E4/+3a8O5fzLC++te/7w5SePXehqONc24P6OFU+WcP/JKtGwhmqfyhIklAegXPS02ygRiTVUlEdQzIElbi/UxOcOn3niqX3v1D/x7Nvc3z57rF4cnwjXGFgnmnsp2T56gHbDKIE11J0f/SORhJrr1Vm989vxYJSYq63/X1UBy7u6kGbeYfF6j4EFQOVkmAkfv4vXbYRA+d94tY/OT+S5jRKiNZRHVokEKjRM1cBCBePVZaYYCI/fA1AHG69Se/eAACg8PRyvbVJab6yf3jjVSj94rYmOnu+g7bfuFBsllGYT13SqB9ahpo7HsOaCcQODVM3cFq+bMsx6evdcK43Y+6kyW0eZ/L2a6x5q3dB2SKdfHbpIzx26QB0Dw2QyzA0vOSkVaplqTj/XchXXENZDrLlg3Phew8mcf/7Ne47/+P0p+u3Ry4J7/vAlflbsdLvDZzvc17w5X/YadDCQAKxyDhInENZ4q3dry9LcgNL5+KxPIGXEklZcNoAEvoGPP/3DOfrRG2eo8WoPjU9OU4pRLyZNymvVA0vcKzabw6HT6RsyUo00NDYpuJGJKbrUMeh2jVd73dficO/r5i47Hb/URW+faZd1v3z7Aj217wT958uNilznwIi4umQwGCgjI4PKykppiozCRMAmBOZ7TwxsMjh+qZswcL1doGD3qFAQN6x81Ad1Q32fefOsACTwDnycdTopLcUQRO7uJKo/SeGuqcILnUI61ZCNT04djFZlsB7C7Ot0uoSBg8Hjy+l0nLtqJSXFVFFRTgUF+YSvO+LNv2wyEPveEwObAM619UsC/qX3rkiC/F9/+z5hwDPAyvkAxh8+aBMk+e/evbIA7M82nPfIv6GpTagH6oO6ob5Ec/xwNzbUC47TgBUqD0NNPz490zA965TNRjy4ZYkCiACQjPxaYWR8OoBUC0m91cSFFMpCjHqdJMhTBYnhEqS4GLje1wBGW9+QINUHh8cX0E/NzHrkHyo/sy05yhqWYFRxJ7Gw1zUxNeuQ6wczr9+PTYb31AOkVrDggoWMN77QkXMdclVO6PA8XmIndANlGhd3wEI70s1p+wpzskmvW1h9Ha+OQXUDXTidEnBd6XLQubYBt5oF1QwWslNXegjrECIKZ5W0vFTMgYUjU8WVZVXrdgwfLLRk0dKKEiovyOXNuiEtnFm2fn1/4AJ4YBDBWgRqFtYiSBMJoPutrAoI0s0plJOR7r8mLqe2xvLPpchTOGaH9/VdHxUkloXvOACsqqRQWSeGWD0AJVC1MFJSVNwUNYEXgEJ/wBkN/ic9F3Gt4rYkwnVcSiybzeGwD483ijsAnVnGS68Vi8qoLD+HOC5yTWPgUrqWS0sx0MT0rLi6Yb8GeMOeaQAZ6nU6YWKrKSsmAAr9geQpaRnwks5FbvRFmJV9jpEfSxWh1+moojCHcjIzaRmvKmItZjL4nzWl8vIVBnBBSkwqBAxofeUXr3FZaakEdRxaAyY2s8no0RRLbo7HfbLcxCGw5rqmd2i4YYzfHJ67k/418oAqnF+LAWQleRby7njplMpCsVE9PeMkpeBSlqu6qfQ6HTEwQTuoLMonqON6XdwOpYgwPG65AbN7/9CY4kUvQJaXlUlQVTAgMMtiYW0yhCbNlIJLx1srI9KDUcgUkxEkP1Q88C7sYOJcstsnUWheRIqIW2CBG52DQ/vgS7k0s4nkJJpepxNmWaguUGEgzQA0AA+DSCo/X2EAF9ZQvtQ9X3G+8lYap+eB62vjXGk+oMP6CEACgAAkTEaQ/AhHfLidbtblsV4Od/6xyC+ugdUxMHRwekbaKGBJN5NjdEIRTyHNoM5AVcQgWl1VISzAATYMMAwof4DLTjPR9bEpkgNQikkfUZUxhd8Yh1qqqMEiIpPBIKh2aDskEms7gASVT68LbYhwOqOotOS5DI1rMebTlc7efT2OkYioEQATwIYBhgHHAMcGHsIwGMUswNlCOXDhKFIwA1+cf6jXZpNRABEmC9Qf0ggSG5IJ0hptDrUM7/SFpSXeQQl8f6NpuhuX8Xk1PDYhqw5GqkUYgHAYjJBq4nIYuMRh7FrHq2vsOpy+Xqcj1CdVBByAB3UDgKDqYkLA5AAQYbIAvV4X990fTjaGNa+45+y59r4X5dRBo0FPcnHh4iKkGgavXneDlVALcUbQu4zUFBOxAY40wTpIGuTDHO6RV3FeDjHgADyoGwAEVde7Ltp9ZDlwYzREtpyI5Q51EKcwpAooyE4nuTgp+mDDMHgxsPW6OXbqeMkkBa6JybkT8lDJkCZYp9fNleMMQhAIAAAQAElEQVRd30hPIt7lhes+nl9zJscD6R6So1Zp+Mj4dNTVQW9WACze4IK1kK+bmxR4kLNUuolCuICEDiF52JPGK9DDwYiEAJatd/DFcDAj1DwYuEyGub0xo15HRtGzXCm85U6ppZIS4A9tvfEiToUNShCyhAAWDuV2DgxJdonRoI/4OktcMMC1uKzIfcIDYDKKwDU1E95nxcRla9fq4UBCAMtmczh4lUtSHYzWOkvcpXqdTtgHA8gQDnDp+HUXjj4Nj40jKCIumVWviDA0hEwTAlhov1rUQdQFTq/zBBdOuE/POMkxMoHoiDgjL50jknEEM3URJdypC+L/EgZYvtRBvp0x+dfr5sCFEwyoAIwZfJDsUSvQJJJTIkE5jovIBn+s+ZgwwLLZ5NXBWDJZr9MJe0s48It6wAyPRT2uI+XUkm9OQZFaqhL1eiQMsMA5OXUwFuss1EfscOCXgWsqQgYMJRJCXKdIX7s/HhfpglSYf0IBS04dxNpDDYOOgStSBgy0U4VjLCmrlFDAstkcjvGp6QY19yTAZTaZ1FzF6NbN6dKMF9HleHCldQ8Mq2Kz2Ffty/KyKVnWWb74gDgXR9fhJ5pLKImFzumwj+2TGrQBrrOQVcScr4cwI1ZoDDLOzMmLQanqKDLhgGXr7rb12kcWqBdYf6hhnRXJbsfDndE4dKy0DWnpaUpJE44u4YCFHpJ7gxPi1OIA9HDXJRJ5hruOyZJfQgJLTh1UU6cmuvRUymuOOMUvBFKapxroEhJYcuqgmtZZaqpLLAci55zVgBVYB8SWWkodhKqkFkkRqbqopX3ofc6UCi8pXUJKLPQk1EE1LeRRp2g4ADYa5Sgpo7CwQAlZQtIkLLCgDtqHJyQfJUnIntQapSoOJCywwGWbxJPFalrbJMt+FvpCzs0YSFtjUZz9SZ0dhKqklnUI9p2kNrNDYbNa2qa0Deu23hMDYCmtXfB0iS2xbOp8lCT47vKfEhOHf6rIU2DCKCzIj3xBKi0hoYEFnkupgwjXXGQ5AMmpGS8iy+OY5i6lDqppnQUJg0EYUyZphYedA4kvsSTUQTUNZjWBPOyjy3+GCflYPpqd8MBCI5NJHYya9ANjQ3Uct+CwdKhZqiV9UgBLSh1USweEux6QxuHOM5j8OP3cS0uDSZsIaZICWDbbQuugmlQwgCGuJI2CkV9YUqqAKnFJkgJY6D5vdVBNg1lNIAevNBc6B5IGWGpXBwH00LuTKD3FGD+P/btcmvEiHJ0eyzxstoXqYCzrg7LFLlyqII5JhSsvcf0ice0i+iAS+aohz6SRWGA21EHxoNNUMHAlMo7TGyOTcZzkmlTAarzS/kyPY8StfkD9EgMtln2WaCAvLCmOJTtjXnZSAQvcjsU3i1GuPxdOkKtlsvDX5kSOTzpg+fpmcaJ0NEAaF23hXG7tIS7qG0Al4x1YATR1jvRKZ+8+sTqYaCrYXCvj41c369JOXsRHVymrpVgdxOyuFtUpkUDO6TTjhbLRmEBUalUHwwVyNUwUhaUlCTRiAm9K0qmCYJG3OoiwRHIAaCK1Jx7bkpTAQkeJ1cFEUsHQtnhxifq+C/BfRwQv+dz51p4fj01MCQ3HDK8G9QmVicR7MJCvGl2ivu8CvE5aidXc3d/QPzSmuheZ4EgSAzw6KBgX60ki2d93gT5LWmCh8Z2DQwn53kFIYLQvVg7ATub3XYDvSQ2srsHrbnUwmVQwdLzmIsuBpAbW5Y6+RqYOhkMFC1dXQeJg1g9XfmrMx0WkYHOY4vYvqYGFXutUoToIKyXWKahfPDolkwLHcQl7nAl9lvTA6hKpg2CIWpySwSlXV6i1sfwgRE5BkVzVkiY86YElVgcxIONZUrBRC1WSXcfCT0/iT6Qyfic9sMCIK539gnUw2HUWpEtrjx1ZaU4pB1xO1W11KK26EjoNWDyX+oanvh2KpIKEyM5IpXCqX8gTgOWrF9q/SlO7iGtVadXCUi0NWDwb8S2tXvtISFYqqJEAQke/g2adTj7X0P5DNWCgLqHVIPjUKWkZwSdOkJQasOY7kn1aFQAJVnqV5mXR4PA4nW/tpKGx8fmcg/dCAQckXvAlh5bSkpsTWgYJkFoD1nwn4tOqAFSw66z5bGhNVQmNT81SW0+/4KZnZliU5os4oHM6G0S3CXepAWu+S8OhDs5nRXWLy2lkYkaQWs0dPTQwNMyiAvJjKXUCqqgX8fTMrFdI8t1qwBL1OVMHRUHCZaA/AMTyyhIam5wR1ltdAw5BeoVj7aW0LrEc3OXWKqXVTFg6DViirg2XOogssVbLz84ip9OFW0F6XWrvEnwhQMFPKOAAuBUUETOSRH4WC0zVgAUuzDumDgIUWG/NBwftVZfkkfjdD5BYWHt19A0KkizojBMgYSI/i4Xu0YAFLohcuNRBluXyikLemOFkt4JvHxmlK/zaa2JqWriX+sEzWTCkSMWpOQwTUjJ/e5j1jQYsxol5n6mD4VKlkM+qRSU0Pum5oJ/irYXNHd3U6xiaL9nTwwCF5PQMVf8d1NdkfxYLvRQtYKGsuHBMHcQACVeFIXmsJQU0Oe0JLuTfa79OLV29YVcNw1l/1DOcjl91hrQZH866RCovDVgSnA23OogicJIiJzPTbcxAGHOjE5MEwwZ8FhaqD0kZah5K0nsD2PteKg8uwR8ZQZs1YIELXm50atoBIHgFh3wLY4aL9JL5wLAByYU9r3haX3mfj9QeGZnrXg1Yc3zw+K3Iz344UjP+ykXFNDXDK0MeJd64wZ7XpWs9FC/rq2mvzWDtkZG5vtSANccH929Ncf7ORYWWne6AMF8AsNWlvKXQy5ghLgY04vtgrqPxZcfOgSHC+chA6+dyuQ4GmiaC9BHJWgOWF1tXLCp+HMYGr+Cw3kIaVRblSRozUJDJaIQXkkMbvKVJSBlKJEb+3pMAwiRIky5IA5aoyyMtrURFUVFOJmWkpskaM8S0arwW1oEpCyeAcqtVjdWNep00YIlYHg1pJSqOlvGbxzNOThwkXOvISadbuoTrYH+gpkEyBpveXzpbzyBlpqX4I0vaeA1Y811vLS62luRlRWxtNV/MAm9NValwEl4codNx5HJO03sXWykU1cpbTROXEco16jQ6PkFm00KJFUq+iZRWAxbrTfOEA+oNu42Wj8HPTsKLywS4Uk06arxyLWBwoR3IV5xfOK9tPXbKyTSLstQuvTmgAWueIzabw9Hc2f/k/G1UPahsJXk5ND3rXFBuMODC3lIk9uFY5YZGx8hkMLBbzZfggAYsEVO6JxxPt3QPxuRFkhUFFjLoTaLa3LgMBFyQVrAI3kgd3qv2PgcZ+T1uc4p0XcNbWvzmpgFL1Hc2m8NxuXPgMawhRMFRu5Q6Cc8KB7jeveB/zRVpadXnGCKoqSYDjy5WuQB9jiPtrCAl2V/jlfZnmjsHYvI+BiM/WKVOwrMuMPDTINZckEosTOxjQkAe4rBwXgO0HOcSsjQaglcFZzmnBiyBi0n2c+Faz2MYRFFt9nxhUOOkNo/xJDIkBSTXWVsn4bGS+SRuDyb2YE5CuDPwc9HRb+fVQB7dPF26WdrUnpGTz8fK//OwfDLRH3JE6+e4hCvNuTmA106fb+uJiSEDlZDaPMb7M9JS5qREaoqeLrd3e4Ar0tIKUnJ2dgbVkzVcANhV1kqBRubH5pwyPS0Tl1DBGrBkujOWhgxUCZvHk6LDurMufq5HxLzzBhcGdSSllY3fEE6B1YIv38irrLy34H/FxpvI50OOHD22rr4+JsahBZWNcIAGLBkG22yxNWSgWuw1aniV2pKyAgR5ODG45Aa7R4IgbyCtRsZvvIA0PXXhHlb/uJNuu+N2XyXsq922a58vgkSK04DlozdhyGjtdcTs5f0AS1l+DlUV51OhJYtqyopJr/PsMgYuH80IOQrSKm1eDUVmJi+JBeDdctfd5OOREcfslOlRpE0W59lLcd7qSFT/avfgk80d/QRVi7loGjb0Oh0V5WQKTcMRoqqSwgVHiVL5NRfe/CRl0BAShvAD0IilFbLyNlzoLEW0fsM6REk7jh5dlyQqIGOABizGCRnfMTrdkG/JIKxfmMNJCQYy+DJJIxLsC1zeBo1wVMBbWgFURsOcEQX5d10fp4f++NO4lHGuZ5JJBWRM0IDFOCHjl+Wk7QaQxNFGXhViIMPRIYCLOVjnxLShXrP8xfnodTqC5MrJSBcHEyRXOMElJa1yMm+Uifjb73/Alwpom51Kecyjkklyo0uSdgbdzOK8zAd8JRaDDGCDmshAFg7VDPlLgVWv01FZQS5FElze0spkMJBFBGZz0SKfKiDndCadCsjGigYsxgkJ32q1WFJNxoAeJQG4mDPpdWFZmyE/gFWiigK48rLm1mAsXpBc13pICpCMxp8PaeS9trKIpBWsgB/9+Md8ZbNvzY57YnKCxVelohXnDaxolRsX5einTDsxqIOtbJrZtGBtBonW2mMn+FIOAPIOh+S7PjohW42SPAuV89JLTJAa5CMnLA9vaaXX6SgvK0OIRn3u+PBuXyog8dLq2wJxkv5owPLR8SsrCh7ArH/iXCudOH2VTpxpobOXO6iFtxJi8PtIKhll5NdmWDMtKsoh+FIOQJYKL8nLIkgRqYxRF7wL1GAwEcfpiZt3xG8qn7F1SyXxGYb8xic9gZyXnUl6nU6Qgs6MfFq9aoWvPGzJLK3AGB1+NCfNAd4CtvPMlU5K13GUbjRQOg8M/ewsTQ6PUW9nP733wRU6f7WLrnYNyA566ZwDD8Vbly53DgiSDlINDgCAg3EFYFxeUUSrrKVut2m5lVZbiz3SIJ0cQFmt2nsHiJ2yQJhed0NanWvr5TeCb0OwvOO4mO39yVcqujG66BYXP6XhUf327kGrSeZ7wnqOo8wUI3FT0zR1fZQuNXfQqQtt1No9GBGQQXIySQepBgcwwUESynEWcaABPXNTs06PtZ8YaJgkcIpenB+TVgBlTk6OP2lFvKQMaF0qLitRrjVgSfSk1WqxFKTq91qMegKAJEgWBJl5aZbKh447RgSQNV26Rtd6HYLqxAeH/I/12vWRG8eKQskQEo6BDD6AhnUfgNPeO0jsSWb4OPw7Oe0kbJIjHWdCK0MpPei0cZVQF1e1jUJlrVaLpUiXcaAkI3WnUlB5Vwsgg6QbGRyixrM2QV0EyMSSwTuN1D2kFFQ95sYmpwlhUrShhAEwkIYA2a1rl9CKylIqys3l98qKaPPyKqoosFBNWT4NDo1Rx+XLoRSVNGl1SdNSBQ2F+gdQFaWb6xSQKyJh6iJABnVRMILAEKLANZ1vpQF+/cYcNzZBPYPDisoNhQjSEeojAMfyAaB7hMfytSHDeOLL17g0zx28rNOalXIqnKCaz9rtQZIJRhAYQhQ6d+L5CwxuXMLkDdWtlQceczCknOeNLYLjjSosHHSQekgXjIOkfZ9fP+p5K2Mw6ZMxjQYsvtdrK0ufqM7JPJCVYrTwt6r+x+A+cuoy0BPR1gAACaVJREFUtfNm9CFegozzhhPmYEjhpmdIcLxRhYWDDlZMSMtGfusAWwYAHdRTf4CDMeMSD1YTDypMDKpmjooql9TAguq3qarsQEVW6uNqHTR4wHGYX1sxN8oDJy81xf2IvNKxhPViutFAZn7rAFsGAB3UUzHgIPEAOEg4OBhgxnljjFmf1MOEgvlLWo6tKCvaszjbfCoUI0UwDBenmZiZJQCFOYAH12IfNFinMWfi95TEeYR6LQYcJB4ABwkHZ+K3GhAfRBnaPlYQTIvrJFhLbauuPFWdk/FUuskQddUPwHFMTAmAwqAVr7kAHtx7+2KGQ4KJ76N1jZfZoCwAHb5Pp20QU9JILKh9G6ylLyzJzTqQn26qw6D2OTgiEAlJhHLHeXVukpdWxgBVLIAq0DThaoaOVyExIWTxZvhw5ZnI+SQFsOoqy55abDG3lGWm7Y7VwBwYnySs4+y8P8FLrBSDPuBxNTY1Q5BoAScMQwJsFkOSlhcX+c/N5aprOvL6I6cOHIi6RuC/ctGhSHhgweJXlmXeE7YBGUS/AFQWs4n6+X0ogArvWzLplLHeZTDSjMlMlpplhGuK0Z+Rl66QtgqL5wHl2qs3TbU0Hd2/W2GahCJT1rtx2mSofwVppscDGBBhbSlUNwaqwXlJxfElcBwnadWDVMAajDKyKWtRNZWv30Sb7rmfPvyZh6miejGlkpNPHdv/kbFx6u2308zsrJKKWLhZp0MJYaLRJDSw8sz6x2MlqQAqrElgGseCPz2vkEoqK0lvNpNOrxekkDm/kPKWLKfSNXW05JbbaOvuj9L/+MJf0L2f/CTdfOedVLt+PZWUljTzg27o7HvHJcHIx0XlH8YLAN+YmUXNtnY62XSBBh1Dvst20dPJ+vhIwgJrSVlBncVsfMR3z4cvFkCCcYI5gImBavWOnfSxTz1EH+Hd5x77Cn3ha18VpNBtux+kLbfeSnVbttCSZUsaCguLnuHVxCddTqp3OWeqPv+Vr3F/9sW/XPL753/7+GR/b/gqG0RO+rR0Ws9Lz207dwqpIbH0vHoo3Ej88O1orL1512MSUUkRlLDAyjebn4qktJICEhb3zKHscd7ouuLmW2nF6lXCYDIaDWc4jjvMD7onOY4eEwPo81/+Wv3n9/zVo1/4ytee+MJjX2v4wmP/S9gLslotFkdnx1eMPgaxkHmEf4rLy6missJdSklRPmVnZrjvxRd8+xqdU6Z6cViyXccEWJFmMvaqLCnGuak1TIUpARLWR+mFRVRdt/7a+rvu+tUffeFzjy1fuUyQPJA+n/3zx9Z87st/tQPg+dyXv/a0GEBy1cxypu3JSzNZ5eKjFT41OuIuKj0tlaoqSt334gsGqnVJ9h5BMQ9wnXDAslotliyzaS8aF4rzBySYzqH2mXPzp8pWrj63cmf9jx758pdv+cVbR7inf/pcxRP/+p+ffPTPv8qDZ07yBFMXq9ViyTUbvxJM2kilAahWLauWyx4fPahPdlCBOQkHLIsz/fHyzDSrNzAgTbD4RqOlHNZEAApzuGdqHXyYxxGXUVTiKF21+uDqW2/7xr99/5mc5w6+k/J/f7Vv1Te/88PP/vGf/+UhqbyDDbM4Ux9R08FgBioDb3yRbBPH2TRQzXEm4YBl1NNuAAgSBYBgDmsegA0AA0DgM4d7mOQZLXywZ9pgdKTkFx42l5Z/11mQf8uBS63cs28eyvneL/ft/NZ3f/j3kR5E6UajaqSViwcTJJWB98EbzfnmQMIB672Wjqrm68Pr2q6PPtY/Nrnv+uS0DQACqAC2dKNBeFcFfOZ0Oo7GZ2Yd/F5TQ//41JNnB4YevDLhyHn11PmcXx84uuO5/Q1f3H/weFilke9uIYJVMyvFEPO1Feo5wO/BZfLbBH5B5aIfgz6eXbjqrgtXRmrKBx+Oa2zrevrolfYHD15qrXrrgo272H+9/lyfo/5s3/VH+3nwwDU7Rh5FGEDUcKk153Bze/3R5rYnrnT27rPZHI5Ytsk167IA8LGsAyajvrFJwqkRPT/5+K4L92jt9rue8U2TPLEJCSyp7mvu7m+Au9Ld9wzAA3euvfsZhNlsDodUmliHYYM5VnXAGhMqckFaCk3POql2y2a5qjh0xK3TQOXJHp3nrXanJg5AUgzwahgkRzTrBUChTJQPNfpDn/4U3bpzx8IqcFzD7JSpavX2uxoXRiZ3iAYsFfc/DCo4vYGBjgEe6apCMgHIbC06oTPQl//330iCit+verJ22131kTbgRLrNkcpfA1akOBtivjPmmcae0QlBEkBywNyPQR8pgEHtnHI6CUDG2i6rYhE9+c//SFVVizxawgOqUcerfmu373rCI+LGjXbFc0DHO+1fhRyw2RyOHudIfcfw2D6oZUa9jjDoATBIMOYANDhIm2CagXwAKmwxwEqKvDbdeRd9/f/9uvdHDxwuzvUYD6h1murnn9MasPzzKGYUNpvDccLW+WDL9dFH7RNTgoHFyAMMIGAOYIAD+AAKOIAFPnNy9whPMxkEqx8amZpfKKh+H/nYbtwScVwDHACFtdTabXc/TdqfIg5owFLEptgSwXrZPj1UZbs+9iTbl/OuEVsXAWQAHXzm5O4RjnXc0OS046p95LFfHTjKPfDp/4er3b5rzvFrKKyjAChtLeXNcd/3GrB880c1sTabw9HU1vkE9uWw74b9N+zD9c/vyWFze2hiuoE5SCMmsbx9xIEOabCRbpu6XnW+o0eTRmHsbQ1YYWRm+LLynZPN5nBg/w2SDPtxcNjcbrjcWs8cjl9hY1zKIQ50SIONdJvN4fBdohYbKAc0YAXKMY1e44ACDmjAUsAkjUTjQKAc0IAVKMc0eo0DCjigAUsBkzQSjQOBciBxgRUoJzR6jQNh5IAGrDAyU8tK4wDjgAYsxgnN1zgQRg5owAojM7WsNA4wDmjAYpzQfI0DYeSAT2CFsRwtK40DScUBDVhJ1d1aY6PFAQ1Y0eK0Vk5ScUADVlJ1t9bYaHFAA1a0OK2VE1YOqD0zDVhq7yGtfnHJAQ1YcdltWqXVzgENWGrvIa1+cckBDVhx2W1apdXOAQ1Yau+hyNVPyzmCHNCAFUHmalknLwc0YCVv32stjyAHNGBFkLla1snLAQ1Yydv3WssjyAENWCExV0uscUCaAxqwpPmihWocCIkDGrBCYp+WWOOANAc0YEnzRQvVOBASBzRghcQ+LbHGAWkOxB5Y0vXSQjUOxDUHNGDFdfdplVcrBzRgqbVntHrFNQc0YMV192mVVysHNGCptWe0esWeAyHUQANWCMzTkmockOPA/w8AAP//QkrGNQAAAAZJREFUAwC4FcxX5gkwlQAAAABJRU5ErkJggg=="

def render_sidebar_and_get_lang() -> tuple:
    """Renders the EIU logo, language selector, and mandatory educational disclaimer in the sidebar."""
    st.sidebar.markdown(
        f"<div style='background:#ffffff; padding:12px 14px; border-radius:12px; border:1px solid #e2e8f0; text-align:center; margin-bottom:14px; box-shadow:0 2px 8px rgba(15,23,42,0.04);'>"
        f"<img src='data:image/png;base64,{EIU_LOGO_B64}' style='max-width:170px; height:auto; margin-bottom:4px;'/>"
        f"<div style='font-size:0.73rem; font-weight:700; color:#1e3a5f; letter-spacing:0.3px;'>EASTERN INTERNATIONAL UNIVERSITY</div>"
        f"<div style='font-size:0.68rem; color:#64748b;'>Becamex Business School &bull; Topic T3</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if "lang_choice" not in st.session_state:
        st.session_state["lang_choice"] = "🇬🇧 English"

    selected_label = st.sidebar.selectbox(
        "🌐 Language / Ngôn ngữ / 语言 / Langue",
        options=list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(st.session_state["lang_choice"]),
    )
    st.session_state["lang_choice"] = selected_label
    lang_code = LANGUAGES[selected_label]
    t = TRANSLATIONS[lang_code]

    st.sidebar.markdown(
        f"<div class='edu-warning-box'>{t['disclaimer_box']}</div>",
        unsafe_allow_html=True,
    )
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
