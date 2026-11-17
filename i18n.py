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


def render_sidebar_and_get_lang() -> tuple:
    """Renders the language selector and mandatory educational disclaimer in the sidebar."""
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

    if main_pipe is not None and base_pipe is not None:
        prob_gb = float(main_pipe.predict_proba(input_df)[0, 1])
        prob_lr = float(base_pipe.predict_proba(input_df)[0, 1])
    else:
        prob_lr = float(1.0 / (1.0 + np.exp(-z_linear)))
        prob_gb = prob_lr

    return prob_gb, prob_lr, contributions
