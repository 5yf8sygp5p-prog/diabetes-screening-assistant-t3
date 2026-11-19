"""
pages/4_Batch_Screening.py — Bonus Batch CSV Prediction (4 Languages: EN / VI / ZH / FR)
========================================================================================
"""

from pathlib import Path
import pandas as pd
import streamlit as st
from i18n import inject_friendly_css, load_artifacts, render_sidebar_and_get_lang, score_single_case

st.set_page_config(
    page_title="Batch Screening | Diabetes Screening Assistant",
    page_icon="📂",
    layout="wide",
)

inject_friendly_css()
lang_code, t = render_sidebar_and_get_lang()
main_pipe, base_pipe, meta = load_artifacts()

required_cols = meta["feature_list"]
default_th = float(meta["cost_assumptions"]["chosen_threshold"])
active_th = float(st.session_state.get("active_threshold", default_th))

BATCH_TITLES = {
    "en": (
        "Batch Screening (CSV Upload)",
        "Upload a CSV file with the 7 questionnaire columns (`Age`, `BMI`, `HighBP`, `HighChol`, `GenHlth`, `DiffWalk`, `HeartDiseaseorAttack`) to get a ranked risk table and downloadable CSV.",
        "Upload CSV file",
        "Showing preview using built-in `data/sample_input.csv` (25 rows). Upload your own CSV above to screen custom rows.",
        "⬇️ Download Ranked Results (CSV)",
    ),
    "vi": (
        "Tầm soát Hàng loạt qua File CSV (Bonus)",
        "Tải lên file CSV chứa 7 cột câu hỏi (`Age`, `BMI`, `HighBP`, `HighChol`, `GenHlth`, `DiffWalk`, `HeartDiseaseorAttack`) để xếp hạng mức độ rủi ro và tải kết quả về máy.",
        "Tải lên file CSV",
        "Đang hiển thị bản xem trước bằng dữ liệu mẫu `data/sample_input.csv` (25 người). Hãy tải file CSV của bạn lên ở khung trên.",
        "⬇️ Tải bảng kết quả đã xếp hạng (CSV)",
    ),
    "zh": (
        "批量筛查 (CSV 文件上传)",
        "上传包含 7 个筛查问题列 (`Age`, `BMI`, `HighBP`, `HighChol`, `GenHlth`, `DiffWalk`, `HeartDiseaseorAttack`) 的 CSV 文件，获取按风险排序的表格并下载结果。",
        "上传 CSV 文件",
        "当前显示内置样本 `data/sample_input.csv`（25 行）的预览结果。您可以在上方上传自定义 CSV 文件。",
        "⬇️ 下载排序后的筛查结果 (CSV)",
    ),
    "fr": (
        "Dépistage par Lot (Import CSV)",
        "Importez un fichier CSV contenant les 7 colonnes (`Age`, `BMI`, `HighBP`, `HighChol`, `GenHlth`, `DiffWalk`, `HeartDiseaseorAttack`) pour obtenir un tableau classé par risque.",
        "Importer un fichier CSV",
        "Aperçu avec le fichier intégré `data/sample_input.csv` (25 lignes). Importez votre propre fichier CSV ci-dessus.",
        "⬇️ Télécharger les résultats classés (CSV)",
    ),
}

b_title, b_sub, b_upload, b_info, b_dl = BATCH_TITLES.get(lang_code, BATCH_TITLES["en"])

st.title(b_title)
st.markdown(b_sub)

uploaded_file = st.file_uploader(b_upload, type=["csv"])
base_dir = Path(__file__).resolve().parent
sample_csv = base_dir.parent / "data" / "sample_input.csv"
if not sample_csv.exists():
    for cand in [base_dir / "data" / "sample_input.csv", base_dir / "sample_input.csv", base_dir.parent / "sample_input.csv"]:
        if cand.exists():
            sample_csv = cand
            break

batch_df = None
if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(
            f"⚠️ Could not read uploaded CSV file. Please upload a valid UTF-8 `.csv` file with the 7 required columns. ({exc})"
        )
        batch_df = None
elif sample_csv.exists():
    try:
        st.info(b_info)
        batch_df = pd.read_csv(sample_csv)
    except Exception as exc:
        st.error(f"⚠️ Could not read built-in sample CSV: {exc}")
        batch_df = None

if batch_df is not None:
    if len(batch_df) == 0:
        st.warning("⚠️ The uploaded CSV file is empty (0 rows). Please upload a CSV with at least 1 row.")
    else:
        missing = [c for c in required_cols if c not in batch_df.columns]
        if missing:
            st.error(
                f"⚠️ Missing required columns: `{missing}`. "
                f"Expected all 7 columns: `{required_cols}`."
            )
        else:
            MAX_ROWS = 5000
            if len(batch_df) > MAX_ROWS:
                st.warning(
                    f"⚠️ Uploaded file has {len(batch_df):,} rows. Showing and scoring the first {MAX_ROWS:,} rows to keep the app fast."
                )
                batch_df = batch_df.head(MAX_ROWS).copy()

            out_df = batch_df[required_cols].copy()
            # Safely coerce non-numeric strings to NaN and drop/impute cleanly without crashing
            bad_cells = 0
            for col in required_cols:
                coerced = pd.to_numeric(out_df[col], errors="coerce")
                bad_cells += int(coerced.isna().sum())
                out_df[col] = coerced

            if bad_cells > 0:
                valid_mask = out_df[required_cols].notna().all(axis=1)
                dropped = int((~valid_mask).sum())
                if dropped == len(out_df):
                    st.error(
                        "⚠️ All rows contain missing or non-numeric values in the 7 required columns. "
                        "Please check your CSV formatting (numbers only)."
                    )
                    st.stop()
                st.warning(
                    f"⚠️ Found {bad_cells} invalid/missing value(s) across {dropped} row(s). "
                    f"Those incomplete rows were skipped safely."
                )
                out_df = out_df[valid_mask].reset_index(drop=True)

            # Clamp extreme out-of-range inputs safely so the model never crashes
            bounds = {
                "Age": (1, 13),
                "BMI": (12.0, 70.0),
                "GenHlth": (1, 5),
                "HighBP": (0, 1),
                "HighChol": (0, 1),
                "HeartDiseaseorAttack": (0, 1),
                "DiffWalk": (0, 1),
            }
            clamped_count = 0
            for col, (lo, hi) in bounds.items():
                if col in out_df.columns:
                    oob = int(((out_df[col] < lo) | (out_df[col] > hi)).sum())
                    clamped_count += oob
                    out_df[col] = out_df[col].clip(lower=lo, upper=hi)
            if clamped_count > 0:
                st.warning(
                    f"⚠️ {clamped_count} out-of-range value(s) were safely clamped to valid clinical ranges "
                    f"(Age 1–13, BMI 12–70, GenHlth 1–5, binary flags 0/1)."
                )

            probs_gb = []
            probs_lr = []
            decisions = []
            for idx in range(len(out_df)):
                row_df = out_df.iloc[[idx]]
                p_gb, p_lr, _ = score_single_case(row_df, main_pipe, base_pipe, meta)
                probs_gb.append(round(p_gb, 4))
                probs_lr.append(round(p_lr, 4))
                decisions.append(
                    t["res_recommend_flag"] if p_gb >= active_th else t["res_recommend_safe"]
                )

            out_df["Risk_Score_HGB"] = probs_gb
            out_df["Risk_Score_LogReg"] = probs_lr
            out_df[f"Decision (Threshold={active_th:.2f})"] = decisions
            out_df = out_df.sort_values(by="Risk_Score_HGB", ascending=False).reset_index(drop=True)

            st.dataframe(out_df, use_container_width=True)
            st.download_button(
                b_dl,
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="diabetes_batch_screening_ranked.csv",
                mime="text/csv",
            )
