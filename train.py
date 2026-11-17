"""
train.py — Complete Training, Validation Threshold Tuning, Permutation Importance & Health Equity Audit
=======================================================================================================
Topic T3: Diabetes Risk Screening (CDC BRFSS 2015)
User-Decision Statement:
"An adult answering questions at home uses the app to decide whether to get a blood sugar test."
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 7 core questions for the user-friendly screening form (no income/education required from user)
MINIMAL_FEATURES = [
    "Age",
    "BMI",
    "HighBP",
    "HighChol",
    "GenHlth",
    "DiffWalk",
    "HeartDiseaseorAttack",
]

# Full available survey features (includes physical activity, education, and income)
FULL_FEATURES = [
    "Age",
    "BMI",
    "HighBP",
    "HighChol",
    "GenHlth",
    "DiffWalk",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Education",
    "Income",
]

TARGET_COL = "Diabetes_binary"


def build_preprocessor(feature_cols: list) -> ColumnTransformer:
    continuous_cols = [c for c in ["BMI"] if c in feature_cols]
    ordinal_cols = [c for c in ["Age", "GenHlth", "Education", "Income"] if c in feature_cols]
    binary_cols = [
        c for c in ["HighBP", "HighChol", "DiffWalk", "HeartDiseaseorAttack", "PhysActivity"] if c in feature_cols
    ]

    return ColumnTransformer(
        transformers=[
            (
                "continuous",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                continuous_cols,
            ),
            (
                "ordinal",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("scaler", StandardScaler())]),
                ordinal_cols,
            ),
            (
                "binary",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
                binary_cols,
            ),
        ],
        remainder="drop",
    )


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Computes 95% confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    half = (z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * n)) / n)) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def compute_subgroup_equity(df_test: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    income_bins = [
        ("Low (< $25k)", df_test["Income"] <= 4),
        ("Middle ($25k–$75k)", (df_test["Income"] >= 5) & (df_test["Income"] <= 7)),
        ("High ($75k+)", df_test["Income"] == 8),
    ]
    edu_bins = [
        ("Less than high school", df_test["Education"] <= 3),
        ("High school graduate", df_test["Education"] == 4),
        ("Some college", df_test["Education"] == 5),
        ("College graduate", df_test["Education"] == 6),
    ]

    out = {"Income": [], "Education": []}
    for group_key, bins in [("Income", income_bins), ("Education", edu_bins)]:
        for label, mask in bins:
            mask_arr = mask.values
            n_grp = int(mask_arr.sum())
            if n_grp == 0:
                continue
            yt = y_true[mask_arr]
            yp = y_pred[mask_arr]
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            pos = int(tp + fn)
            neg = int(tn + fp)

            recall_val = tp / pos if pos > 0 else 0.0
            rec_lo, rec_hi = wilson_ci(tp, pos)

            fnr_val = fn / pos if pos > 0 else 0.0
            fnr_lo, fnr_hi = wilson_ci(fn, pos)

            fpr_val = fp / neg if neg > 0 else 0.0
            fpr_lo, fpr_hi = wilson_ci(fp, neg)

            flagged_count = int(tp + fp)
            prec_val = tp / flagged_count if flagged_count > 0 else 0.0
            prec_lo, prec_hi = wilson_ci(tp, flagged_count)

            out[group_key].append(
                {
                    "group": label,
                    "n": n_grp,
                    "prevalence": round(float(yt.mean()), 4),
                    "recall": round(float(recall_val), 4),
                    "recall_ci": [rec_lo, rec_hi],
                    "fnr": round(float(fnr_val), 4),
                    "fnr_ci": [fnr_lo, fnr_hi],
                    "fpr": round(float(fpr_val), 4),
                    "fpr_ci": [fpr_lo, fpr_hi],
                    "precision": round(float(prec_val), 4),
                    "precision_ci": [prec_lo, prec_hi],
                    "flagged": round(float(yp.mean()), 4),
                }
            )
    return out


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    models_dir = base_dir / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = data_dir / "diabetes_binary_health_indicators_BRFSS2015.csv"
    print(f"[1/6] Loading CDC BRFSS 2015 dataset from {raw_csv_path}...")
    df_raw = pd.read_csv(raw_csv_path)
    n_raw = len(df_raw)
    prev_raw = float(df_raw[TARGET_COL].mean())

    # Only drop exact duplicates if all 21 original BRFSS columns are present;
    # otherwise keep all 253,680 respondents so distinct respondents sharing 7-10 discrete answers are preserved
    if df_raw.shape[1] >= 21:
        df = df_raw.drop_duplicates().reset_index(drop=True)
    else:
        df = df_raw.reset_index(drop=True)
    n_clean = len(df)
    n_dups = n_raw - n_clean
    prev_clean = float(df[TARGET_COL].mean())
    print(f"      Raw rows: {n_raw:,} ({prev_raw:.1%} positive) -> Deduped: {n_clean:,} ({prev_clean:.1%} positive)")

    # Save 25-row sample_input.csv for Batch Screening
    df[MINIMAL_FEATURES].head(25).to_csv(data_dir / "sample_input.csv", index=False)

    # Stratified 60 / 20 / 20 Split (Train 60%, Validation 20%, Test 20%)
    y_all = df[TARGET_COL].astype(int)
    df_train, df_temp, y_train, y_temp = train_test_split(
        df, y_all, test_size=0.40, random_state=42, stratify=y_all
    )
    df_val, df_test, y_val, y_test = train_test_split(
        df_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    print(f"[2/6] Split sizes: Train={len(df_train):,} | Val={len(df_val):,} | Test={len(df_test):,}")

    # Train all 4 Models
    print("[3/6] Training 4 Models (Logistic Regression & Histogram Gradient Boosting, 7-Q vs Full)...")
    lr_min = Pipeline(
        [
            ("preprocessor", build_preprocessor(MINIMAL_FEATURES)),
            ("classifier", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ]
    )
    lr_full = Pipeline(
        [
            ("preprocessor", build_preprocessor(FULL_FEATURES)),
            ("classifier", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ]
    )
    hgb_min = Pipeline(
        [
            ("preprocessor", build_preprocessor(MINIMAL_FEATURES)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=200, learning_rate=0.05, max_depth=5, min_samples_leaf=30, random_state=42
                ),
            ),
        ]
    )
    hgb_full = Pipeline(
        [
            ("preprocessor", build_preprocessor(FULL_FEATURES)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=200, learning_rate=0.05, max_depth=5, min_samples_leaf=30, random_state=42
                ),
            ),
        ]
    )

    lr_min.fit(df_train[MINIMAL_FEATURES], y_train)
    lr_full.fit(df_train[FULL_FEATURES], y_train)
    hgb_min.fit(df_train[MINIMAL_FEATURES], y_train)
    hgb_full.fit(df_train[FULL_FEATURES], y_train)

    # Validation threshold optimization (default cost ratio FN:FP = 5:1, i.e., C_FN=5.0, C_FP=1.0)
    print("[4/6] Computing validation cost curve & test threshold grid...")
    val_probs = hgb_min.predict_proba(df_val[MINIMAL_FEATURES])[:, 1]
    test_probs = hgb_min.predict_proba(df_test[MINIMAL_FEATURES])[:, 1]
    test_probs_full = hgb_full.predict_proba(df_test[FULL_FEATURES])[:, 1]

    thresholds_grid = np.round(np.linspace(0.02, 0.80, 40), 2)
    curve_data = []
    y_val_arr = y_val.values
    y_test_arr = y_test.values
    n_val = len(y_val_arr)
    n_test = len(y_test_arr)

    for th in thresholds_grid:
        vp = (val_probs >= th).astype(int)
        v_tn, v_fp, v_fn, v_tp = confusion_matrix(y_val_arr, vp, labels=[0, 1]).ravel()

        tp_pred = (test_probs >= th).astype(int)
        t_tn, t_fp, t_fn, t_tp = confusion_matrix(y_test_arr, tp_pred, labels=[0, 1]).ravel()
        t_pos = t_tp + t_fn
        t_neg = t_tn + t_fp

        curve_data.append(
            {
                "threshold": float(th),
                "val_fp_rate_per_person": round(float(v_fp / n_val), 5),
                "val_fn_rate_per_person": round(float(v_fn / n_val), 5),
                "test_recall": round(float(t_tp / t_pos if t_pos > 0 else 0.0), 4),
                "test_precision": round(float(t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0.0), 4),
                "test_fpr": round(float(t_fp / t_neg if t_neg > 0 else 0.0), 4),
                "test_flagged": round(float((t_tp + t_fp) / n_test), 4),
                "per_1000": {
                    "tp": int(round(1000.0 * t_tp / n_test)),
                    "fn": int(round(1000.0 * t_fn / n_test)),
                    "fp": int(round(1000.0 * t_fp / n_test)),
                    "tn": int(round(1000.0 * t_tn / n_test)),
                },
            }
        )

    # Optimal threshold at default 5:1 cost ratio on validation set
    default_cost_ratio = 5.0
    best_pt = min(
        curve_data,
        key=lambda row: row["val_fp_rate_per_person"] * 1.0 + row["val_fn_rate_per_person"] * default_cost_ratio,
    )
    optimal_threshold = float(best_pt["threshold"])
    print(f"      Optimal threshold at 5:1 cost ratio: tau* = {optimal_threshold:.2f}")

    # Evaluate all 4 models for "Models compared" table
    models_compared = []
    for name, pipe, feats in [
        ("Logistic Regression (minimal)", lr_min, MINIMAL_FEATURES),
        ("Logistic Regression (full)", lr_full, FULL_FEATURES),
        ("Histogram Gradient Boosting (minimal)", hgb_min, MINIMAL_FEATURES),
        ("Histogram Gradient Boosting (full)", hgb_full, FULL_FEATURES),
    ]:
        p_tr = pipe.predict_proba(df_train[feats])[:, 1]
        p_v = pipe.predict_proba(df_val[feats])[:, 1]
        p_te = pipe.predict_proba(df_test[feats])[:, 1]

        # Each model's own validation threshold at 5:1 cost ratio
        best_th_m = 0.18
        best_c_m = 1e9
        for th in thresholds_grid:
            vp_m = (p_v >= th).astype(int)
            _, fp_m, fn_m, _ = confusion_matrix(y_val_arr, vp_m, labels=[0, 1]).ravel()
            c_m = (fp_m + 5.0 * fn_m) / n_val
            if c_m < best_c_m:
                best_c_m = c_m
                best_th_m = float(th)

        pred_te = (p_te >= best_th_m).astype(int)
        tn_m, fp_m, fn_m, tp_m = confusion_matrix(y_test_arr, pred_te, labels=[0, 1]).ravel()
        models_compared.append(
            {
                "model": name,
                "questions": len(feats),
                "threshold": best_th_m,
                "pr_auc_train": round(float(average_precision_score(y_train, p_tr)), 3),
                "pr_auc_test": round(float(average_precision_score(y_test, p_te)), 3),
                "roc_auc_test": round(float(roc_auc_score(y_test, p_te)), 3),
                "brier_test": round(float(brier_score_loss(y_test, p_te)), 4),
                "recall": round(float(recall_score(y_test, pred_te, zero_division=0)), 3),
                "precision": round(float(precision_score(y_test, pred_te, zero_division=0)), 3),
                "fpr": round(float(fp_m / (tn_m + fp_m)), 3),
                "f1": round(float(f1_score(y_test, pred_te, zero_division=0)), 3),
            }
        )

    # Permutation Importance on Validation Slice (Drop in PR-AUC) + Odds Ratios
    print("[5/6] Computing Permutation Importance (Drop in PR-AUC) & Odds Ratios...")
    sub_val = df_val.sample(n=min(8000, len(df_val)), random_state=42)
    perm = permutation_importance(
        hgb_min,
        sub_val[MINIMAL_FEATURES],
        sub_val[TARGET_COL].astype(int),
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
    )
    perm_list = []
    for idx, feat in enumerate(MINIMAL_FEATURES):
        perm_list.append(
            {
                "feature": feat,
                "importance_mean": round(float(perm.importances_mean[idx]), 4),
                "importance_std": round(float(perm.importances_std[idx]), 4),
            }
        )
    perm_list.sort(key=lambda x: x["importance_mean"], reverse=True)

    # Fit unscaled Logistic Regression on MINIMAL_FEATURES to get natural per-1-unit Odds Ratios
    lr_raw = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr_raw.fit(df_train[MINIMAL_FEATURES], y_train)
    odds_ratios = []
    for idx, feat in enumerate(MINIMAL_FEATURES):
        odds_ratios.append(
            {
                "feature": feat,
                "odds_ratio": round(float(np.exp(lr_raw.coef_[0][idx])), 2),
                "coef_raw": round(float(lr_raw.coef_[0][idx]), 4),
            }
        )
    odds_ratios.sort(key=lambda x: x["odds_ratio"], reverse=True)

    # Standardized log-odds weights for single-case explanation in app
    clf_std = lr_min.named_steps["classifier"]
    # ColumnTransformer order: continuous=["BMI"], ordinal=["Age", "GenHlth"], binary=["HighBP", "HighChol", "DiffWalk", "HeartDiseaseorAttack"]
    ordered_cols = ["BMI", "Age", "GenHlth", "HighBP", "HighChol", "DiffWalk", "HeartDiseaseorAttack"]
    feature_weights = {}
    feature_means = {}
    feature_stds = {}
    for idx, feat in enumerate(ordered_cols):
        feature_weights[feat] = round(float(clf_std.coef_[0][idx]), 4)
        feature_means[feat] = round(float(df_train[feat].mean()), 4)
        if feat in ["BMI", "Age", "GenHlth"]:
            feature_stds[feat] = round(float(df_train[feat].std() + 1e-8), 4)
        else:
            feature_stds[feat] = 1.0

    # Health Equity Subgroup Check (App 7-question model vs Full model)
    equity_app = compute_subgroup_equity(df_test, y_test_arr, test_probs, optimal_threshold)
    equity_full = compute_subgroup_equity(df_test, y_test_arr, test_probs_full, optimal_threshold)

    # Save Trained Models via joblib
    print("[6/6] Saving trained Pipeline to models/pipeline.joblib and models/inydb...")
    joblib.dump(hgb_min, models_dir / "pipeline.joblib")
    joblib.dump(hgb_min, models_dir / "inydb")
    joblib.dump(lr_min, models_dir / "baseline_logreg.joblib")

    app_model_summary = [m for m in models_compared if m["model"] == "Histogram Gradient Boosting (minimal)"][0]
    lr_model_summary = [m for m in models_compared if m["model"] == "Logistic Regression (minimal)"][0]

    model_meta = {
        "project_topic": "T3: Diabetes Risk Screening (CDC BRFSS 2015)",
        "user_decision_statement": "An adult answering questions at home uses the app to decide whether to get a blood sugar test.",
        "dataset_summary": {
            "source": "CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 — diabetes_binary_health_indicators_BRFSS2015.csv",
            "n_raw": int(n_raw),
            "n_duplicates_removed": int(n_dups),
            "n_clean": int(n_clean),
            "prevalence_raw": round(prev_raw, 4),
            "prevalence_clean": round(prev_clean, 4),
            "n_train": int(len(df_train)),
            "n_val": int(len(df_val)),
            "n_test": int(len(df_test)),
        },
        "feature_list": MINIMAL_FEATURES,
        "cost_assumptions": {
            "default_cost_ratio_fn_to_fp": 5.0,
            "chosen_threshold": optimal_threshold,
            "cost_false_positive_usd": 45.0,
            "cost_false_negative_usd": 225.0,
        },
        "app_model_test_metrics": app_model_summary,
        "baseline_model_test_metrics": lr_model_summary,
        "models_compared": models_compared,
        "threshold_curve": curve_data,
        "permutation_importance": perm_list,
        "odds_ratios": odds_ratios,
        "explainability_reference": {
            "logreg_intercept": round(float(clf_std.intercept_[0]), 4),
            "feature_weights": feature_weights,
            "feature_means": feature_means,
            "feature_stds": feature_stds,
        },
        "health_equity": {
            "threshold": optimal_threshold,
            "app_model": equity_app,
            "full_model": equity_full,
        },
    }

    with open(models_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2)
    print("[SUCCESS] Saved complete model_meta.json!")
    print(json.dumps(models_compared, indent=2))


if __name__ == "__main__":
    main()
