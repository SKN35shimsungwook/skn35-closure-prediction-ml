# -*- coding: utf-8 -*-
"""
train_catboost.py

새 모델 후보: CatBoost. LightGBM처럼 Gradient Boosting 계열이지만
  - 범주형을 원핫 없이 그대로 받음(cat_features로 컬럼 지정만 하면 됨)
  - ordered boosting으로 타겟 인코딩류 피처의 누수를 자체적으로 줄여줌
  - 기본값만으로도 꽤 잘 나오는 편이라 튜닝 부담이 적음
train_and_evaluate_real.py와 동일한 데이터/5-fold 구조로 비교 가능하게 맞춤.

출력: results/benchmark_catboost.csv
"""
import sys
import time
import warnings

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

from data_path import real_data_path

IN_PATH = real_data_path()
OUT_PATH = "results/benchmark_catboost.csv"

TARGET = "is_closed_next"
FOLD_COL = "fold"
EXCLUDE_COLS = {"snapshot_date", "store_id", "fold", TARGET}
CATEGORICAL_COLS = ["industry_dae_code", "industry_group", "industry_jung_code",
                    "industry_code", "gu_name", "floor_category"]


def load_data(path):
    df = pd.read_csv(path, dtype={"snapshot_date": str, "store_id": str})
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS
                    and c not in ("industry_jung_name", "industry_name")]
    for c in CATEGORICAL_COLS:
        df[c] = df[c].fillna("결측").astype(str)
    return df, feature_cols


def eval_fold(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "PR-AUC": average_precision_score(y_true, y_prob),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Brier": brier_score_loss(y_true, y_prob),
    }


def main():
    print(f"[1/2] 데이터 로드: {IN_PATH}")
    df, feature_cols = load_data(IN_PATH)
    cat_idx = [feature_cols.index(c) for c in CATEGORICAL_COLS]
    print(f"      rows={len(df):,}  features={len(feature_cols)}  "
          f"positive_rate={df[TARGET].mean():.4f}")

    print("[2/2] CatBoost 5-fold 교차검증 ...")
    fold_ids = sorted(df[FOLD_COL].unique())
    fold_metrics = []
    t0 = time.time()

    for k in fold_ids:
        train_df = df[df[FOLD_COL] != k]
        test_df = df[df[FOLD_COL] == k]
        y_train = train_df[TARGET].values
        y_test = test_df[TARGET].values

        model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            auto_class_weights="Balanced",  # scale_pos_weight와 동일한 목적, CatBoost식 표기
            cat_features=cat_idx,
            random_seed=42,
            verbose=False,
        )
        model.fit(train_df[feature_cols], y_train)
        y_prob = model.predict_proba(test_df[feature_cols])[:, 1]
        m = eval_fold(y_test, y_prob)
        fold_metrics.append(m)
        print(f"        fold {k} 완료  ROC-AUC={m['ROC-AUC']:.4f}  F1={m['F1']:.4f}")

    elapsed_min = (time.time() - t0) / 60
    metrics_df = pd.DataFrame(fold_metrics)
    summary = metrics_df.mean().to_dict()
    summary["학습시간(분)"] = elapsed_min
    summary["모델"] = "CatBoost"

    result_df = pd.DataFrame([summary])
    col_order = ["모델", "PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "Brier", "학습시간(분)"]
    result_df = result_df[col_order]

    print("\n=== CatBoost 최종 결과 (전체 189만행, 5-fold 평균) ===")
    print(result_df.to_string(index=False))
    print("\n비교 참고 — LightGBM(기본 파라미터): ROC-AUC=0.7461, F1=0.3287, PR-AUC=0.3760, Brier=0.1900, 2.35분")

    result_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
