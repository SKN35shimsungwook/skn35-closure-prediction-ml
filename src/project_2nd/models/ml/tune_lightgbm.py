# -*- coding: utf-8 -*-
"""
tune_lightgbm.py

LightGBM이 이미 3개 모델 중 제일 좋았으니(ROC-AUC 0.746), 이 모델의 하이퍼파라미터를
그리드서치로 튜닝해서 더 개선되는지 확인한다. train_and_evaluate_real.py와 동일한
데이터/전처리/5-fold 구조를 그대로 재사용 (비교 가능하게 유지).

튜닝 대상: num_leaves, min_child_samples, feature_fraction (총 2x2x2=8 조합)
고정: learning_rate=0.05, n_estimators=300, scale_pos_weight=fold별 neg/pos 비율

각 조합마다 5-fold 평균 ROC-AUC로 비교 -> 제일 좋은 조합을 "최종 결과"로 보고.

출력: results/lightgbm_tuning_results.csv (조합별 전체 비교표)
"""
import sys
import time
import warnings
from itertools import product

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import lightgbm as lgb

warnings.filterwarnings("ignore")

from data_path import real_data_path

IN_PATH = real_data_path()
OUT_PATH = "results/lightgbm_tuning_results.csv"

TARGET = "is_closed_next"
FOLD_COL = "fold"
EXCLUDE_COLS = {"snapshot_date", "store_id", "fold", TARGET}
CATEGORICAL_COLS = ["industry_dae_code", "industry_group", "industry_jung_code",
                    "industry_code", "gu_name", "floor_category"]

# 튜닝 그리드 (커리큘럼에서 배운 itertools.product 그리드서치 패턴 그대로)
NUM_LEAVES_GRID = [31, 63]
MIN_CHILD_SAMPLES_GRID = [20, 50]
FEATURE_FRACTION_GRID = [0.8, 1.0]


def load_data(path):
    df = pd.read_csv(path, dtype={"snapshot_date": str, "store_id": str})
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS
                    and c not in ("industry_jung_name", "industry_name")]
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    for c in CATEGORICAL_COLS:
        df[c] = df[c].fillna("결측")
        df[c] = df[c].astype("category")
    return df, feature_cols, numeric_cols


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


def run_one_config(df, feature_cols, num_leaves, min_child_samples, feature_fraction):
    fold_ids = sorted(df[FOLD_COL].unique())
    fold_metrics = []
    t0 = time.time()

    for k in fold_ids:
        train_df = df[df[FOLD_COL] != k]
        test_df = df[df[FOLD_COL] == k]
        y_train = train_df[TARGET].values
        y_test = test_df[TARGET].values

        pos = y_train.sum()
        neg = len(y_train) - pos
        scale_pos_weight = neg / max(pos, 1)

        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            feature_fraction=feature_fraction,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbosity=-1,
        )
        model.fit(train_df[feature_cols], y_train, categorical_feature=CATEGORICAL_COLS)
        y_prob = model.predict_proba(test_df[feature_cols])[:, 1]
        fold_metrics.append(eval_fold(y_test, y_prob))

    elapsed_min = (time.time() - t0) / 60
    metrics_df = pd.DataFrame(fold_metrics)
    summary = metrics_df.mean().to_dict()
    summary["학습시간(분)"] = elapsed_min
    return summary


def main():
    print(f"[1/2] 데이터 로드: {IN_PATH}")
    df, feature_cols, numeric_cols = load_data(IN_PATH)
    print(f"      rows={len(df):,}  features={len(feature_cols)}  "
          f"positive_rate={df[TARGET].mean():.4f}")

    grid = list(product(NUM_LEAVES_GRID, MIN_CHILD_SAMPLES_GRID, FEATURE_FRACTION_GRID))
    print(f"[2/2] 그리드서치: 총 {len(grid)}개 조합 x 5-fold\n")

    results = []
    for i, (num_leaves, min_child_samples, feature_fraction) in enumerate(grid, 1):
        print(f"  [{i}/{len(grid)}] num_leaves={num_leaves}, "
              f"min_child_samples={min_child_samples}, feature_fraction={feature_fraction}")
        summary = run_one_config(df, feature_cols, num_leaves, min_child_samples, feature_fraction)
        summary.update({
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "feature_fraction": feature_fraction,
        })
        results.append(summary)
        print(f"        ROC-AUC={summary['ROC-AUC']:.4f}  PR-AUC={summary['PR-AUC']:.4f}  "
              f"F1={summary['F1']:.4f}  Brier={summary['Brier']:.4f}  "
              f"({summary['학습시간(분)']:.2f}분)")

    result_df = pd.DataFrame(results)
    col_order = ["num_leaves", "min_child_samples", "feature_fraction",
                 "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "Brier", "학습시간(분)"]
    result_df = result_df[col_order].sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

    print("\n=== 전체 조합 비교 (ROC-AUC 내림차순) ===")
    print(result_df.to_string(index=False))

    best = result_df.iloc[0]
    print("\n=== 최고 조합 (기본값 대비) ===")
    print(f"  num_leaves={int(best['num_leaves'])}, "
          f"min_child_samples={int(best['min_child_samples'])}, "
          f"feature_fraction={best['feature_fraction']}")
    print(f"  ROC-AUC={best['ROC-AUC']:.4f}  (기본값 num_leaves=31 결과: "
          f"{result_df[(result_df.num_leaves==31)&(result_df.min_child_samples==20)&(result_df.feature_fraction==1.0)]['ROC-AUC'].values})")
    print(f"  기존 결과(train_and_evaluate_real.py, 기본 파라미터): ROC-AUC=0.7461, F1=0.3287, PR-AUC=0.3760, Brier=0.1900")

    result_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
