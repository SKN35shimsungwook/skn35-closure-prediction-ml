# -*- coding: utf-8 -*-
"""
train_and_evaluate.py

목적: "정확도 확보"가 아니라 "파이프라인이 끝까지 에러 없이 도는지 검증".
data/features/modeling_dataset_fake.csv (가짜 데이터, 실제 스키마 흉내)를 읽어서
  인코딩 -> store_id 기반 5-fold 교차검증 -> LogisticRegression/RandomForest/LightGBM
  3개 모델 학습 -> PR-AUC 등 평가지표 계산
까지 실제 코드 경로를 그대로 돌려본다.

실제 데이터가 준비되면 IN_PATH만 실데이터 경로로 바꾸면 된다 (컬럼명이 같다면
나머지 코드는 그대로 재사용 가능하도록 설계했다).

출력: results/benchmark_results.csv (계획표 '벤치마크' 시트에 옮겨 적을 수 있는 형식)
"""
import sys
import time
import warnings

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import lightgbm as lgb

warnings.filterwarnings("ignore")

IN_PATH = "data/features/modeling_dataset_fake.csv"
OUT_PATH = "results/benchmark_results.csv"

TARGET = "is_closed_next"
FOLD_COL = "fold"

# 학습에서 제외할 컬럼 — 식별자, 라벨 시점 정보, 정답 직접 노출 컬럼
EXCLUDE_COLS = {
    "snapshot_date", "store_id", "fold", TARGET,
    "transitioned_next",  # 폐업 이후 시점 정보라 누수 가능성 -> 제외 (DL팀과 동일 기준)
}
CATEGORICAL_COLS = ["industry_group", "gu_name", "floor_category"]


def load_data(path):
    df = pd.read_csv(path, dtype={"snapshot_date": str, "store_id": str})
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    return df, feature_cols, numeric_cols


def make_logreg_pipeline(numeric_cols, categorical_cols):
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def make_rf_pipeline(numeric_cols, categorical_cols):
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def prep_for_lightgbm(df, categorical_cols):
    df = df.copy()
    for c in categorical_cols:
        df[c] = df[c].astype("category")
    return df


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


def run_cv(name, df, feature_cols, numeric_cols, categorical_cols, model_kind):
    fold_ids = sorted(df[FOLD_COL].unique())
    fold_metrics = []
    t0 = time.time()

    for k in fold_ids:
        train_df = df[df[FOLD_COL] != k]
        test_df = df[df[FOLD_COL] == k]
        y_train = train_df[TARGET].values
        y_test = test_df[TARGET].values

        if model_kind == "logreg":
            model = make_logreg_pipeline(numeric_cols, categorical_cols)
            model.fit(train_df[feature_cols], y_train)
            y_prob = model.predict_proba(test_df[feature_cols])[:, 1]

        elif model_kind == "rf":
            model = make_rf_pipeline(numeric_cols, categorical_cols)
            model.fit(train_df[feature_cols], y_train)
            y_prob = model.predict_proba(test_df[feature_cols])[:, 1]

        elif model_kind == "lightgbm":
            train_lgb = prep_for_lightgbm(train_df[feature_cols], categorical_cols)
            test_lgb = prep_for_lightgbm(test_df[feature_cols], categorical_cols)
            pos = y_train.sum()
            neg = len(y_train) - pos
            scale_pos_weight = neg / max(pos, 1)
            model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                verbosity=-1,
            )
            model.fit(train_lgb, y_train, categorical_feature=categorical_cols)
            y_prob = model.predict_proba(test_lgb)[:, 1]

        else:
            raise ValueError(model_kind)

        fold_metrics.append(eval_fold(y_test, y_prob))

    elapsed_min = (time.time() - t0) / 60
    metrics_df = pd.DataFrame(fold_metrics)
    summary = metrics_df.mean().to_dict()
    summary["안정성(0~100)"] = float(
        max(0.0, 100 - metrics_df["PR-AUC"].std() * 500)
    )  # fold간 PR-AUC 표준편차가 작을수록 높은 점수 (임시 산식)
    summary["학습시간(분)"] = elapsed_min
    summary["모델"] = name
    return summary, metrics_df


def main():
    print(f"[1/3] 데이터 로드: {IN_PATH}")
    df, feature_cols, numeric_cols = load_data(IN_PATH)
    print(f"      rows={len(df):,}  features={len(feature_cols)}  "
          f"positive_rate={df[TARGET].mean():.4f}  folds={sorted(df[FOLD_COL].unique())}")

    print("[2/3] 3개 모델 학습 (5-fold 교차검증) ...")
    results = []
    for name, kind in [
        ("LogisticRegression", "logreg"),
        ("RandomForestClassifier", "rf"),
        ("LightGBM", "lightgbm"),
    ]:
        print(f"      - {name} 학습 중...")
        summary, fold_df = run_cv(name, df, feature_cols, numeric_cols, CATEGORICAL_COLS, kind)
        results.append(summary)
        print(
            f"        PR-AUC={summary['PR-AUC']:.4f}  ROC-AUC={summary['ROC-AUC']:.4f}  "
            f"F1={summary['F1']:.4f}  Brier={summary['Brier']:.4f}  "
            f"({summary['학습시간(분)']:.2f}분)"
        )

    result_df = pd.DataFrame(results)
    col_order = ["모델", "PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "Brier",
                 "안정성(0~100)", "학습시간(분)"]
    result_df = result_df[col_order]

    print("\n[3/3] 결과 요약 (가짜 데이터 — 수치 자체는 무의미, 코드 정상 동작 확인용)")
    print(result_df.to_string(index=False))

    result_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUT_PATH}")
    print("파이프라인 끝까지 에러 없이 실행됨 — 실제 데이터로 IN_PATH만 바꾸면 재사용 가능.")


if __name__ == "__main__":
    main()
