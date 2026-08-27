# -*- coding: utf-8 -*-
"""
generate_fake_dataset.py

파이프라인 코드 검증용 가짜 데이터 생성기.
build_modeling_dataset.py의 최종 컬럼 구성(main 브랜치 기준)을 그대로 흉내낸다.
실제 자치구명/업종명 등은 팀 실데이터와 다를 수 있음 — 컬럼명·타입·값 범위·불균형
비율만 맞춘 더미이며, 여기서 나온 성능 수치는 참고용이 아니라 "코드가 끝까지
도는지" 검증용이다.

출력: data/features/modeling_dataset_fake.csv
"""
import hashlib

import numpy as np
import pandas as pd

OUT_PATH = "data/features/modeling_dataset_fake.csv"

N_STORES = 4000
ORDER = ["202312", "202406", "202412", "202506", "202512", "202606"]
INDUSTRY_GROUPS = ["외식업", "소매업", "생활서비스", "수리업", "교육서비스", "오락서비스", "숙박업"]
GU_LIST = [f"임시구{i:02d}" for i in range(1, 26)]  # 실제 자치구명 아님 — 더미 라벨
FLOOR_CATS = ["1층", "2층이상", "지하"]

rng = np.random.default_rng(42)


def fold_of(store_id, k=5):
    h = hashlib.md5(store_id.encode()).hexdigest()
    return int(h, 16) % k


def build():
    rows = []
    for i in range(N_STORES):
        store_id = f"S{i:06d}"
        industry_group = rng.choice(INDUSTRY_GROUPS)
        gu_name = rng.choice(GU_LIST)
        floor_category = rng.choice(FLOOR_CATS, p=[0.6, 0.3, 0.1])
        base_lng = rng.uniform(126.76, 127.18)
        base_lat = rng.uniform(37.42, 37.70)

        # fold-safe historical rate를 흉내낸 값 (실제로는 fold별로 재계산되어야 하지만
        # 더미 데이터에서는 store 단위 고정값으로 근사)
        industry_hist_rate = float(np.clip(rng.beta(2, 15), 0, 1))
        dong_hist_rate = float(np.clip(rng.beta(2, 15), 0, 1))
        dong_industry_hist_rate = float(
            np.clip((industry_hist_rate + dong_hist_rate) / 2 + rng.normal(0, 0.03), 0, 1)
        )

        korean_pop = rng.uniform(1000, 40000)
        foreign_long_pop = rng.uniform(0, 2000)
        foreign_short_pop = rng.uniform(0, 3000)
        total_pop_avg = korean_pop + foreign_long_pop + foreign_short_pop
        foreign_short_ratio = foreign_short_pop / total_pop_avg
        tourist_zone_candidate = int(rng.random() < 0.1)
        population_is_proxied = int(rng.random() < 0.05)

        n_snap = int(rng.integers(1, 7))
        snap_idx_choices = sorted(rng.choice(len(ORDER), size=n_snap, replace=False))
        first_idx = snap_idx_choices[0]

        for snap_idx in snap_idx_choices:
            snap = ORDER[snap_idx]
            store_age_months = (snap_idx - first_idx) * 6
            same_industry_count_300m = int(rng.poisson(6))
            total_count_300m = same_industry_count_300m + int(rng.poisson(30))
            nearest_same_industry_distance_m = float(rng.exponential(80) + 5)
            dong_industry_count = int(rng.poisson(25))
            coord_cluster_size = int(rng.poisson(60) + 5)
            previously_transitioned = int(rng.random() < 0.05)
            keyword_growth_score = float(
                (rng.random() < 0.08) * rng.uniform(0.1, 2.0)
            )
            transitioned_next = int(rng.random() < 0.03)

            # 약한 신호를 섞은 가짜 폐업확률 — 실제 신호 강도와 무관, 파이프라인이
            # "뭔가 학습은 되는지"를 확인하려고 임의로 넣은 상관관계일 뿐이다.
            logit = (
                -2.2
                + 2.5 * industry_hist_rate
                + 2.0 * dong_hist_rate
                + 1.5 * dong_industry_hist_rate
                - 0.15 * np.log1p(total_pop_avg / 1000)
                + 0.4 * (store_age_months < 12)
                - 0.3 * keyword_growth_score
                + rng.normal(0, 0.6)
            )
            p_close = 1 / (1 + np.exp(-logit))
            is_closed_next = int(rng.random() < p_close)

            rows.append(
                dict(
                    snapshot_date=snap,
                    store_id=store_id,
                    industry_group=industry_group,
                    gu_name=gu_name,
                    floor_category=floor_category,
                    lng=base_lng + rng.normal(0, 0.001),
                    lat=base_lat + rng.normal(0, 0.001),
                    same_industry_count_300m=same_industry_count_300m,
                    total_count_300m=total_count_300m,
                    nearest_same_industry_distance_m=nearest_same_industry_distance_m,
                    dong_industry_count=dong_industry_count,
                    coord_cluster_size=coord_cluster_size,
                    store_age_months=store_age_months,
                    previously_transitioned=previously_transitioned,
                    keyword_growth_score=keyword_growth_score,
                    korean_pop=korean_pop,
                    foreign_long_pop=foreign_long_pop,
                    foreign_short_pop=foreign_short_pop,
                    total_pop_avg=total_pop_avg,
                    foreign_short_ratio=foreign_short_ratio,
                    tourist_zone_candidate=tourist_zone_candidate,
                    population_is_proxied=population_is_proxied,
                    industry_historical_rate=industry_hist_rate,
                    dong_historical_rate=dong_hist_rate,
                    dong_industry_historical_rate=dong_industry_hist_rate,
                    transitioned_next=transitioned_next,
                    fold=fold_of(store_id),
                    is_closed_next=is_closed_next,
                )
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build()
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"saved {OUT_PATH}")
    print(f"rows={len(df):,}  stores={df.store_id.nunique():,}  "
          f"positive_rate={df.is_closed_next.mean():.4f}")
    print("fold distribution:")
    print(df.fold.value_counts().sort_index())
