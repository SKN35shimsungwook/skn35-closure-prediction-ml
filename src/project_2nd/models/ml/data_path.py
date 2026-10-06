# -*- coding: utf-8 -*-
"""
실데이터(modeling_dataset_refined_pjw.csv) 경로를 한 곳에서 정한다.

기본 위치는 저장소 안의 data/features/ 이고, 다른 곳에 두었다면
환경 변수 CLOSURE_DATA_PATH 로 지정한다. 스크립트는 저장소 루트에서 실행한다.

    CLOSURE_DATA_PATH=D:/data/modeling_dataset_refined_pjw.csv python src/project_2nd/models/ml/tune_lightgbm.py
"""
import os
import sys
from pathlib import Path

DEFAULT_PATH = Path("data/features/modeling_dataset_refined_pjw.csv")


def real_data_path() -> str:
    path = Path(os.environ.get("CLOSURE_DATA_PATH", DEFAULT_PATH))
    if not path.exists():
        sys.exit(
            f"실데이터를 찾을 수 없습니다: {path}\n"
            "팀 프로젝트 전처리본(modeling_dataset_refined_pjw.csv)을 위 경로에 두거나 "
            "환경 변수 CLOSURE_DATA_PATH로 위치를 지정하세요.\n"
            "데이터 없이 파이프라인만 확인하려면 README의 '재현 방법 > 가짜 데이터로 실행'을 따르세요."
        )
    return str(path)
