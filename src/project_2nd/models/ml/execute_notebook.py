# -*- coding: utf-8 -*-
"""execute_notebook.py — model_comparison.ipynb를 실행하고 결과를 그대로 저장."""
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import nbformat
from nbclient import NotebookClient

NB_PATH = "model_comparison.ipynb"

print(f"[시작] {NB_PATH} 실행 중 ...")
t0 = time.time()
nb = nbformat.read(NB_PATH, as_version=4)
client = NotebookClient(nb, timeout=1800, kernel_name="python3")
try:
    client.execute()
    print(f"[완료] 총 {(time.time()-t0)/60:.1f}분 소요")
except Exception as e:
    print(f"[중단됨] {type(e).__name__}: {e}")
    print("      -> 실패 지점 이전 셀들의 출력은 그대로 저장됨 (아래 저장 경로 확인)")
finally:
    # 성공/실패와 무관하게 지금까지 실행된 셀 출력은 항상 저장 (한 셀 실패로
    # 그 전까지의 결과를 통째로 날리지 않기 위함)
    nbformat.write(nb, NB_PATH)
    print(f"저장: {NB_PATH}")
