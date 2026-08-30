# -*- coding: utf-8 -*-
"""convert_report_to_pdf.py — REPORT.md -> REPORT.html -> REPORT.pdf.

xhtml2pdf/reportlab 조합은 한글(Malgun) 렌더링이 계속 실패해서(빈 화면), 대신
Edge/Chrome의 headless --print-to-pdf 기능을 쓴다. 브라우저는 OS에 설치된
'맑은 고딕' 폰트를 이름으로 바로 찾아 쓰므로 폰트 파일을 직접 다룰 필요가 없다.
"""
import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import markdown

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))  # ml_dryrun root
MD_PATH = os.path.join(BASE_DIR, "REPORT.md")
HTML_PATH = os.path.join(BASE_DIR, "REPORT.html")
PDF_PATH = os.path.join(BASE_DIR, "REPORT.pdf")

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]

with open(MD_PATH, encoding="utf-8") as f:
    md_text = f.read()

html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

# 이미지 상대경로(results/xxx.png)를 브라우저가 찾을 수 있는 file:// 절대경로로 치환
img_abs_prefix = "file:///" + os.path.join(BASE_DIR, "results").replace("\\", "/")
html_body = html_body.replace('src="results/', f'src="{img_abs_prefix}/')

html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
  body {{ font-family: "Malgun Gothic", sans-serif; font-size: 10.5pt; line-height: 1.6;
         color: #222; max-width: 900px; margin: 0 auto; padding: 20px; }}
  h1 {{ font-size: 20pt; color: #1F4E78; border-bottom: 3px solid #1F4E78; padding-bottom: 8px; }}
  h2 {{ font-size: 15pt; color: #1F4E78; margin-top: 26px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
  h3 {{ font-size: 12.5pt; color: #2E75B6; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0 16px 0; }}
  th, td {{ border: 1px solid #999; padding: 5px 8px; font-size: 9.5pt; text-align: left; }}
  th {{ background-color: #4472C4; color: white; }}
  tr:nth-child(even) {{ background-color: #f5f5f5; }}
  code {{ background-color: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-family: Consolas, monospace; }}
  pre {{ background-color: #f0f0f0; padding: 8px; overflow-x: auto; }}
  img {{ max-width: 100%; display: block; margin: 12px auto; }}
  blockquote {{ border-left: 4px solid #ccc; margin: 8px 0; padding: 4px 12px; color: #555; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html)
print(f"HTML 저장: {HTML_PATH}")

browser = next((p for p in EDGE_CANDIDATES if os.path.isfile(p)), None)
if browser is None:
    print("Edge/Chrome를 찾지 못했습니다. HTML만 생성했습니다.")
    sys.exit(1)

file_url = "file:///" + HTML_PATH.replace("\\", "/")
cmd = [
    browser,
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={PDF_PATH}",
    "--print-to-pdf-no-header",
    file_url,
]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
print("returncode:", result.returncode)
if result.stdout:
    print("stdout:", result.stdout)
if result.stderr:
    print("stderr:", result.stderr[:2000])

if os.path.isfile(PDF_PATH):
    print(f"저장 완료: {PDF_PATH} ({os.path.getsize(PDF_PATH):,} bytes)")
else:
    print("PDF 생성 실패 — 위 stderr 확인 필요")
    sys.exit(1)
