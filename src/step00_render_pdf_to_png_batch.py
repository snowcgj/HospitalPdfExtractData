import os
import json
import time
from pathlib import Path

import fitz  # PyMuPDF


'''
2026/01/01  @author chatgpt ，sxj
description: 将一个文件夹下面所有的pdf文件【包括单页情况】，转换为 png 文件 

生成
data/pages_png 文件夹

'''

PROJECT_ROOT = Path("/home/test/ALabFile/CommonProject/HospitalPdf")  # 你按实际路径改一下
BATCH = "batch_001"

PDF_DIR = PROJECT_ROOT / "data" / "pdfs" / BATCH
OUT_DIR = PROJECT_ROOT / "data" / "pages_png" / BATCH
LOG_DIR = PROJECT_ROOT / "data" / "logs" / BATCH
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "render_pages.jsonl"

# 渲染分辨率：建议先用 2.0（约144dpi），清晰度和速度比较均衡
ZOOM = 6.0


def log_json(obj: dict):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def render_one_pdf(pdf_path: Path):
    pdf_name = pdf_path.stem
    out_subdir = OUT_DIR / pdf_name
    out_subdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log_json({
            "ok": False,
            "stage": "open_pdf",
            "pdf_path": str(pdf_path),
            "error": f"{type(e).__name__}: {e}",
        })
        return

    page_count = doc.page_count
    log_json({
        "ok": True,
        "stage": "start_pdf",
        "pdf_path": str(pdf_path),
        "pdf_name": pdf_name,
        "page_count": page_count,
    })

    for i in range(page_count):
        # 页码从 1 开始命名：p0001.png
        out_png = out_subdir / f"p{i+1:04d}.png"
        if out_png.exists() and out_png.stat().st_size > 0:
            continue  # 断点续跑：存在就跳过

        try:
            page = doc.load_page(i)
            mat = fitz.Matrix(ZOOM, ZOOM)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(out_png)
        except Exception as e:
            log_json({
                "ok": False,
                "stage": "render_page",
                "pdf_path": str(pdf_path),
                "page_index": i,
                "out_png": str(out_png),
                "error": f"{type(e).__name__}: {e}",
            })
            continue

    doc.close()

    log_json({
        "ok": True,
        "stage": "finish_pdf",
        "pdf_path": str(pdf_path),
        "elapsed_sec": round(time.time() - t0, 3),
    })


def main():
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF_DIR not found: {PDF_DIR}")

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} pdf(s) in {PDF_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for idx, pdf_path in enumerate(pdfs, 1):
        print(f"[{idx}/{len(pdfs)}] Rendering: {pdf_path.name}")
        render_one_pdf(pdf_path)

    print(f"Done. Output: {OUT_DIR}")
    print(f"Log: {LOG_FILE}")


if __name__ == "__main__":
    main()
