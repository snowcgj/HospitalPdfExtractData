# Y-ChangGuiShuQi/step01_paddle_ocr.py
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from paddleocr import PaddleOCR

'''
运行命令：
先修改下面参数；
再CUDA_VISIBLE_DEVICES=4,5,6,7  python 文件名.py
'''

def find_rec_bundle(obj: Any) -> Optional[Dict[str, Any]]:
    """在任意嵌套结构里查找同时包含 rec_texts/rec_polys/rec_scores 的 dict"""
    if isinstance(obj, dict):
        if "rec_texts" in obj and "rec_polys" in obj and "rec_scores" in obj:
            return obj
        for v in obj.values():
            got = find_rec_bundle(v)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for it in obj:
            got = find_rec_bundle(it)
            if got is not None:
                return got
    return None


def main():
    # ========= 你只需要改这三项 =========
    # ========= 你只需要改这三项 =========
    # ========= 你只需要改这三项 =========
    # ========= 你只需要改这三项 =========
    # ========= 你只需要改这三项 =========
    IN_DIR = Path("../data/buckets/batch_001/常规通气")   # 你的 png 文件夹
    OUT_DIR = Path("./out_stage01_raw")                       # 输出根目录
    SKIP_EXISTING = True                                      # 已有 raw.json 就跳过
    # ==================================

    raw_dir = OUT_DIR / "raw_json"
    vis_dir = OUT_DIR / "vis"
    full_dir = OUT_DIR / "full_json"
    log_dir = OUT_DIR / "logs"
    for d in (raw_dir, vis_dir, full_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 让 “模型源连通性检查” 不每次都卡一下（你控制台看到的那句）
    # 如果你希望保留检查，就删掉这行
    os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        # 如果你要强制中文，可加 lang="ch"，不加也能识别
        # lang="ch",
    )

    pngs = sorted(IN_DIR.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"No png found in {IN_DIR}")

    log_path = log_dir / f"step01_{time.strftime('%Y%m%d_%H%M%S')}.log"
    ok_cnt = 0
    fail_cnt = 0
    skip_cnt = 0

    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"IN_DIR={IN_DIR}\nOUT_DIR={OUT_DIR}\nTOTAL={len(pngs)}\n\n")

        for idx, img_path in enumerate(pngs, 1):
            stem = img_path.stem
            raw_path = raw_dir / f"{stem}_raw.json"
            vis_path = vis_dir / f"{stem}_ocr_vis.png"
            full_path = full_dir / f"{stem}_full.json"

            if SKIP_EXISTING and raw_path.exists():
                skip_cnt += 1
                lf.write(f"[SKIP] {idx}/{len(pngs)} {img_path.name}\n")
                continue

            t0 = time.time()
            try:
                result = ocr.predict(input=str(img_path))
                if not result:
                    raise RuntimeError("Empty OCR result")

                res = result[0]  # 单页/单图
                # 保存可视化（可选）
                res.save_to_img(str(vis_dir))

                # 取出 dict
                j = res.json  # 你已经验证过是 dict

                # full json（可选/排错）
                full_path.write_text(json.dumps(j, ensure_ascii=False, indent=2), encoding="utf-8")

                bundle = find_rec_bundle(j)
                if bundle is None:
                    raise RuntimeError("Cannot find rec_texts/rec_polys/rec_scores in OCR json")

                raw = {
                    "rec_texts": bundle["rec_texts"],
                    "rec_scores": bundle["rec_scores"],
                    "rec_polys": bundle["rec_polys"],
                }
                raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

                dt = time.time() - t0
                ok_cnt += 1
                lf.write(f"[OK]   {idx}/{len(pngs)} {img_path.name}  tokens={len(raw['rec_texts'])}  {dt:.2f}s\n")

            except Exception as e:
                dt = time.time() - t0
                fail_cnt += 1
                lf.write(f"[FAIL] {idx}/{len(pngs)} {img_path.name}  {dt:.2f}s  err={repr(e)}\n")

        lf.write(f"\nDONE ok={ok_cnt} fail={fail_cnt} skip={skip_cnt}\n")

    print(f"Step01 DONE. ok={ok_cnt} fail={fail_cnt} skip={skip_cnt}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
