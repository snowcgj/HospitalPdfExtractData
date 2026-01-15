import os
import json
import re
from pathlib import Path
from collections import defaultdict

'''
step02会用到， step02负责根据这个信息把图片放到正确的桶里。
在 data/buckets  文件夹下面生成一个 buckets/  文件夹，下面不同文件夹 存放不同类型报告文件

'''



PROJECT_ROOT = Path("/home/test/ALabFile/CommonProject/HospitalPdf")
BATCH = "batch_001"

PASS1_JSONL = PROJECT_ROOT / "data" / "jsonl" / BATCH / "pass1.jsonl"
BUCKET_DIR  = PROJECT_ROOT / "data" / "buckets" / BATCH          # 可视化桶目录（symlink）
OUT_MAP_JSON = PROJECT_ROOT / "data" / "jsonl" / BATCH / "buckets_map.json"
OUT_SUMMARY  = PROJECT_ROOT / "data" / "jsonl" / BATCH / "buckets_summary.txt"

def safe_name(s: str, maxlen: int = 80) -> str:
    s = s.strip()
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)   # windows非法字符也顺手处理
    s = re.sub(r"\s+", "_", s)
    return s[:maxlen] if len(s) > maxlen else s

def main():
    if not PASS1_JSONL.exists():
        raise FileNotFoundError(PASS1_JSONL)

    buckets = defaultdict(list)

    with open(PASS1_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            title = (obj.get("title") or "").strip()
            kind = (obj.get("kind_guess") or "").strip()
            key = title if title else (kind if kind else "UNKNOWN")
            buckets[key].append(obj["image_path"])

    # 1) 输出桶映射（后续pass2直接读这个就行）
    OUT_MAP_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MAP_JSON, "w", encoding="utf-8") as f:
        json.dump(buckets, f, ensure_ascii=False, indent=2)

    # 2) 输出一个人类可读的 summary
    items = sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        for k, v in items:
            f.write(f"[{len(v):>3}] {k}\n")
            for p in v[:5]:
                f.write(f"      - {p}\n")
            if len(v) > 5:
                f.write(f"      ... (+{len(v)-5} more)\n")
            f.write("\n")

    print("Buckets summary:")
    for k, v in items:
        print(f"  {len(v):>3}  {k}")

    # 3) 可选：生成桶文件夹（用软链接，不搬动原图）
    BUCKET_DIR.mkdir(parents=True, exist_ok=True)
    for k, paths in items:
        folder = BUCKET_DIR / safe_name(k)
        folder.mkdir(parents=True, exist_ok=True)
        for p in paths:
            src = Path(p)
            pdf_dir = src.parent.name   # 原PDF文件夹名
            link_name = f"{pdf_dir}__{src.name}"
            link = folder / link_name
            if link.exists():
                continue
            try:
                os.symlink(src, link)  # Linux上OK
            except Exception:
                # 如果文件系统不支持symlink，就跳过（仍然有 buckets_map.json 可用）
                pass

    print(f"\nWrote: {OUT_MAP_JSON}")
    print(f"Wrote: {OUT_SUMMARY}")
    print(f"Bucket folders (symlink): {BUCKET_DIR}")

if __name__ == "__main__":
    main()
