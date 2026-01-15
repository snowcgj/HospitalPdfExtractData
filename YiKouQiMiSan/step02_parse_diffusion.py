# YiKouQiMiSa/step02_parse_diffusion.py
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


def poly_to_bbox(poly: List[List[int]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def is_number(s: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", s.strip()))


def to_float(s: str) -> Optional[float]:
    s = s.strip()
    if not is_number(s):
        return None
    try:
        return float(s)
    except Exception:
        return None


def build_tokens(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    texts = raw["rec_texts"]
    scores = raw["rec_scores"]
    polys = raw["rec_polys"]

    tokens: List[Dict[str, Any]] = []
    for text, score, poly in zip(texts, scores, polys):
        x1, y1, x2, y2 = poly_to_bbox(poly)
        tokens.append({
            "text": str(text),
            "score": float(score),
            "bbox": (x1, y1, x2, y2),
            "cx": (x1 + x2) // 2,
            "cy": (y1 + y2) // 2,
            "h": (y2 - y1),
            "w": (x2 - x1),
        })
    return tokens


def group_lines(tokens: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not tokens:
        return []

    heights = sorted([t["h"] for t in tokens if t["h"] > 0])
    median_h = heights[len(heights)//2] if heights else 20
    y_thresh = max(8, int(median_h * 0.6))

    tokens_sorted = sorted(tokens, key=lambda t: (t["cy"], t["cx"]))
    lines: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    cur_y = None

    for t in tokens_sorted:
        if cur_y is None:
            cur = [t]
            cur_y = t["cy"]
            continue
        if abs(t["cy"] - cur_y) <= y_thresh:
            cur.append(t)
            cur_y = int(cur_y * 0.8 + t["cy"] * 0.2)
        else:
            lines.append(sorted(cur, key=lambda x: x["cx"]))
            cur = [t]
            cur_y = t["cy"]

    if cur:
        lines.append(sorted(cur, key=lambda x: x["cx"]))
    return lines


def parse_patient(lines: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    patient: Dict[str, Any] = {}

    def grab_after(line: List[Dict[str, Any]], key: str) -> Optional[str]:
        for i, t in enumerate(line):
            if key in t["text"]:
                for j in range(i + 1, len(line)):
                    v = line[j]["text"].strip()
                    if v:
                        return v
        return None

    for line in lines[:30]:
        name = grab_after(line, "姓名")
        if name:
            patient["name"] = name
        sex = grab_after(line, "性别")
        if sex:
            patient["sex"] = sex
        height = grab_after(line, "身高")
        if height:
            m = re.search(r"(\d+(\.\d+)?)", height)
            if m:
                patient["height_cm"] = float(m.group(1))
        weight = grab_after(line, "体重")
        if weight:
            m = re.search(r"(\d+(\.\d+)?)", weight)
            if m:
                patient["weight_kg"] = float(m.group(1))
        idv = grab_after(line, "测试号")
        if idv and re.fullmatch(r"\d{6,}", idv):
            patient["id"] = idv

        for t in line:
            m = re.fullmatch(r"(\d+)\s*岁", t["text"].strip())
            if m:
                patient["age"] = int(m.group(1))

        for t in line:
            s = t["text"].strip()
            if re.fullmatch(r"\d{10,}", s):
                patient["test_no"] = s

    return patient


def parse_impression(lines: List[List[Dict[str, Any]]]) -> str:
    for i, line in enumerate(lines):
        joined = "".join([t["text"] for t in line])
        if "意见" in joined or "结论" in joined:
            if "：" in joined:
                after = joined.split("：", 1)[1].strip()
                if after:
                    return after
            if i + 1 < len(lines):
                nxt = "".join([t["text"] for t in lines[i+1]]).strip()
                if nxt:
                    return nxt
    return ""


def find_table_header_line(lines: List[List[Dict[str, Any]]]) -> Tuple[int, Dict[str, int]]:
    def norm(s: str) -> str:
        return s.replace("（", "(").replace("）", ")").replace("／", "/").strip()

    for idx, line in enumerate(lines):
        texts = [norm(t["text"]) for t in line]
        joined = " ".join(texts)
        if ("Pred" in texts) and ("ANG" in texts) and ("Act1" in texts) and ("Act2" in texts):
            colx: Dict[str, int] = {}
            for t in line:
                s = norm(t["text"])
                if s == "Pred":
                    colx["Pred"] = t["cx"]
                elif s == "ANG":
                    colx["ANG"] = t["cx"]
                elif s in ("ANG/P", "ANG\\P", "ANGIP"):
                    colx["ANG_P"] = t["cx"]
                elif s == "Act1":
                    colx["Act1"] = t["cx"]
                elif s == "Act2":
                    colx["Act2"] = t["cx"]

            for t in line:
                s = norm(t["text"])
                if s.startswith("%") and ("A1" in s or "a1" in s):
                    colx["Act1_P"] = t["cx"]
                if s.startswith("%") and ("A2" in s or "a2" in s):
                    colx["Act2_P"] = t["cx"]

            if "ANG_P" not in colx and "ANG/P" in joined:
                for t in line:
                    if "ANG" in t["text"] and "/" in t["text"]:
                        colx["ANG_P"] = t["cx"]
                        break

            return idx, colx

    raise RuntimeError("Table header line not found.")


def nearest_col(cx: int, colx: Dict[str, int]) -> Optional[str]:
    best_k, best_d = None, 10**18
    for k, x in colx.items():
        d = abs(cx - x)
        if d < best_d:
            best_d, best_k = d, k
    return best_k


def parse_table(lines: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    header_idx, colx = find_table_header_line(lines)
    table: List[Dict[str, Any]] = []

    for line in lines[header_idx + 1:]:
        texts = [t["text"] for t in line]
        joined = " ".join(texts)

        if ("意见" in joined) or ("报告人" in joined) or ("Date" in joined and "Time" in joined):
            break
        if len(line) < 3:
            continue

        # 找 unit 起点
        unit_start = None
        for i, t in enumerate(line):
            s = t["text"].strip()
            if s.startswith("[") or s in ("%]", "[L]", "[s]") or "mmol" in s:
                unit_start = i
                break
        if unit_start is None:
            continue

        item = " ".join([t["text"] for t in line[:unit_start]]).strip()
        if not item:
            continue

        # 合并 unit
        unit_tokens = []
        j = unit_start
        while j < len(line):
            s = line[j]["text"].strip()
            unit_tokens.append(s)
            if "]" in s or s == "%]":
                j += 1
                break
            j += 1
        unit = "".join(unit_tokens).replace("%]", "%").strip().strip("[]")

        value_tokens = line[j:]
        row: Dict[str, Any] = {"item": item, "unit": unit}

        for t in value_tokens:
            v = to_float(t["text"])
            if v is None:
                continue
            k = nearest_col(t["cx"], colx)
            if k:
                row[k] = v

        keys_present = sum(1 for k in ("Pred", "ANG", "Act1", "Act2") if k in row)
        if keys_present >= 2:
            table.append(row)

    return table


def parse_one(raw_path: Path) -> Dict[str, Any]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    tokens = build_tokens(raw)
    lines = group_lines(tokens)

    patient = parse_patient(lines)
    table = parse_table(lines)
    impression = parse_impression(lines)

    return {
        "report_type": "diffusion_single_breath",
        "patient": patient,
        "table": table,
        "impression": impression,
    }


def main():
    # ========= 你只需要改这两项 =========
    # ========= 你只需要改这两项 =========
    # ========= 你只需要改这两项 =========
    # ========= 你只需要改这两项 =========
    # ========= 你只需要改这两项 =========
    # ========= 你只需要改这两项 =========
    IN_RAW_DIR = Path("./out_stage01_raw/raw_json")
    OUT_DIR = Path("./out_stage02_parsed")
    SKIP_EXISTING = True
    # ==================================

    out_json_dir = OUT_DIR / "parsed_json"
    log_dir = OUT_DIR / "logs"
    out_json_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(IN_RAW_DIR.glob("*_raw.json"))
    if not raw_files:
        raise FileNotFoundError(f"No *_raw.json found in {IN_RAW_DIR}")

    log_path = log_dir / f"step02_{time.strftime('%Y%m%d_%H%M%S')}.log"
    ok_cnt = fail_cnt = skip_cnt = 0

    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"IN_RAW_DIR={IN_RAW_DIR}\nOUT_DIR={OUT_DIR}\nTOTAL={len(raw_files)}\n\n")
        for idx, raw_path in enumerate(raw_files, 1):
            stem = raw_path.name.replace("_raw.json", "")
            out_path = out_json_dir / f"{stem}.parsed.json"

            if SKIP_EXISTING and out_path.exists():
                skip_cnt += 1
                lf.write(f"[SKIP] {idx}/{len(raw_files)} {raw_path.name}\n")
                continue

            t0 = time.time()
            try:
                parsed = parse_one(raw_path)
                out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
                ok_cnt += 1
                lf.write(f"[OK]   {idx}/{len(raw_files)} {raw_path.name}  rows={len(parsed['table'])}  {time.time()-t0:.2f}s\n")
            except Exception as e:
                fail_cnt += 1
                lf.write(f"[FAIL] {idx}/{len(raw_files)} {raw_path.name}  {time.time()-t0:.2f}s  err={repr(e)}\n")

        lf.write(f"\nDONE ok={ok_cnt} fail={fail_cnt} skip={skip_cnt}\n")

    print(f"Step02 DONE. ok={ok_cnt} fail={fail_cnt} skip={skip_cnt}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
