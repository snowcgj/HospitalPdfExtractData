# Y-ChangGuiShuQi/step02_parse_ventilation.py
# 常规通气 Step02：raw_json -> parsed_json
#
# 设计目标（按你要求）：
# 1) 只要识别出“表格行”(item + unit)，无论是否抓到数值，都入表
# 2) 不用数值大小做启发式分列
# 3) 解决 Best / %(Best) 很近、甚至表头粘连的问题：用“数字右对齐(x2)”学习列并分列
# 4) 学列时只从“含 unit 的行”采样，避免把图表坐标数字学进去（防 IndexError/列爆炸）
#
# 注意：
# - 报告里列名是 "%(Best)"，工程内部字段名用 "Best_P"（Percent）更干净
#   如果你坚持 JSON 里也要 "%(Best)"，把 field_order_all 中 "Best_P" 改成 "%(Best)" 即可。

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


# ---------------------------
# 基础工具
# ---------------------------
def poly_to_bbox(poly: List[List[int]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def is_number(s: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", (s or "").strip()))


def to_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not is_number(s):
        return None
    try:
        return float(s)
    except Exception:
        return None


def norm_text(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("（", "(").replace("）", ")").replace("／", "/").replace("％", "%")
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_item(item: str) -> str:
    """
    让 item 更稳定：
    - 合并多余空格
    - 合并字母+数字间空格：FEV 1 -> FEV1, MEF 75 -> MEF75
    - 合并 % 和 / 周围空格：FEV 1 % FVC -> FEV1%FVC, 75 / 25 -> 75/25
    """
    s = norm_text(item)
    s = re.sub(r"\s*%\s*", "%", s)
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"([A-Za-z])\s+(\d)", r"\1\2", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------
# Token / Line
# ---------------------------
def build_tokens(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    texts = raw["rec_texts"]
    scores = raw["rec_scores"]
    polys = raw["rec_polys"]

    tokens: List[Dict[str, Any]] = []
    for text, score, poly in zip(texts, scores, polys):
        x1, y1, x2, y2 = poly_to_bbox(poly)
        tokens.append(
            {
                "text": str(text),
                "score": float(score),
                "bbox": (x1, y1, x2, y2),
                "cx": (x1 + x2) // 2,
                "cy": (y1 + y2) // 2,
                "h": (y2 - y1),
                "w": (x2 - x1),
            }
        )
    return tokens


def group_lines(tokens: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not tokens:
        return []

    heights = sorted([t["h"] for t in tokens if t["h"] > 0])
    median_h = heights[len(heights) // 2] if heights else 20
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


# ---------------------------
# Patient / Impression
# ---------------------------
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

    for line in lines[:50]:
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

        # 测试号
        test_no = grab_after(line, "测试号")
        if test_no and re.fullmatch(r"\d{6,}", test_no):
            patient["test_no"] = test_no

        # ID号
        idno = grab_after(line, "ID号")
        if idno and re.fullmatch(r"\d{6,}", idno):
            patient["id"] = idno

        # 年龄
        for t in line:
            m = re.fullmatch(r"(\d+)\s*岁", t["text"].strip())
            if m:
                patient["age"] = int(m.group(1))

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
                nxt = "".join([t["text"] for t in lines[i + 1]]).strip()
                if nxt:
                    return nxt
    return ""


# ---------------------------
# Header line
# ---------------------------
def find_table_header_line(lines: List[List[Dict[str, Any]]]) -> int:
    """
    只负责定位表头行 index，不负责分列。
    允许 OCR 把 "Best %(Best" 粘连在一起。
    """
    for idx, line in enumerate(lines):
        texts = [norm_text(t["text"]) for t in line]

        has_pred = any(t == "Pred" for t in texts)
        # Best 可能是 "Best %(Best" / "Best%(Best" / "Best"
        has_best = any(t == "Best" or t.startswith("Best") or ("Best" in t) for t in texts)
        has_act = any(t.startswith("Act") for t in texts)  # Act1/Act2...

        if has_pred and has_best and has_act:
            return idx

    raise RuntimeError("Ventilation table header line not found.")


# ---------------------------
# 列学习：用数字 token 的右边界 x2（右对齐）
# ---------------------------
def line_has_unit(line: List[Dict[str, Any]]) -> bool:
    """
    判断这一行像不像“表格行”：必须出现单位 token
    这能过滤掉上方图表坐标数字行，从根源防列爆炸。
    """
    for t in line:
        s = (t["text"] or "").strip()
        if s.startswith("["):
            return True
        if s in ("%]", "[L]", "[s]"):
            return True
        if "mmol" in s.lower():
            return True
        if "/min" in s:
            return True
    return False


def learn_value_columns_by_right_edge(lines: List[List[Dict[str, Any]]], start_idx: int, max_lines: int = 60) -> List[int]:
    """
    从 header 下面若干行里学习“数值列”的右边界 x2。
    只采样含 unit 的行，避免图表坐标数字污染。
    返回：从左到右排序的一组列右边界 x2（每个代表一列）
    """
    xs: List[int] = []

    for line in lines[start_idx : start_idx + max_lines]:
        joined = " ".join([t["text"] for t in line])

        if ("意见" in joined) or ("报告人" in joined) or (("Date" in joined) and ("Time" in joined)):
            break

        if not line_has_unit(line):
            continue

        for t in line:
            if to_float(t["text"]) is not None:
                xs.append(t["bbox"][2])  # x2

    if not xs:
        return []

    xs.sort()

    # 1D 聚类阈值：你这类 300dpi 大图，16 像素通常够
    thresh = 16
    clusters: List[List[int]] = []
    cur = [xs[0]]
    for x in xs[1:]:
        if abs(x - cur[-1]) <= thresh:
            cur.append(x)
        else:
            clusters.append(cur)
            cur = [x]
    clusters.append(cur)

    cols = []
    for c in clusters:
        c = sorted(c)
        cols.append(c[len(c) // 2])

    return sorted(set(cols))


def make_assignment_threshold(col_right: List[int]) -> int:
    """
    根据列间距自适应一个“最大允许距离”，避免数字被硬贴到错列。
    """
    if len(col_right) < 2:
        return 35
    gaps = [col_right[i + 1] - col_right[i] for i in range(len(col_right) - 1)]
    gaps_sorted = sorted(gaps)
    gap_med = gaps_sorted[len(gaps_sorted) // 2]
    return max(12, int(gap_med * 0.55))


# ---------------------------
# 表格解析
# ---------------------------
def parse_table(lines: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    header_idx = find_table_header_line(lines)

    col_right = learn_value_columns_by_right_edge(lines, header_idx + 1, max_lines=80)
    if not col_right:
        raise RuntimeError("Cannot learn value columns (right-edge) from table body.")

    # 报告列名 "%(Best)" -> 工程内部字段 "Best_P"
    field_order_all = ["Pred", "Best", "Best_P", "Act1", "Act2", "Act3", "Act4", "Act5", "Act6"]

    # 多出来的列基本是噪声（或者是一些额外区域数字），只取最左边 N 列
    if len(col_right) > len(field_order_all):
        col_right = col_right[: len(field_order_all)]

    field_order = field_order_all[: len(col_right)]
    max_d = make_assignment_threshold(col_right)

    table: List[Dict[str, Any]] = []

    for line in lines[header_idx + 1 :]:
        texts = [t["text"] for t in line]
        joined = " ".join(texts)

        if ("意见" in joined) or ("报告人" in joined) or (("Date" in joined) and ("Time" in joined)):
            break

        if len(line) < 2:
            continue

        # 表格行判定：必须有 unit；否则跳过（进一步过滤非表格区域）
        if not line_has_unit(line):
            continue

        # 找 unit 起点
        unit_start = None
        for i, t in enumerate(line):
            s = (t["text"] or "").strip()
            if s.startswith("[") or s in ("%]", "[L]", "[s]") or "mmol" in s.lower() or "/min" in s:
                unit_start = i
                break
        if unit_start is None:
            continue

        raw_item = " ".join([t["text"] for t in line[:unit_start]]).strip()
        if not raw_item:
            continue
        item = normalize_item(raw_item)

        # 合并 unit tokens
        unit_tokens = []
        j = unit_start
        while j < len(line):
            s = (line[j]["text"] or "").strip()
            unit_tokens.append(s)
            if "]" in s or s == "%]":
                j += 1
                break
            j += 1

        unit_joined = "".join(unit_tokens).replace("%]", "%").strip()
        unit = unit_joined.strip().strip("[]")

        row: Dict[str, Any] = {"item": item, "unit": unit}

        # 值 tokens
        value_tokens = line[j:]
        for t in value_tokens:
            v = to_float(t["text"])
            if v is None:
                continue

            x2 = t["bbox"][2]

            # 找最近列（按右边界）
            best_i, best_d = None, 10**18
            for i, xr in enumerate(col_right):
                d = abs(x2 - xr)
                if d < best_d:
                    best_d, best_i = d, i

            if best_i is None or best_d > max_d:
                continue

            # 永久防炸：即使学习列异常，也不会越界
            if best_i >= len(field_order):
                continue

            k = field_order[best_i]
            row[k] = v

        # ✅ 你的需求：只要有 item，就入表（不管有没有数值）
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
        "report_type": "ventilation_routine",
        "patient": patient,
        "table": table,
        "impression": impression,
    }


def main():
    IN_RAW_DIR = Path("./out_stage01_raw/raw_json")
    OUT_DIR = Path("./out_stage02_parsed")
    SKIP_EXISTING = True

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
                lf.write(
                    f"[OK]   {idx}/{len(raw_files)} {raw_path.name}  rows={len(parsed['table'])}  {time.time()-t0:.2f}s\n"
                )
            except Exception as e:
                fail_cnt += 1
                lf.write(f"[FAIL] {idx}/{len(raw_files)} {raw_path.name}  {time.time()-t0:.2f}s  err={repr(e)}\n")

        lf.write(f"\nDONE ok={ok_cnt} fail={fail_cnt} skip={skip_cnt}\n")

    print(f"Step02 DONE. ok={ok_cnt} fail={fail_cnt} skip={skip_cnt}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
