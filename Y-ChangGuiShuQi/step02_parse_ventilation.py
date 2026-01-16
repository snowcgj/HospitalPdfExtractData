# Y-ChangGuiShuQi/step02_parse_ventilation.py
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


# -----------------------------
# 基础工具
# -----------------------------
def poly_to_bbox(poly: List[List[int]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def is_number(s: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", s.strip()))


def to_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not is_number(s):
        return None
    try:
        return float(s)
    except Exception:
        return None


def norm_text(s: str) -> str:
    """轻度归一化：全角括号/斜杠/百分号等"""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("／", "/").replace("％", "%")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -----------------------------
# Token & 行聚类
# -----------------------------
def build_tokens(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    texts = raw["rec_texts"]
    scores = raw["rec_scores"]
    polys = raw["rec_polys"]

    tokens: List[Dict[str, Any]] = []
    for text, score, poly in zip(texts, scores, polys):
        x1, y1, x2, y2 = poly_to_bbox(poly)

        # cx/cy 是“中心点坐标”，用于做列对齐：
        # - cx 越大，越靠右
        # - 同一列的数字，cx 会集中在一条竖直带状区域里
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        tokens.append({
            "text": str(text),
            "score": float(score),
            "poly": poly,
            "bbox": (x1, y1, x2, y2),
            "cx": cx,
            "cy": cy,
            "h": (y2 - y1),
            "w": (x2 - x1),
        })
    return tokens


def group_lines(tokens: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    把 token 按 y 坐标聚成“行”。
    这是个简化版聚类：按 token 的中心点 cy 排序，然后用阈值把相近的归为同一行。
    """
    if not tokens:
        return []

    heights = sorted([t["h"] for t in tokens if t["h"] > 0])
    median_h = heights[len(heights) // 2] if heights else 20
    y_thresh = max(8, int(median_h * 0.6))

    tokens_sorted = sorted(tokens, key=lambda t: (t["cy"], t["cx"]))
    lines: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    cur_y: Optional[float] = None

    for t in tokens_sorted:
        if cur_y is None:
            cur = [t]
            cur_y = t["cy"]
            continue

        if abs(t["cy"] - cur_y) <= y_thresh:
            cur.append(t)
            # 平滑一下当前行的 y（避免被某个 token 拉偏）
            cur_y = cur_y * 0.8 + t["cy"] * 0.2
        else:
            lines.append(sorted(cur, key=lambda x: x["cx"]))
            cur = [t]
            cur_y = t["cy"]

    if cur:
        lines.append(sorted(cur, key=lambda x: x["cx"]))

    return lines


# -----------------------------
# 病人信息 / 结论
# -----------------------------
def parse_patient(lines: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    patient: Dict[str, Any] = {}

    def grab_after(line: List[Dict[str, Any]], key: str) -> Optional[str]:
        for i, t in enumerate(line):
            if key in t["text"]:
                for j in range(i + 1, len(line)):
                    v = (line[j]["text"] or "").strip()
                    if v:
                        return v
        return None

    for line in lines[:40]:
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

        test_no = grab_after(line, "测试号")
        if test_no and re.fullmatch(r"\d{6,}", test_no):
            patient["test_no"] = test_no

        # 年龄可能是 “51岁”
        for t in line:
            m = re.fullmatch(r"(\d+)\s*岁", (t["text"] or "").strip())
            if m:
                patient["age"] = int(m.group(1))

    # 兼容你之前的字段名（你喜欢 id/test_no 这俩都留）
    if "test_no" in patient and "id" not in patient:
        patient["id"] = patient["test_no"]

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


# -----------------------------
# 关键：表头修复 + 表头定位
# -----------------------------
def split_best_combined_token(t: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    把 OCR 粘连的 'Best %(Best' 切成两个“虚拟 token”：
      - Best
      - Best_P（也就是 %(Best) 那一列）

    仅当 t 的文本同时包含 Best + % 才会切。
    切分只用 bbox 做几何切：按比例把 bbox 一分为二。
    """
    s = norm_text(t.get("text", ""))
    if ("Best" not in s) or ("%" not in s):
        return None

    x1, y1, x2, y2 = t["bbox"]
    w = max(1, x2 - x1)

    # 经验切点：让右边的 "%(Best)" 稍微宽一点点
    split_x = x1 + int(w * 0.45)

    # 左：Best
    left_bbox = (x1, y1, split_x, y2)
    left = dict(t)
    left["text"] = "Best"
    left["bbox"] = left_bbox
    left["cx"] = (left_bbox[0] + left_bbox[2]) / 2.0
    left["cy"] = (left_bbox[1] + left_bbox[3]) / 2.0
    left["w"] = left_bbox[2] - left_bbox[0]
    left["h"] = left_bbox[3] - left_bbox[1]

    # 右：Best_P
    right_bbox = (split_x, y1, x2, y2)
    right = dict(t)
    right["text"] = "%(Best"
    right["bbox"] = right_bbox
    right["cx"] = (right_bbox[0] + right_bbox[2]) / 2.0
    right["cy"] = (right_bbox[1] + right_bbox[3]) / 2.0
    right["w"] = right_bbox[2] - right_bbox[0]
    right["h"] = right_bbox[3] - right_bbox[1]

    return left, right


def find_table_header_line(lines: List[List[Dict[str, Any]]]) -> Tuple[int, Dict[str, float]]:
    """
    常规通气表头（期望列）：
      Pred | Best | %(Best) | Act1 | Act2 | Act3 | Act4 | Act5 | Act6

    返回：
      - header 行 index
      - colx: 每列的中心 x 坐标（cx）
        之后每个数值 token 用 nearest_col(cx) 贴到最近的列中心即可。
    """
    for idx, line in enumerate(lines):
        # 先把本行 token 做一次“表头粘连修复”
        repaired: List[Dict[str, Any]] = []
        for t in line:
            sp = split_best_combined_token(t)
            if sp is None:
                repaired.append(t)
            else:
                repaired.extend(list(sp))

        # 归一化文本列表
        texts = [norm_text(t["text"]) for t in repaired]

        has_pred = any(s == "Pred" for s in texts)
        has_act1 = any(s == "Act1" for s in texts)

        # Best 可能来自 split 后的 "Best"，也可能本来就是独立 token
        has_best = any(s == "Best" for s in texts)
        if not (has_pred and has_best and has_act1):
            continue

        colx: Dict[str, float] = {}

        for t in repaired:
            s = norm_text(t["text"])
            if s == "Pred":
                colx["Pred"] = t["cx"]
            elif s == "Best":
                colx["Best"] = t["cx"]
            elif s in ("%(Best", "%(Best)", "%(Best"):
                colx["Best_P"] = t["cx"]
            elif s == "Act1":
                colx["Act1"] = t["cx"]
            elif s == "Act2":
                colx["Act2"] = t["cx"]
            elif s == "Act3":
                colx["Act3"] = t["cx"]
            elif s == "Act4":
                colx["Act4"] = t["cx"]
            elif s == "Act5":
                colx["Act5"] = t["cx"]
            elif s == "Act6":
                colx["Act6"] = t["cx"]

        # 如果 Best_P 仍然没拿到（极端情况），用 Best 与 Act1 的中点兜底
        if "Best_P" not in colx and ("Best" in colx) and ("Act1" in colx):
            colx["Best_P"] = (colx["Best"] + colx["Act1"]) / 2.0

        return idx, colx

    raise RuntimeError("Ventilation table header line not found.")


def nearest_col(cx: float, colx: Dict[str, float]) -> Optional[str]:
    best_k = None
    best_d = 10**18
    for k, x in colx.items():
        d = abs(cx - x)
        if d < best_d:
            best_d = d
            best_k = k
    return best_k


# -----------------------------
# 表解析
# -----------------------------
def parse_table(lines: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    header_idx, colx = find_table_header_line(lines)
    table: List[Dict[str, Any]] = []

    for line in lines[header_idx + 1:]:
        texts = [t["text"] for t in line]
        joined = " ".join(texts)

        # 结束条件
        if ("意见" in joined) or ("报告人" in joined) or ("Date" in joined and "Time" in joined):
            break

        if len(line) < 2:
            continue

        # 找 unit 起点：一般是 [L] / [L/min] / [s] / [%] 这类
        unit_start = None
        for i, t in enumerate(line):
            s = (t["text"] or "").strip()
            if s.startswith("[") or s in ("%]", "[L]", "[s]") or "L/min" in s or "mmol" in s:
                unit_start = i
                break
        if unit_start is None:
            continue

        # item：unit_start 之前的 token 组成
        item = " ".join([(t["text"] or "").strip() for t in line[:unit_start]]).strip()
        item = re.sub(r"\s+", " ", item).strip()
        if not item:
            continue

        # 过滤明显不是表项的行（尽量保守，不乱过滤）
        # 常规通气 item 基本都有英文字母，如 FVC / FEV 1 / MEF 75 等
        if not re.search(r"[A-Za-z]", item):
            # 但允许像 "VC IN" 这种有字母的；这里已经 cover 了
            continue

        # 合并 unit：从 unit_start 开始一直拼到包含 ']' 的 token 为止
        unit_tokens: List[str] = []
        j = unit_start
        while j < len(line):
            s = (line[j]["text"] or "").strip()
            if s:
                unit_tokens.append(s)
            if "]" in s:  # 见到 ] 就认为 unit 结束
                j += 1
                break
            j += 1

        unit_raw = "".join(unit_tokens).replace("%]", "%").strip()
        unit = unit_raw.strip().strip("[]")  # 输出时不要括号，保持你之前风格：L / L/min / s / %
        # 如果 unit 解析失败（极端），也继续入表，但 unit 给空
        if unit is None:
            unit = ""

        # 取数值 token：unit 后面的部分
        value_tokens = line[j:] if j <= len(line) else []

        row: Dict[str, Any] = {"item": item, "unit": unit}

        for t in value_tokens:
            v = to_float(t["text"])
            if v is None:
                continue
            k = nearest_col(t["cx"], colx)
            if k:
                row[k] = v

        # 关键要求：只要有 item，就入表（不再用 keys_present 过滤）
        table.append(row)

    return table


# -----------------------------
# 单文件解析 + 批处理入口
# -----------------------------
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
                    f"[OK]   {idx}/{len(raw_files)} {raw_path.name}  rows={len(parsed.get('table', []))}  {time.time()-t0:.2f}s\n"
                )
            except Exception as e:
                fail_cnt += 1
                lf.write(f"[FAIL] {idx}/{len(raw_files)} {raw_path.name}  {time.time()-t0:.2f}s  err={repr(e)}\n")

        lf.write(f"\nDONE ok={ok_cnt} fail={fail_cnt} skip={skip_cnt}\n")

    print(f"Step02 DONE. ok={ok_cnt} fail={fail_cnt} skip={skip_cnt}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
