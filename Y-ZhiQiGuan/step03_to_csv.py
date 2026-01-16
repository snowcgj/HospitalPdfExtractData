# YiKouQiMiSa/step03_to_csv.py
import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Set


def safe_key(s: str) -> str:
    """
    把 item 变成适合当列名的字符串：
    - 去首尾空白
    - 空格 -> _
    - / -> _
    - 其他奇怪字符尽量变 _
    """
    s = (s or "").strip()
    s = s.replace(" ", "_").replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^0-9A-Za-z_\-\%]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def collect_all_metric_columns(files: List[Path]) -> List[str]:
    """
    扫一遍所有 parsed.json，收集所有出现过的 item__field 列名，保证 CSV 列固定。
    """
    metric_cols: Set[str] = set()
    value_fields = ["unit", "Pred", "Act1", "Act1_P", "Act2", "Act2_P", "Delta_P"]

    for p in files:
        data = json.loads(p.read_text(encoding="utf-8"))
        for row in data.get("table", []):
            item = safe_key(row.get("item", ""))
            if not item:
                continue
            for f in value_fields:
                metric_cols.add(f"{item}__{f}")

    return sorted(metric_cols)


def main():
    IN_PARSED_DIR = Path("./out_stage02_parsed/parsed_json")
    OUT_DIR = Path("./out_stage03_csv")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_csv = OUT_DIR / "diffusion_reports_wide.csv"

    files = sorted(IN_PARSED_DIR.glob("*.parsed.json"))
    if not files:
        raise FileNotFoundError(f"No parsed json found in {IN_PARSED_DIR}")

    # 固定的基础字段（一张报告一行）
    base_cols = ["file_stem", "name", "sex", "age", "height_cm", "weight_kg", "id", "test_no",
             "report_date", "report_time", "operator", "reviewer",
             "impression"]

    # 动态指标字段（item__Pred 等）
    metric_cols = collect_all_metric_columns(files)

    fieldnames = base_cols + metric_cols

    value_fields = ["unit", "Pred", "Act1", "Act1_P", "Act2", "Act2_P", "Delta_P"]

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for p in files:
            data = json.loads(p.read_text(encoding="utf-8"))
            patient = data.get("patient", {})
            impression = data.get("impression", "")

            out: Dict[str, Any] = {k: "" for k in fieldnames}
            out["file_stem"] = p.stem.replace(".parsed", "")
            out["name"] = patient.get("name", "")
            out["sex"] = patient.get("sex", "")
            out["age"] = patient.get("age", "")
            out["height_cm"] = patient.get("height_cm", "")
            out["weight_kg"] = patient.get("weight_kg", "")
            out["id"] = patient.get("id", "")
            out["test_no"] = patient.get("test_no", "")
            out["report_date"] = patient.get("report_date", "")
            out["report_time"] = patient.get("report_time", "")
            out["operator"] = patient.get("operator", "")
            out["reviewer"] = patient.get("reviewer", "")
            out["impression"] = impression

            # 把 table 里的每个 item 展开成列
            for row in data.get("table", []):
                item = safe_key(row.get("item", ""))
                if not item:
                    continue
                for f in value_fields:
                    col = f"{item}__{f}"
                    if col in out:
                        out[col] = row.get(f, "")

            w.writerow(out)

    print("OK:", out_csv)
    print("reports:", len(files))
    print("columns:", len(fieldnames))


if __name__ == "__main__":
    main()
