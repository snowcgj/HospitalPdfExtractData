import re
import csv
from pathlib import Path

IN_DIR = Path("/home/test/ALabFile/CommonProject/HospitalPdf/data/txt/batch_001/心脏超声诊断报告单")
OUT_CSV = Path("/home/test/ALabFile/CommonProject/HospitalPdf/data/csv/batch_001/心脏超声诊断报告单.csv")

# 你模板里 B超字段的固定顺序（注意 4CV 这种 key）
B_KEYS = ["RV", "4CV", "LV", "LA", "AO", "AAO", "PA", "RA", "IVS", "LVPW", "IVC"]

def extract_simple_line(text: str, key: str) -> str:
    """
    从 'key: value' 这种行里抽 value
    例：姓名: 彭连清
    """
    m = re.search(rf"^{re.escape(key)}\s*:\s*(.*)$", text, flags=re.M)
    return m.group(1).strip() if m else "NA"

def extract_kv_inline(line: str) -> dict:
    """
    解析类似：RV=15;4CV=30;...;RA=43x41;IVS=8
    返回 dict
    """
    out = {}
    for part in line.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out

def normalize_number(s: str) -> str:
    """
    把 '68%' '87次/分' ' 5.1 ' 这种尽量清洗成纯数字字符串
    清洗失败返回原串或NA
    """
    if s is None:
        return "NA"
    s = s.strip()
    if s == "" or s.upper() == "NA":
        return "NA"
    # 找第一个数字（含小数）
    m = re.search(r"-?\d+(\.\d+)?", s)
    return m.group(0) if m else s

def parse_b_ultrasound(text: str) -> dict:
    """
    从 'B超_单位mm: RV=...;...'
    """
    m = re.search(r"^B超_单位mm\s*:\s*(.*)$", text, flags=re.M)
    if not m:
        return {k: "NA" for k in B_KEYS} | {"RA1": "NA", "RA2": "NA"}
    kv = extract_kv_inline(m.group(1))

    out = {k: kv.get(k, "NA") for k in B_KEYS}
    # RA 拆成 RA1/RA2
    ra = out.get("RA", "NA")
    ra1 = ra2 = "NA"
    if ra != "NA":
        mm = re.match(r"\s*(\d+)\s*[xX×*]\s*(\d+)\s*", ra)
        if mm:
            ra1, ra2 = mm.group(1), mm.group(2)
    out["RA1"], out["RA2"] = ra1, ra2
    return out

def parse_d_jet(text: str) -> dict:
    m = re.search(r"^D超_射流速_单位m/s\s*:\s*(.*)$", text, flags=re.M)
    if not m:
        return {"MV_jet": "NA", "AV_jet": "NA", "PV_jet": "NA", "TV_jet": "NA"}
    kv = extract_kv_inline(m.group(1))
    return {
        "MV_jet": kv.get("MV", "NA"),
        "AV_jet": kv.get("AV", "NA"),
        "PV_jet": kv.get("PV", "NA"),
        "TV_jet": kv.get("TV", "NA"),
    }

def parse_d_reg(text: str) -> dict:
    """
    你这行有两种：
    1) MV= 2.8;PG= ;AV= 2.3;PV= 1.2;TV= 1.8
    2) MV=2.8;PG=31;AV=无;PV=2.0;TV=3.1
    """
    m = re.search(r"^D超_反流速_单位m/s\s*:\s*(.*)$", text, flags=re.M)
    if not m:
        return {"MV_reg": "NA", "PG": "NA", "AV_reg": "NA", "PV_reg": "NA", "TV_reg": "NA"}
    kv = extract_kv_inline(m.group(1))
    return {
        "MV_reg": kv.get("MV", "NA"),
        "PG": kv.get("PG", "NA"),
        "AV_reg": kv.get("AV", "NA"),
        "PV_reg": kv.get("PV", "NA"),
        "TV_reg": kv.get("TV", "NA"),
    }

def parse_function(text: str) -> dict:
    m = re.search(r"^心功能\s*:\s*(.*)$", text, flags=re.M)
    if not m:
        return {"SV": "NA", "CO": "NA", "EF": "NA", "FS": "NA", "HR": "NA"}
    kv = extract_kv_inline(m.group(1))
    # 统一清洗成数字
    return {
        "SV": normalize_number(kv.get("SV", "NA")),
        "CO": normalize_number(kv.get("CO", "NA")),
        "EF": normalize_number(kv.get("EF", "NA")),
        "FS": normalize_number(kv.get("FS", "NA")),
        "HR": normalize_number(kv.get("HR", "NA")),
    }

def parse_multiline_block(text: str, key: str) -> str:
    """
    抽取类似：
    检查提示:
    xxx
    yyy
    报告日期时间: ...
    这种块，直到遇到下一个 'xxx:' 行。
    """
    # 从 key: 开始捕获到下一个 “行首有xxx:” 的位置
    pattern = rf"^{re.escape(key)}\s*:\s*\n(.*?)(?=^\S+\s*:|\Z)"
    m = re.search(pattern, text, flags=re.S | re.M)
    if not m:
        return "NA"
    # 压成一行，方便 csv
    content = m.group(1).strip()
    content = re.sub(r"\s*\n\s*", " | ", content)
    return content if content else "NA"

def parse_report(txt_path: Path) -> dict:
    t = txt_path.read_text(encoding="utf-8", errors="ignore")

    row = {
        "source_txt": txt_path.name,
        "报告类型": extract_simple_line(t, "报告类型"),
        "医院": extract_simple_line(t, "医院"),
        "姓名": extract_simple_line(t, "姓名"),
        "性别": extract_simple_line(t, "性别"),
        "出生日期": extract_simple_line(t, "出生日期"),
        "年龄": normalize_number(extract_simple_line(t, "年龄")),
        "床号": normalize_number(extract_simple_line(t, "床号")),
        "申请科室": extract_simple_line(t, "申请科室"),
        "患者ID": extract_simple_line(t, "患者ID"),
        "检查号": extract_simple_line(t, "检查号"),
        "申请医生": extract_simple_line(t, "申请医生"),
        "临床诊断": extract_simple_line(t, "临床诊断"),
        "报告日期时间": extract_simple_line(t, "报告日期时间"),
        "录入者": extract_simple_line(t, "录入者"),
        "报告医师": extract_simple_line(t, "报告医师"),
        "备注": extract_simple_line(t, "备注"),
        "检查提示": parse_multiline_block(t, "检查提示"),
    }

    row |= parse_b_ultrasound(t)
    row |= parse_d_jet(t)
    row |= parse_d_reg(t)
    row |= parse_function(t)

    # 检查所见：你现在是“检查所见:”下面有二维/彩色两段
    # 为了稳，我们分别抓“二维：...”和“彩色：...”
    m2d = re.search(r"二维[：:]\s*(.*)", t)
    mc = re.search(r"彩色[：:]\s*(.*)", t)
    row["检查所见_二维"] = m2d.group(1).strip() if m2d else "NA"
    row["检查所见_彩色"] = mc.group(1).strip() if mc else "NA"

    # 数值字段再做一遍清洗（避免 '15 ' 这种）
    for k in ["RV","4CV","LV","LA","AO","AAO","PA","IVS","LVPW","IVC","RA1","RA2",
              "AV_jet","PV_jet","MV_reg","AV_reg","PV_reg","TV_reg","PG"]:
        if k in row:
            row[k] = normalize_number(row[k])

    return row

def main():
    assert IN_DIR.exists(), f"输入目录不存在：{IN_DIR}"
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    txts = sorted(IN_DIR.glob("*.txt"))
    assert txts, f"没有找到txt：{IN_DIR}/*.txt"

    rows = [parse_report(p) for p in txts]

    # 固定列顺序（你后面想加字段就往这里加）
    fieldnames = [
        "source_txt","报告类型","医院","姓名","性别","出生日期","年龄","床号","申请科室","患者ID","检查号","申请医生","临床诊断",
        "报告日期时间","录入者","报告医师","备注",
        "RV","4CV","LV","LA","AO","AAO","PA","RA1","RA2","IVS","LVPW","IVC",
        "MV_jet","AV_jet","PV_jet","TV_jet",
        "MV_reg","PG","AV_reg","PV_reg","TV_reg",
        "SV","CO","EF","FS","HR",
        "检查所见_二维","检查所见_彩色","检查提示",
    ]

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "NA") for k in fieldnames})

    print(f"OK: 写入 {OUT_CSV}  共 {len(rows)} 行")

if __name__ == "__main__":
    main()
