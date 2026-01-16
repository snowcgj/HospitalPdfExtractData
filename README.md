
# Hospital PDF Report Pipeline (PNG → OCR → JSON → CSV)

本项目用于批量处理医院检查报告（肺功能：一口气弥散 / 常规通气 / 支气管舒张等；以及后续可扩展的心超等）。

核心目标：把已经拆分好的 **单页 PNG** 报告，稳定、可控地转成 **结构化 JSON**，再转成 **一行一报告的扁平 CSV**，方便后续 Pandas / sklearn / torch 使用。
---

## Features

- ✅ Step01 通用 OCR：所有报告类型共用同一份 OCR 脚本（复制即可）
- ✅ Step02 规则解析：按报告类型定制（表头定位、列对齐、行解析、意见提取等）
- ✅ Step03 扁平化 CSV：列名统一使用 `item__field` 形式，工程化、下游友好
- ✅ 目录物理隔离：每种报告类型一个独立子目录，互不影响
- ✅ 支持 GPU 环境（RTX 4090 / PaddlePaddle-GPU）


---

## Environment

推荐环境：

- OS: Linux
- Python: 3.10 (conda)
- GPU: RTX 4090
- PaddlePaddle-GPU: 3.2.2 (CUDA 11.8 runtime, Driver 12.4)
- OCR: PaddleOCR 3.x / PP-OCRv5

---

## Input / Output

### Input
- 每份 PDF 已拆成 **单页 PNG**
- 不同报告类型放在不同文件夹（物理隔离）

### Output
- Stage01: OCR 原始产物 `*_raw.json`
- Stage02: 结构化结果 `*.parsed.json`
- Stage03: 扁平 CSV（一行一报告）

---

## Project Structure

以某一种报告类型目录为例：

```

YiKouQiMiSan/
step01_paddle_ocr.py          # 通用 OCR（直接复制，不改）
step02_parse_xxx.py           # 该报告类型的规则解析（定制）
step03_to_csv.py              # 扁平化 CSV（定制）
out_stage01_raw/
raw_json/*.json             # 只保留 rec_texts/rec_scores/rec_polys
vis_png/*.png               # OCR 可视化（用于人工校验）
out_stage02_parsed/
parsed_json/*.parsed.json
logs/*.log
out_stage03_csv/
*.csv

````

---

## Pipeline Overview (Step00 ~ Step04)

> 不同目录里脚本名可能略有不同，但 pipeline 的阶段含义一致。

### Step01 - OCR (通用)
输入：PNG 文件夹  
输出：
- `out_stage01_raw/raw_json/*_raw.json`
- `out_stage01_raw/vis_png/*.png`（用于人工检查）

OCR 输出统一裁剪为：

```json
{
  "rec_texts": [...],
  "rec_scores": [...],
  "rec_polys": [...]
}
````

### Step02 - Parse (按报告类型定制)

输入：Stage01 的 `_raw.json`
输出：结构化 `*.parsed.json`

结构示例：

```json
{
  "report_type": "xxx",
  "patient": { ... },
  "table": [
    { "item": "FVC", "unit": "L", "Pred": 3.00, "Act1": 2.89, ... }
  ],
  "impression": "..."
}
```

实现骨架通常包含：

* `build_tokens()`：把 OCR 文本 + poly 转为 token（带 bbox / cx / cy）
* `group_lines()`：按 y 坐标聚类成行
* `parse_patient()`：姓名/性别/年龄/身高体重/测试号等
* `find_table_header_line()`：定位表头（最关键，按类型定制）
* `parse_table()`：按列对齐规则把数值归到字段
* `parse_impression()`：解析意见/结论

> 一些报告会出现表头 token 粘连（如 `Best %(Best`），此时必须依赖 poly/bbox 做列定位与拆分策略。

### Step03 - JSON → CSV (扁平化)

输入：Stage02 的 `*.parsed.json`
输出：一行一报告的 CSV

列命名规则：

* `列名 = item__field`
* 示例：`FVC__Pred`、`FEV1__Act1`、`MMEF_75_25__Best_P`

优势：

* 没有 multi-index
* 不展开成多行
* 对机器学习/统计处理非常友好

---

## Quick Start

进入某个报告类型目录后按顺序执行：

```bash
# Step01: OCR
python step01_paddle_ocr.py

# Step02: 结构化解析
python step02_parse_xxx.py

# Step03: 扁平 CSV
python step03_to_csv.py
```

---

## Notes / Design Principles

* **不使用大模型做表格解析**：速度慢、结果不稳定，且工程可控性差
* **强依赖 bbox/poly**：仅靠纯文本顺序会错位；必须用几何对齐来还原表格结构
* **每种报告类型独立目录**：避免“一个脚本改坏所有类型”
* **新增报告类型流程**：

  1. 复制通用 `step01_paddle_ocr.py`
  2. 新写/改写 `step02_parse_xxx.py` 的表头定位与列规则
  3. 写 `step03_to_csv.py`（设定 value_fields 和输出文件名）

---

## Extra

* `test_qwen3vl_read_image.py`
  用于单独测试 VL 模型读取图片（不参与主 pipeline）。

---

## Roadmap (Optional)

* [ ] 增加更多肺功能报告模板的 parser
* [ ] 心超报告：OCR → 规则抽取 → 结构化/CSV
* [ ] 更完善的异常日志与失败样本归档（便于快速迭代规则）

---

## License

Internal / Research use.

