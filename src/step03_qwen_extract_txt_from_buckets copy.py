import re
import gc
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
'''
大模型分析每一张图片， 将所读内容放到txt文件中；
运行命令：CUDA_VISIBLE_DEVICES=4,5,6,7 python step03_extract_echo_only.py 

'''


# ====== 只处理这个文件夹 ======
IN_DIR = Path("/home/test/ALabFile/CommonProject/HospitalPdf/data/buckets/batch_001/心脏超声诊断报告单")
OUT_DIR = Path("/home/test/ALabFile/CommonProject/HospitalPdf/data/txt/batch_001/心脏超声诊断报告单")

MODEL_ID = "/home/test/ALabFile/CommonProject/HospitalPdf/model_weights/Qwen3-VL-8B-Instruct"

MAX_NEW_TOKENS = 2048
SKIP_EXISTING = True
EVERY_N_EMPTY_CACHE = 20  # 每处理N张清一次缓存，设为0关闭

# ====== 心超固定模板 prompt ======
PROMPT = (
    "你是医学检查报告信息抽取器。只根据图片中可见文字抽取信息，严格按模板输出。\n"
    "要求：\n"
    "1) 不要解释、不要补充、不要猜测；看不到就填NA。\n"
    "2) 数值保留原单位；像 43*41 统一写成 43x41。\n"
    "3) 只输出【BEGIN】到【END】之间的内容。\n\n"
    "【BEGIN】\n"
    "报告类型: 心脏超声诊断报告单\n"
    "医院: \n"
    "姓名: \n"
    "性别: \n"
    "出生日期: \n"
    "年龄: \n"
    "床号: \n"
    "申请科室: \n"
    "患者ID: \n"
    "检查号: \n"
    "申请医生: \n"
    "临床诊断: \n"
    "\n"
    "B超_单位mm: RV= ;4CV= ;LV= ;LA= ;AO= ;AAO= ;PA= ;RA= ;IVS= ;LVPW= ;IVC=\n"
    "D超_射流速_单位m/s: MV= ;AV= ;PV= ;TV=\n"
    "D超_反流速_单位m/s: MV= ;PG= ;AV= ;PV= ;TV=\n"
    "心功能: SV= ;CO= ;EF= ;FS= ;HR=\n"
    "\n"
    "检查所见: \n"
    "检查提示: \n"
    "\n"
    "报告日期时间: \n"
    "录入者: \n"
    "报告医师: \n"
    "备注: \n"
    "【END】\n"
)

def safe_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    return name

def main():
    assert IN_DIR.exists(), f"输入目录不存在：{IN_DIR}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 模型只加载一次
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={0: "22GiB", 1: "22GiB", 2: "22GiB", 3: "22GiB"},
        low_cpu_mem_usage=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("hf_device_map =", model.hf_device_map)

    pngs = sorted(IN_DIR.glob("*.png"))
    print(f"输入图片数量：{len(pngs)}")
    if not pngs:
        return

    fail_log = OUT_DIR / "fail.log"
    total = ok = skip = fail = 0

    for img_path in pngs:
        total += 1
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在处理第{total}张图片。")
        out_txt = OUT_DIR / f"{safe_filename(img_path.stem)}.txt"

        if SKIP_EXISTING and out_txt.exists():
            skip += 1
            continue

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(img_path)},
                {"type": "text", "text": PROMPT},
            ],
        }]

        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            # 多卡切分：inputs 放到第一张卡
            first_dev = next(iter(model.hf_device_map.values()))
            if isinstance(first_dev, int):
                first_dev = f"cuda:{first_dev}"
            inputs = {k: v.to(first_dev) if torch.is_tensor(v) else v for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,       # ✅ 确定性，减少8B波动
                    temperature=0.0,
                    top_p=1.0,
                )

            # 只 decode 新生成部分
            prompt_len = inputs["input_ids"].shape[-1]
            new_tokens = generated_ids[0][prompt_len:]
            text = processor.decode(new_tokens, skip_special_tokens=True).strip()

            out_txt.write_text(text, encoding="utf-8")
            ok += 1

        except Exception as e:
            fail += 1
            msg = f"[FAIL] {img_path} err={repr(e)}\n"
            print(msg)
            with open(fail_log, "a", encoding="utf-8") as f:
                f.write(msg)

        if EVERY_N_EMPTY_CACHE and (total % EVERY_N_EMPTY_CACHE == 0):
            gc.collect()
            torch.cuda.empty_cache()
        print("处理完毕")

    print(f"\nDONE: total={total}, ok={ok}, skip={skip}, fail={fail}")
    if fail:
        print(f"失败详情：{fail_log}")

if __name__ == "__main__":
    main()
