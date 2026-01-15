import re
import gc
import time
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


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


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_gpu_mem_str():
    """
    轻量显存日志：只取当前进程在 CUDA 上的 allocated/reserved。
    注意：这是 PyTorch 视角，不等于 nvidia-smi 的总显存占用，但足够用来观察趋势。
    """
    if not torch.cuda.is_available():
        return "gpu=NA"

    # 当前进程默认设备的显存（对多卡切分来说，这里主要看“inputs所在卡”的趋势）
    dev = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(dev) / (1024 ** 3)
    reserv = torch.cuda.memory_reserved(dev) / (1024 ** 3)
    return f"gpu{dev}: alloc={alloc:.2f}GiB reserved={reserv:.2f}GiB"


def log_line(fp, s: str):
    print(s, flush=True)
    fp.write(s + "\n")
    fp.flush()


def main():
    assert IN_DIR.exists(), f"输入目录不存在：{IN_DIR}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_log = OUT_DIR / "run.log"
    with open(run_log, "a", encoding="utf-8") as lf:
        log_line(lf, f"\n========== RUN START {now_str()} ==========")
        log_line(lf, f"IN_DIR={IN_DIR}")
        log_line(lf, f"OUT_DIR={OUT_DIR}")
        log_line(lf, f"MODEL_ID={MODEL_ID}")
        log_line(lf, f"MAX_NEW_TOKENS={MAX_NEW_TOKENS} SKIP_EXISTING={SKIP_EXISTING} EMPTY_CACHE_EVERY={EVERY_N_EMPTY_CACHE}")

        # 模型只加载一次
        t_load0 = time.perf_counter()
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "22GiB", 1: "22GiB", 2: "22GiB", 3: "22GiB"},
            low_cpu_mem_usage=True,
        ).eval()
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        t_load1 = time.perf_counter()

        log_line(lf, f"[{now_str()}] model loaded in {(t_load1 - t_load0):.2f}s")
        log_line(lf, f"hf_device_map={model.hf_device_map}")

        pngs = sorted(IN_DIR.glob("*.png"))
        log_line(lf, f"输入图片数量：{len(pngs)}")
        if not pngs:
            log_line(lf, f"========== RUN END {now_str()} ==========")
            return

        fail_log = OUT_DIR / "fail.log"

        total = ok = skip = fail = 0
        total_gen_tokens = 0
        total_ok_time = 0.0

        t_all0 = time.perf_counter()

        for img_path in pngs:
            total += 1
            out_txt = OUT_DIR / f"{safe_filename(img_path.stem)}.txt"

            # 进度头
            log_line(lf, f"[{now_str()}] ({total}/{len(pngs)}) start  file={img_path.name}  {get_gpu_mem_str()}")

            if SKIP_EXISTING and out_txt.exists():
                skip += 1
                log_line(lf, f"[{now_str()}] ({total}/{len(pngs)}) SKIP exists -> {out_txt.name}")
                continue

            t0 = time.perf_counter()

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
                gen_tokens = int(new_tokens.shape[-1])
                total_gen_tokens += gen_tokens

                text = processor.decode(new_tokens, skip_special_tokens=True).strip()
                out_txt.write_text(text, encoding="utf-8")

                ok += 1
                dt = time.perf_counter() - t0
                total_ok_time += dt

                log_line(
                    lf,
                    f"[{now_str()}] ({total}/{len(pngs)}) OK  out={out_txt.name}  gen_tokens={gen_tokens}  time={dt:.2f}s  {get_gpu_mem_str()}",
                )

            except Exception as e:
                fail += 1
                dt = time.perf_counter() - t0
                msg = f"[FAIL] {img_path} err={repr(e)} time={dt:.2f}s\n"
                print(msg, flush=True)
                with open(fail_log, "a", encoding="utf-8") as f:
                    f.write(msg)
                log_line(lf, f"[{now_str()}] ({total}/{len(pngs)}) FAIL time={dt:.2f}s err={repr(e)}")

            if EVERY_N_EMPTY_CACHE and (total % EVERY_N_EMPTY_CACHE == 0):
                gc.collect()
                torch.cuda.empty_cache()
                log_line(lf, f"[{now_str()}] cache cleared  {get_gpu_mem_str()}")

        t_all1 = time.perf_counter()

        wall = t_all1 - t_all0
        avg_ok = (total_ok_time / ok) if ok else 0.0
        imgs_per_sec = (ok / total_ok_time) if total_ok_time > 0 else 0.0
        avg_gen = (total_gen_tokens / ok) if ok else 0.0

        log_line(lf, "\n===== SUMMARY =====")
        log_line(lf, f"total={total} ok={ok} skip={skip} fail={fail}")
        log_line(lf, f"wall_time={wall:.2f}s  ok_compute_time={total_ok_time:.2f}s")
        log_line(lf, f"avg_time_per_ok={avg_ok:.2f}s  throughput={imgs_per_sec:.3f} img/s")
        log_line(lf, f"avg_generated_tokens={avg_gen:.1f}  total_generated_tokens={total_gen_tokens}")
        if fail:
            log_line(lf, f"失败详情：{fail_log}")
        log_line(lf, f"========== RUN END {now_str()} ==========")


if __name__ == "__main__":
    main()
