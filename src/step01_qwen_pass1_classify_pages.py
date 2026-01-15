import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

'''
在jsonl文件夹下面生成一个 pass1.jsonl文件， 里面存储一些信息， 
step02会用到， step02负责根据这个信息把图片放到正确的桶里。
'''


print("SCRIPT_VERSION=PASS1_V3")  # <- 看到这行才说明你跑的是这份脚本


PROJECT_ROOT = Path("/home/test/ALabFile/CommonProject/HospitalPdf")
BATCH = "batch_001"

MODEL_DIR = PROJECT_ROOT / "model_weights" / "Qwen3-VL-4B-Instruct"
IMG_DIR = PROJECT_ROOT / "data" / "pages_png" / BATCH
OUT_DIR = PROJECT_ROOT / "data" / "jsonl" / BATCH
LOG_DIR = PROJECT_ROOT / "data" / "logs" / BATCH

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "pass1.jsonl"
ERR_JSONL = LOG_DIR / "pass1_errors.jsonl"

DTYPE = torch.float16


def append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def load_done_set(jsonl_path: Path) -> set:
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                p = obj.get("image_path")
                if p:
                    done.add(p)
            except Exception:
                continue
    return done


def parse_first_json_object(s: str):
    """
    只解析第一个 JSON 对象（dict），忽略后续任何文本/第二个 JSON
    这能根治 JSONDecodeError: Extra data
    """
    dec = json.JSONDecoder()
    start = s.find("{")
    if start == -1:
        return None, "no_json_start"

    # raw_decode 只吃掉第一个 JSON，然后告诉你吃到哪里结束
    try:
        obj, end = dec.raw_decode(s[start:])
        if not isinstance(obj, dict):
            return None, "first_json_not_object"
        return obj, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def build_prompt() -> str:
    schema = {
        "image_path": "",
        "kind_guess": "",
        "title": "",
        "hospital": "",
        "has_table": False,
        "table_headers": [],
        "key_terms": [],
        "brief": "",
        "bucket_hint": ""
    }

    return (
        "你在做医疗报告页面的【归类/分桶】辅助。\n"
        "请严格只输出一个 JSON（禁止输出任何解释、markdown、换行后的补充）。\n"
        "JSON 必须可被标准解析，字段必须齐全，不确定就填空。\n"
        "kind_guess 只能从：检验/超声/心电/CT/住院/缴费/其他 中选一个。\n"
        "输出模板：\n"
        "一定要记住：其中title只包含 检查项目 不能包含医院名字 "
        f"{json.dumps(schema, ensure_ascii=False)}\n"
        "再次强调：只输出一个 JSON，对 JSON 之外的任何字符都不要输出。"
    )


def main():
    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=DTYPE,
        device_map="auto",               # ✅ 关键：让 HF 自动切多卡
        # 下面这行可选，但建议加：告诉它每张卡最多用多少显存（单位可以是 "GiB"）
        max_memory={0: "22GiB", 1: "22GiB", 2: "22GiB", 3: "22GiB"},
        trust_remote_code=True,
    ).eval()

    processor = AutoProcessor.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    prompt = build_prompt()
    done = load_done_set(OUT_JSONL)

    imgs = sorted(IMG_DIR.rglob("*.png"))
    print(f"Found {len(imgs)} png(s) in {IMG_DIR}")
    print(f"Done  {len(done)} png(s) already in {OUT_JSONL}")

    for idx, img_path in enumerate(imgs, 1):
        img_abs = str(img_path.resolve())
        if img_abs in done:
            continue

        print(f"[{idx}/{len(imgs)}] ENTER {img_path.name}")
        t0 = time.time()
        out = ""

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_abs},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )

            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[0][prompt_len:]     # 只取模型新生成的部分
            out = processor.decode(new_tokens, skip_special_tokens=True)

            obj, err = parse_first_json_object(out)
            if err:
                raise ValueError(f"parse_json_failed: {err}")

            obj["image_path"] = img_abs
            obj.setdefault("kind_guess", "")
            obj.setdefault("title", "")
            obj.setdefault("hospital", "")
            obj.setdefault("has_table", False)
            obj.setdefault("table_headers", [])
            obj.setdefault("key_terms", [])
            obj.setdefault("brief", "")
            obj.setdefault("bucket_hint", "")

            append_jsonl(OUT_JSONL, obj)

            dt = time.time() - t0
            print(f"[{idx}/{len(imgs)}] ok {img_path.name}  {dt:.2f}s")

        except Exception as e:
            append_jsonl(ERR_JSONL, {
                "image_path": img_abs,
                "error": f"{type(e).__name__}: {e}",
                "out_preview": out[:800],
            })
            dt = time.time() - t0
            print(f"[{idx}/{len(imgs)}] FAIL {img_path.name}  {type(e).__name__}: {e}  {dt:.2f}s")

    print("Done.")
    print(f"Output: {OUT_JSONL}")
    print(f"Errors: {ERR_JSONL}")


if __name__ == "__main__":
    main()
