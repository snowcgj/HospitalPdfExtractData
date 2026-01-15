import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "/home/test/ALabFile/CommonProject/HospitalPdf/model_weights/Qwen3-VL-8B-Instruct"
# model_id = "/home/test/ALabFile/CommonProject/HospitalPdf/model_weights/qwen3_32b_vl"
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",               # ✅ 关键：让 HF 自动切多卡
    # 下面这行可选，但建议加：告诉它每张卡最多用多少显存（单位可以是 "GiB"）
    max_memory={0: "22GiB", 1: "22GiB", 2: "22GiB", 3: "22GiB"},
).eval()

processor = AutoProcessor.from_pretrained(model_id)

# img_abs_path = "/home/test/ALabFile/CommonProject/HospitalPdf/data/buckets/batch_001/一口气弥散报告/肺功能-0012598804-喻细兰__p0003.png"
img_abs_path = "data/buckets/batch_001/常规通气/肺功能-0003120865-李榕__p0001.png"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_abs_path},
            {"type": "text", "text": "这是一份检查报告，请尽可能完整读取其中所有可见信息，并按条目输出。忽略图像但不要忽略表格"}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

# ⚠️ 注意：当 device_map="auto" 时，inputs 不要简单 .to(model.device)
# 因为模型被切到多张卡了；通常把 inputs 放到第一张卡即可：
first_device = next(iter(model.hf_device_map.values()))
# hf_device_map 的 value 可能是 'cuda:0' 或者 int/str，这里简单处理：
if isinstance(first_device, int):
    first_device = f"cuda:{first_device}"
inputs = {k: v.to(first_device) if torch.is_tensor(v) else v for k, v in inputs.items()}

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=2048)

print("模型使用了：" + str(model.hf_device_map) + "张卡")
prompt_len = inputs["input_ids"].shape[-1]
new_tokens = generated_ids[0][prompt_len:]
print(processor.decode(new_tokens, skip_special_tokens=True))
