import time
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_path = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

# 1. Load processor and model in full precision (FP32)
processor = AutoProcessor.from_pretrained(model_path)
model_fp32 = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    _attn_implementation="eager",
).to("cpu").eval()

# 2. Apply dynamic INT8 quantization for speed
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
).to("cpu").eval()

warmup_message = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Just say any random word and stop."}
        ]
    }
]

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "pfp.jpg"},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

warmup_input = processor.apply_chat_template(
    warmup_message,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to("cpu")

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to("cpu")

print("Warming up")
_ = model_int8.generate(**warmup_input, max_new_tokens=64)
print("Warming up - int8 done")

print("Benchmarking INT8")
# 4. Benchmark INT8
t1 = time.time()
out_int8 = model_int8.generate(**inputs, max_new_tokens=64)
lat_int8 = time.time() - t1

print("Decoding both responses")
# 5. Decode responses
resp_int8 = processor.batch_decode(out_int8, skip_special_tokens=True)[0]

print(f"INT8 output (took {lat_int8:.3f} s):\n{resp_int8}")
