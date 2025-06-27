import time
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_path = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

# 1) Load processor
processor = AutoProcessor.from_pretrained(model_path)

# 2) Load model in FP32 then convert ALL weights to FP16
model = (
    AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        _attn_implementation="eager",
    )
    .half()       # convert weights to FP16
    .to("cuda")
    .eval()
)

# 3) Prepare your warm-up and real messages
warmup_message = [
    {"role":"user","content":[{"type":"text","text":"Just say any random word and stop."}]}
]
messages = [
    {
        "role":"user",
        "content":[
            {"type":"image","path":"pfp.jpg"},
            {"type":"text","text":"Describe this image in detail."}
        ]
    }
]

# 4) Helper to tokenize + move to GPU
def make_inputs(msgs):
    batch = processor.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to("cuda")

    # ────── ensure image tensors are in FP16 ──────
    if "pixel_values" in batch:
        batch["pixel_values"] = batch["pixel_values"].half()

    return batch

warmup_input = make_inputs(warmup_message)
inputs       = make_inputs(messages)

# 5) Warm-up
print("Warming up FP16 on GPU…")
_ = model.generate(**warmup_input, max_new_tokens=64)

# 6) Benchmark
print("Benchmarking FP16 GPU…")
t0 = time.time()
out_fp16 = model.generate(**inputs, max_new_tokens=64)
lat_fp16 = time.time() - t0

# 7) Decode & print
resp_fp16 = processor.batch_decode(out_fp16, skip_special_tokens=True)[0]
print(f"FP16 output (took {lat_fp16:.3f}s):\n{resp_fp16}")
