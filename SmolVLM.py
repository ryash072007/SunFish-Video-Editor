from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_path = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    _attn_implementation="eager"  # No FlashAttention on CPU
).to("cpu")

# For image input
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "text", "text": "Can you describe this image?"},
        ]
    }
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to("cpu")

outputs = model.generate(**inputs, max_new_tokens=64)
print(processor.batch_decode(outputs, skip_special_tokens=True)[0])
