import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"


def load_clip(model_name=MODEL_NAME, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


def _normalize(x):
    if not isinstance(x, torch.Tensor):
        x = x.pooler_output
    return x / x.norm(dim=-1, keepdim=True)


@torch.no_grad()
def embed_texts(texts, model, processor, device, batch_size=256):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        inputs = processor(text=batch, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        out.append(_normalize(model.get_text_features(**inputs)).cpu())
    return torch.cat(out).numpy().astype(np.float32)


@torch.no_grad()
def embed_images(paths, model, processor, device, batch_size=64):
    out = []
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in paths[i : i + batch_size]]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        out.append(_normalize(model.get_image_features(**inputs)).cpu())
    return torch.cat(out).numpy().astype(np.float32)
