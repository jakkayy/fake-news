import numpy as np

from .embed_store import load_embeddings

MODES = ["text", "image", "text+image"]


def build_features(mode, text, image):
    if mode == "text":
        return text
    if mode == "image":
        return image
    if mode == "text+image":
        return np.hstack([text, image])
    raise ValueError(f"unknown mode: {mode}")


def load_split_features(emb_dir, split_file, split):
    text, image, y, ids = load_embeddings(emb_dir, split_file, split)
    return {m: build_features(m, text, image) for m in MODES}, y, ids
