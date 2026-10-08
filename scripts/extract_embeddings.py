import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fakenews.data import ROOT
from fakenews.embed import MODEL_NAME, load_clip
from fakenews.embed_store import extract_to_dir

IMG_DIR = ROOT / "data" / "images"
EMB_DIR = ROOT / "data" / "embeddings"

df = pd.read_csv(ROOT / "data" / "processed" / "sample.csv")
model, processor, device = load_clip()
t = time.time()
extract_to_dir(df, IMG_DIR, EMB_DIR, model, processor, device)
print(f"{MODEL_NAME} on {device}: {len(df)} rows in {time.time() - t:.0f}s")

text, image = np.load(EMB_DIR / "text.npy"), np.load(EMB_DIR / "image.npy")
ids = pd.read_csv(EMB_DIR / "ids.csv")["id"]
assert len(text) == len(image) == len(ids) == len(df)
assert (ids.to_numpy() == df["id"].to_numpy()).all()
assert np.isfinite(text).all() and np.isfinite(image).all()
assert np.allclose(np.linalg.norm(text, axis=1), 1, atol=1e-3) and np.allclose(np.linalg.norm(image, axis=1), 1, atol=1e-3)
print("text", text.shape, "image", image.shape, "checks passed")
