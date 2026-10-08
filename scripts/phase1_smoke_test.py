import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fakenews.data import ID_COL, LABEL_COL, ROOT, TEXT_COL, URL_COL, load_raw
from fakenews.embed import embed_images, embed_texts, load_clip
from fakenews.images import download_images, image_path

N = 100
SEED = 42
IMG_DIR = ROOT / "data" / "smoke" / "images"

df = load_raw("train").dropna(subset=[URL_COL])
df = df.groupby(LABEL_COL, group_keys=False).sample(frac=N / len(df), random_state=SEED).reset_index(drop=True)
print(f"sample: {len(df)} rows, labels {df[LABEL_COL].value_counts().to_dict()}")

t = time.time()
status = download_images(df, IMG_DIR)
t_dl = time.time() - t
n_ok = (status["status"] == "ok").sum()
print(f"download: {n_ok}/{len(df)} ok in {t_dl:.1f}s")
print("failures:", status.loc[status["status"] != "ok", "reason"].value_counts().to_dict())

df = df[df[ID_COL].isin(status.loc[status["status"] == "ok", ID_COL])].reset_index(drop=True)

t = time.time()
model, processor, device = load_clip()
t_load = time.time() - t
t = time.time()
X_text = embed_texts(df[TEXT_COL].tolist(), model, processor, device)
t_text = time.time() - t
t = time.time()
X_img = embed_images([image_path(IMG_DIR, i) for i in df[ID_COL]], model, processor, device)
t_img = time.time() - t
print(f"clip ({device}): load {t_load:.1f}s, text {t_text:.1f}s, image {t_img:.1f}s for {len(df)} rows")
assert X_text.shape[0] == X_img.shape[0] == len(df)
assert np.isfinite(X_text).all() and np.isfinite(X_img).all()

y = df[LABEL_COL].to_numpy()
modes = {"text": X_text, "image": X_img, "text+image": np.hstack([X_text, X_img])}
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.3, stratify=y, random_state=SEED)
for name, X in modes.items():
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X[idx_tr], y[idx_tr])
    f1 = f1_score(y[idx_te], clf.predict(X[idx_te]), average="macro")
    print(f"{name:>10}: dim {X.shape[1]}, macro-F1 {f1:.3f} (100 แถว ยังไม่มีความหมาย)")

per_img = t_img / len(df)
print(f"\nประมาณเวลา image embeddings บน {device}: {per_img*1000:.0f} ms/รูป -> 10k ≈ {per_img*1e4/60:.0f} นาที, 20k ≈ {per_img*2e4/60:.0f} นาที")
print(f"ประมาณเวลาโหลดรูป (8 threads): 10k ≈ {t_dl/len(status)*1e4/60:.0f} นาที")
