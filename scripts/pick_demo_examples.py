import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fakenews.data import ID_COL, ROOT, load_raw

OUT_DIR = ROOT / "app" / "examples"
SKIP = {"at5mt6"}

df = pd.read_csv(ROOT / "results" / "test_predictions.csv")
df = df.merge(pd.read_csv(ROOT / "data" / "processed" / "sample.csv")[[ID_COL, "clean_title"]], on=ID_COL)
df = df.merge(load_raw("train", cols=[ID_COL, "subreddit"]), on=ID_COL)
df = df[~df[ID_COL].isin(SKIP) & df["clean_title"].str.split().str.len().between(3, 14)]
for m in ["text", "image", "text+image"]:
    df[f"ok_{m}"] = (df[f"prob_real_{m}"] >= 0.5).astype(int) == df["label"]
df["conf"] = (df["prob_real_text+image"] - 0.5).abs()

groups = [
    ("ทายถูก: ข่าวปลอมชัดเจน", df["ok_text+image"] & (df["label"] == 0) & df["ok_text"] & df["ok_image"], "conf", False),
    ("ทายถูก: ข่าวจริงชัดเจน", df["ok_text+image"] & (df["label"] == 1) & df["ok_text"] & df["ok_image"], "conf", False),
    ("ข้อความถูก รูปผิด รวมแล้วถูก", df["ok_text"] & ~df["ok_image"] & df["ok_text+image"], "conf", False),
    ("รูปถูก ข้อความผิด รวมแล้วถูก", ~df["ok_text"] & df["ok_image"] & df["ok_text+image"], "conf", False),
    ("ทายผิด: รูปจริงแต่มาจาก subreddit ปลอม", ~df["ok_text+image"] & (df["label"] == 0), "conf", False),
    ("ทายผิด: ภาพต้นฉบับที่ดูเหมือนตัดต่อ", ~df["ok_text+image"] & (df["label"] == 1), "conf", False),
    ("ทายผิด: ข่าวเสียดสี", ~df["ok_text+image"] & df["subreddit"].isin(["theonion", "satire"]), "conf", False),
    ("ไม่แน่ใจ: ความน่าจะเป็นใกล้ 0.5", df["prob_real_text+image"].between(0.45, 0.55), "conf", True),
]
picked = []
for name, mask, key, asc in groups:
    cand = df[mask & ~df[ID_COL].isin([p[ID_COL] for p in picked])]
    cand = cand[~cand["subreddit"].isin([p["subreddit"] for p in picked])] if len(picked) < 6 else cand
    r = cand.sort_values(key, ascending=asc).iloc[0].to_dict()
    r["case"] = name
    picked.append(r)

out = pd.DataFrame(picked)
OUT_DIR.mkdir(parents=True, exist_ok=True)
for f in OUT_DIR.glob("*.jpg"):
    f.unlink()
for pid in out[ID_COL]:
    shutil.copy(ROOT / "data" / "images" / f"{pid}.jpg", OUT_DIR / f"{pid}.jpg")
out["image"] = out[ID_COL] + ".jpg"
out["label_name"] = out["label"].map({0: "ปลอม (0)", 1: "จริง (1)"})
cols = ["case", ID_COL, "clean_title", "image", "label", "label_name", "subreddit",
        "prob_real_text", "prob_real_image", "prob_real_text+image"]
out[cols].to_csv(OUT_DIR / "examples.csv", index=False)
print(out[cols].to_string(index=False))
