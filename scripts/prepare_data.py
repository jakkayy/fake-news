import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fakenews.data import ID_COL, LABEL_COL, ROOT, URL_COL, load_raw
from fakenews.images import download_with_retry, image_path

SEED = 42
IMG_DIR = ROOT / "data" / "images"
SPLIT_DIR = ROOT / "data" / "splits"
PROCESSED_DIR = ROOT / "data" / "processed"

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=16500)
parser.add_argument("--workers", type=int, default=32)
parser.add_argument("--short-side", type=int, default=256)
args = parser.parse_args()

stats = {"seed": SEED}
df = load_raw("train")
stats["raw_rows"] = len(df)

df = df.dropna(subset=[URL_COL])
stats["after_drop_null_url"] = len(df)

n_labels = df.groupby(URL_COL)[LABEL_COL].nunique()
conflict = n_labels[n_labels > 1].index
df = df[~df[URL_COL].isin(conflict)]
stats["conflicting_label_urls_dropped"] = len(conflict)
stats["after_drop_conflicts"] = len(df)

df = df.drop_duplicates(subset=[URL_COL])
stats["after_dedup_url"] = len(df)

df, _ = train_test_split(df, train_size=args.n, stratify=df[LABEL_COL], random_state=SEED)
df = df.reset_index(drop=True)
stats["sampled"] = len(df)
print(json.dumps(stats, indent=2))

t = time.time()
status = download_with_retry(df, IMG_DIR, workers=args.workers, short_side=args.short_side)
stats["download_seconds"] = round(time.time() - t)
ok_ids = set(status.loc[status["status"] == "ok", ID_COL])
stats["download_ok"] = len(ok_ids)
stats["download_failed"] = status.loc[status["status"] != "ok", "reason"].value_counts().to_dict()
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
status[status["status"] != "ok"].to_csv(PROCESSED_DIR / "failed_downloads.csv", index=False)

df = df[df[ID_COL].isin(ok_ids)]
unreadable = []
for pid in df[ID_COL]:
    try:
        with Image.open(image_path(IMG_DIR, pid)) as im:
            im.verify()
    except Exception:
        unreadable.append(pid)
df = df[~df[ID_COL].isin(unreadable)].sort_values(ID_COL).reset_index(drop=True)
stats["unreadable_after_download"] = len(unreadable)
stats["final_rows"] = len(df)

rest, test = train_test_split(df, test_size=0.15, stratify=df[LABEL_COL], random_state=SEED)
train, val = train_test_split(rest, test_size=0.15 / 0.85, stratify=rest[LABEL_COL], random_state=SEED)
split = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
df = df.merge(split[[ID_COL, "split"]], on=ID_COL)

assert df[ID_COL].is_unique and df[URL_COL].is_unique
assert all(image_path(IMG_DIR, i).exists() for i in df[ID_COL])

stats["splits"] = {
    s: {"rows": int((df["split"] == s).sum()),
        "real_ratio": round(float(df.loc[df["split"] == s, LABEL_COL].mean()), 4)}
    for s in ["train", "val", "test"]
}

SPLIT_DIR.mkdir(parents=True, exist_ok=True)
df[[ID_COL, "split", LABEL_COL]].to_csv(SPLIT_DIR / "split.csv", index=False)
df.to_csv(PROCESSED_DIR / "sample.csv", index=False)
(SPLIT_DIR / "prepare_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(stats, indent=2, ensure_ascii=False))
