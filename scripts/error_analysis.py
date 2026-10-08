import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fakenews.data import ID_COL, ROOT, load_raw
from fakenews.features import MODES
from fakenews.images import image_path

RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
IMG_DIR = ROOT / "data" / "images"

pred = pd.read_csv(RESULTS_DIR / "test_predictions.csv")
sample = pd.read_csv(ROOT / "data" / "processed" / "sample.csv")[[ID_COL, "clean_title", "image_url"]]
subreddit = load_raw("train", cols=[ID_COL, "subreddit"])
df = pred.merge(sample, on=ID_COL).merge(subreddit, on=ID_COL)
assert len(df) == len(pred)

for mode in MODES:
    df[f"pred_{mode}"] = (df[f"prob_real_{mode}"] >= 0.5).astype(int)
    df[f"correct_{mode}"] = df[f"pred_{mode}"] == df["label"]

by_sub = df.groupby("subreddit").agg(
    n=("label", "size"), label=("label", "first"),
    **{f"acc_{m}": (f"correct_{m}", "mean") for m in MODES},
).sort_values("n", ascending=False).round(3)
by_sub.to_csv(RESULTS_DIR / "errors_by_subreddit.csv")

largest = by_sub.index[0]
rest = df["subreddit"] != largest
pd.DataFrame([
    {"mode": m, "macro_f1_all": f1_score(df["label"], df[f"pred_{m}"], average="macro"),
     f"macro_f1_without_{largest}": f1_score(df.loc[rest, "label"], df.loc[rest, f"pred_{m}"], average="macro")}
    for m in MODES
]).round(4).to_csv(RESULTS_DIR / "macro_f1_without_largest_subreddit.csv", index=False)

t, i, f = df["correct_text"], df["correct_image"], df["correct_text+image"]
agree = df["pred_text"] == df["pred_image"]
summary = pd.DataFrame([
    {"group": "text and image both correct", "n": int((t & i).sum()), "fusion_acc": f[t & i].mean()},
    {"group": "only text correct", "n": int((t & ~i).sum()), "fusion_acc": f[t & ~i].mean()},
    {"group": "only image correct", "n": int((~t & i).sum()), "fusion_acc": f[~t & i].mean()},
    {"group": "both wrong", "n": int((~t & ~i).sum()), "fusion_acc": f[~t & ~i].mean()},
    {"group": "text and image disagree", "n": int((~agree).sum()), "fusion_acc": f[~agree].mean()},
    {"group": "fusion fixes text error", "n": int((~t & f).sum()), "fusion_acc": None},
    {"group": "fusion breaks text-correct", "n": int((t & ~f).sum()), "fusion_acc": None},
]).round(3)
summary.to_csv(RESULTS_DIR / "modality_agreement.csv", index=False)

df["fusion_confidence"] = (df["prob_real_text+image"] - 0.5).abs() * 2
errors = df[~f].sort_values("fusion_confidence", ascending=False)
cols = [ID_COL, "label", "subreddit", "clean_title", "prob_real_text", "prob_real_image", "prob_real_text+image", "image_url"]
errors[cols].to_csv(RESULTS_DIR / "fusion_errors.csv", index=False)
disagree = df[~agree][cols + ["correct_text+image"]]
disagree.to_csv(RESULTS_DIR / "modality_disagreements.csv", index=False)

top = errors.head(12)
fig, axes = plt.subplots(3, 4, figsize=(14, 11.5), constrained_layout=True)
for ax, (_, r) in zip(axes.flat, top.iterrows()):
    ax.imshow(Image.open(image_path(IMG_DIR, r[ID_COL])))
    ax.set_axis_off()
    truth = "real" if r["label"] == 1 else "fake"
    title = textwrap.shorten(r["clean_title"], 70, placeholder="…")
    ax.set_title(
        f"{textwrap.fill(title, 34)}\ntrue {truth} · r/{r['subreddit']}\n"
        f"P(real) text {r['prob_real_text']:.2f} · image {r['prob_real_image']:.2f} · both {r['prob_real_text+image']:.2f}",
        fontsize=8.5, color="#1f1f1f", loc="left")
fig.suptitle("Most confident text+image errors on test set", fontsize=12, color="#555555")
fig.savefig(FIG_DIR / "error_examples.png", dpi=150, facecolor="white")

print(by_sub.to_string())
print()
print(summary.to_string(index=False))
print()
print(errors[cols[:-1]].head(15).to_string(index=False))
