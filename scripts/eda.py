import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fakenews.data import ID_COL, LABEL_COL, ROOT, TEXT_COL, load_raw

RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
REAL, FAKE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK})

sample = pd.read_csv(ROOT / "data" / "processed" / "sample.csv")
sample = sample.merge(load_raw("train", cols=[ID_COL, "subreddit"]), on=ID_COL)
stats = json.loads((ROOT / "data" / "splits" / "prepare_stats.json").read_text())
sample["title_words"] = sample[TEXT_COL].str.split().str.len()

counts = sample.groupby("subreddit").agg(n=(LABEL_COL, "size"), label=(LABEL_COL, "first")).sort_values("n")
fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
ax.barh(counts.index, counts["n"], height=0.7, color=[REAL if l == 1 else FAKE for l in counts["label"]],
        edgecolor="white", linewidth=1)
for y, (n, share) in enumerate(zip(counts["n"], counts["n"] / counts["n"].sum())):
    ax.text(n + 40, y, f"{n:,} ({share:.1%})", va="center", fontsize=8.5, color=MUTED)
ax.set_xlim(0, counts["n"].max() * 1.22)
ax.set_xlabel("posts in sample")
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=FAKE), plt.Rectangle((0, 0), 1, 1, color=REAL)],
          labels=["fake (0)", "real (1)"], loc="lower right", frameon=False)
ax.set_title(f"Posts per subreddit in the sample (n = {len(sample):,})\nEvery subreddit has a single label",
             loc="left", fontsize=11)
fig.savefig(FIG_DIR / "eda_subreddits.png", dpi=200, facecolor="white")

cap = 30
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
for label, color, name in [(0, FAKE, "fake (0)"), (1, REAL, "real (1)")]:
    words = sample.loc[sample[LABEL_COL] == label, "title_words"].clip(upper=cap)
    share = words.value_counts(normalize=True).reindex(range(1, cap + 1), fill_value=0)
    ax.plot(share.index, share.values * 100, color=color, linewidth=2, label=f"{name}, median {int(words.median())} words")
ax.set_xlabel(f"words in clean_title ({cap} = {cap} or more)")
ax.set_ylabel("% of posts in class")
ax.grid(axis="y", color=GRID, linewidth=0.8)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
ax.legend(frameon=False)
ax.set_title("Title length by class: fake titles are shorter", loc="left", fontsize=11)
fig.savefig(FIG_DIR / "eda_title_length.png", dpi=200, facecolor="white")

steps = [("rows in multimodal_train.tsv", stats["raw_rows"]),
         ("has image_url", stats["after_drop_null_url"]),
         (f"drop {stats['conflicting_label_urls_dropped']} URLs with conflicting labels", stats["after_drop_conflicts"]),
         ("one post per image_url", stats["after_dedup_url"]),
         (f"stratified sample (seed {stats['seed']})", stats["sampled"]),
         ("image downloaded and readable", stats["final_rows"])]
fails = stats["download_failed"]
length = sample.groupby(LABEL_COL)["title_words"].describe()[["mean", "50%", "max"]].round(1)
lines = [
    "# Exploration summary", "",
    "สร้างจาก `scripts/eda.py` ตัวเลขทั้งหมดมาจาก `data/splits/prepare_stats.json` และ `data/processed/sample.csv`", "",
    "## ขั้นการกรองข้อมูล", "", "| ขั้น | แถวที่เหลือ |", "|---|---|",
    *[f"| {name} | {n:,} |" for name, n in steps], "",
    f"รูปที่โหลดไม่ได้ {stats['sampled'] - stats['final_rows']:,} รูป ({1 - stats['final_rows'] / stats['sampled']:.1%}): "
    + ", ".join(f"{k} {v}" for k, v in fails.items()), "",
    "## สัดส่วนคลาสในแต่ละ split", "", "| split | แถว | ข่าวจริง (1) |", "|---|---|---|",
    *[f"| {s} | {v['rows']:,} | {v['real_ratio']:.1%} |" for s, v in stats["splits"].items()], "",
    "## ความยาวหัวข้อ (คำ)", "", "| คลาส | เฉลี่ย | มัธยฐาน | สูงสุด |", "|---|---|---|---|",
    *[f"| {'real (1)' if l == 1 else 'fake (0)'} | {r['mean']} | {r['50%']:.0f} | {r['max']:.0f} |" for l, r in length.iterrows()], "",
    "## subreddit", "",
    f"มี {len(counts)} subreddit ในตัวอย่าง แต่ละ subreddit มีป้ายเดียว จึงห้ามใช้เป็นฟีเจอร์ "
    f"ใหญ่สุดคือ r/{counts.index[-1]} ({counts['n'].iloc[-1] / counts['n'].sum():.1%}) ดู `figures/eda_subreddits.png`", "",
]
(RESULTS_DIR / "eda_summary.md").write_text("\n".join(lines))
print("\n".join(lines))
