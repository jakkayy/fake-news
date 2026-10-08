import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fakenews.data import ID_COL, ROOT, load_raw
from fakenews.features import MODES, build_features

RESULTS_DIR = ROOT / "results"
EMB_DIR = ROOT / "data" / "embeddings"
MIN_POSTS = 100
SEED = 42

ids = pd.read_csv(EMB_DIR / "ids.csv")
meta = ids.merge(pd.read_csv(ROOT / "data" / "splits" / "split.csv"), on=ID_COL, how="left")
meta = meta.merge(load_raw("train", cols=[ID_COL, "subreddit"]), on=ID_COL, how="left")
assert (meta[ID_COL] == ids[ID_COL]).all()
text, image = np.load(EMB_DIR / "text.npy"), np.load(EMB_DIR / "image.npy")
y = np.load(EMB_DIR / "labels.npy")
X = {m: build_features(m, text, image) for m in MODES}
chosen = json.loads((RESULTS_DIR / "chosen_params.json").read_text())["modes"]
in_dist = pd.read_csv(RESULTS_DIR / "errors_by_subreddit.csv").set_index("subreddit")

counts = meta["subreddit"].value_counts()
rows = []
for sub in counts[counts >= MIN_POSTS].index:
    held = (meta["subreddit"] == sub).to_numpy()
    train = (meta["split"] == "train").to_numpy() & ~held
    row = {"subreddit": sub, "label": int(y[held][0]), "held_out_posts": int(held.sum()),
           "in_dist_test_posts": int(in_dist.loc[sub, "n"]) if sub in in_dist.index else 0}
    for mode in MODES:
        clf = LogisticRegression(C=chosen[mode]["C"], class_weight="balanced", max_iter=5000, random_state=SEED)
        clf.fit(X[mode][train], y[train])
        row[f"held_out_acc_{mode}"] = (clf.predict(X[mode][held]) == y[held]).mean()
        row[f"in_dist_acc_{mode}"] = in_dist.loc[sub, f"acc_{mode}"] if sub in in_dist.index else np.nan
    rows.append(row)
    print(sub, {k: round(v, 3) for k, v in row.items() if k.startswith(("held", "in_dist_acc")) and "posts" not in k})

out = pd.DataFrame(rows).round(3)
out.to_csv(RESULTS_DIR / "leave_subreddit_out.csv", index=False)

summary = []
for mode in MODES:
    for label in [0, 1]:
        part = out[out["label"] == label]
        summary.append({"mode": mode, "label": label, "subreddits": len(part),
                        "mean_in_dist_acc": part[f"in_dist_acc_{mode}"].mean(),
                        "mean_held_out_acc": part[f"held_out_acc_{mode}"].mean()})
summary = pd.DataFrame(summary).round(3)
summary.to_csv(RESULTS_DIR / "leave_subreddit_out_summary.csv", index=False)
print()
print(summary.to_string(index=False))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEN, UNSEEN = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
mode = "text+image"
plot = out.assign(drop=out[f"in_dist_acc_{mode}"] - out[f"held_out_acc_{mode}"]).sort_values("drop")
fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
for i, (a, b) in enumerate(zip(plot[f"in_dist_acc_{mode}"], plot[f"held_out_acc_{mode}"])):
    ax.plot([a, b], [i, i], color=GRID, linewidth=2.5, zorder=1)
ax.scatter(plot[f"in_dist_acc_{mode}"], range(len(plot)), s=64, color=SEEN, edgecolor="white", linewidth=1.5, zorder=2, label="subreddit seen in training (test split)")
ax.scatter(plot[f"held_out_acc_{mode}"], range(len(plot)), s=64, color=UNSEEN, edgecolor="white", linewidth=1.5, zorder=3, label="subreddit held out of training")
ax.set_yticks(range(len(plot)), [f"{s} ({'real' if l == 1 else 'fake'})" for s, l in zip(plot["subreddit"], plot["label"])])
ax.set_xlim(0, 1.02)
ax.set_xlabel("accuracy on the subreddit's posts (text+image)", color=MUTED)
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0, colors=MUTED)
ax.tick_params(axis="x", colors=MUTED)
ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.08), ncol=2, frameon=False, fontsize=9)
ax.set_title("Accuracy drops when a subreddit is never seen in training\nsorted by size of the drop (largest at top)",
             loc="left", fontsize=11, color=INK)
fig.savefig(RESULTS_DIR / "figures" / "leave_subreddit_out.png", dpi=200, facecolor="white")
