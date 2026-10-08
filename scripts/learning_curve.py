import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fakenews.data import ROOT
from fakenews.features import MODES, load_split_features

RESULTS_DIR = ROOT / "results"
EMB_DIR = ROOT / "data" / "embeddings"
SPLIT_FILE = ROOT / "data" / "splits" / "split.csv"
FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
REPEATS = 5
SEED = 42
COLORS = {"text": "#2a78d6", "image": "#eb6834", "text+image": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

X_train, y_train, _ = load_split_features(EMB_DIR, SPLIT_FILE, "train")
X_val, y_val, _ = load_split_features(EMB_DIR, SPLIT_FILE, "val")
chosen = json.loads((RESULTS_DIR / "chosen_params.json").read_text())["modes"]

rows = []
for frac in FRACTIONS:
    for rep in range(1 if frac == 1.0 else REPEATS):
        if frac < 1.0:
            idx, _ = train_test_split(np.arange(len(y_train)), train_size=frac, stratify=y_train, random_state=SEED + rep)
        else:
            idx = np.arange(len(y_train))
        for mode in MODES:
            clf = LogisticRegression(C=chosen[mode]["C"], class_weight="balanced", max_iter=5000, random_state=SEED)
            clf.fit(X_train[mode][idx], y_train[idx])
            rows.append({"fraction": frac, "train_rows": len(idx), "repeat": rep, "mode": mode,
                         "val_macro_f1": f1_score(y_val, clf.predict(X_val[mode]), average="macro")})
    print(frac, "done")

raw = pd.DataFrame(rows)
curve = raw.groupby(["mode", "fraction", "train_rows"])["val_macro_f1"].agg(["mean", "min", "max"]).reset_index().round(4)
curve.to_csv(RESULTS_DIR / "learning_curve.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
for mode in MODES:
    c = curve[curve["mode"] == mode]
    ax.plot(c["train_rows"], c["mean"], color=COLORS[mode], linewidth=2, marker="o", markersize=6,
            markeredgecolor="white", label=mode)
    ax.fill_between(c["train_rows"], c["min"], c["max"], color=COLORS[mode], alpha=0.15, linewidth=0)
    last = c.iloc[-1]
    ax.annotate(f"{last['mean']:.3f}", (last["train_rows"], last["mean"]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color=MUTED)
ax.set_xlabel("training posts", color=MUTED)
ax.set_ylabel("validation macro-F1", color=MUTED)
ax.grid(color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
for side in ["left", "bottom"]:
    ax.spines[side].set_color(GRID)
ax.tick_params(colors=MUTED)
handles, labels = ax.get_legend_handles_labels()
order = curve[curve["fraction"] == 1.0].sort_values("mean", ascending=False)["mode"].tolist()
ax.legend([handles[labels.index(m)] for m in order], order, frameon=False, loc="lower right")
ax.set_title(f"Learning curve: mean of {REPEATS} random subsets per size, band = min to max",
             loc="left", fontsize=11, color=INK)
fig.savefig(RESULTS_DIR / "figures" / "learning_curve.png", dpi=200, facecolor="white")
print(curve.to_string(index=False))
