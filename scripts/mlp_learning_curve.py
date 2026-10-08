import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fakenews.data import ROOT
from fakenews.features import load_split_features

RESULTS_DIR = ROOT / "results"
EMB_DIR = ROOT / "data" / "embeddings"
SPLIT_FILE = ROOT / "data" / "splits" / "split.csv"
MODE = "text+image"
FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
REPEATS = 3
THRESHOLDS = np.round(np.arange(0.2, 0.801, 0.02), 2)
SEED = 42
COLORS = {"logistic regression": "#2a78d6", "MLP": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

X_train, y_train, _ = load_split_features(EMB_DIR, SPLIT_FILE, "train")
X_val, y_val, _ = load_split_features(EMB_DIR, SPLIT_FILE, "val")
params = json.loads((RESULTS_DIR / "mlp_chosen_params.json").read_text())["modes"][MODE]

rows = []
for frac in FRACTIONS:
    for rep in range(REPEATS):
        idx = np.arange(len(y_train)) if frac == 1.0 else train_test_split(
            np.arange(len(y_train)), train_size=frac, stratify=y_train, random_state=SEED + rep)[0]
        clf = MLPClassifier(hidden_layer_sizes=tuple(params["hidden"]), alpha=params["alpha"], early_stopping=True,
                            validation_fraction=0.1, n_iter_no_change=10, max_iter=300, random_state=SEED + 100 + rep)
        clf.fit(X_train[MODE][idx], y_train[idx])
        prob = clf.predict_proba(X_val[MODE])[:, 1]
        f1 = max(f1_score(y_val, (prob >= t).astype(int), average="macro") for t in THRESHOLDS)
        rows.append({"fraction": frac, "train_rows": len(idx), "repeat": rep, "val_macro_f1": f1})
    print(frac, "done", flush=True)

mlp = pd.DataFrame(rows).groupby(["fraction", "train_rows"])["val_macro_f1"].agg(["mean", "min", "max"]).reset_index().round(4)
mlp.to_csv(RESULTS_DIR / "mlp_learning_curve.csv", index=False)
lr = pd.read_csv(RESULTS_DIR / "learning_curve.csv")
lr = lr[lr["mode"] == MODE]

fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
for name, c in [("MLP", mlp), ("logistic regression", lr)]:
    ax.plot(c["train_rows"], c["mean"], color=COLORS[name], linewidth=2, marker="o", markersize=6,
            markeredgecolor="white", label=name)
    ax.fill_between(c["train_rows"], c["min"], c["max"], color=COLORS[name], alpha=0.15, linewidth=0)
    last = c.iloc[-1]
    ax.annotate(f"{last['mean']:.3f}", (last["train_rows"], last["mean"]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color=MUTED)
handles, labels = ax.get_legend_handles_labels()
order = sorted(labels, key=lambda n: -(mlp if n == "MLP" else lr)["mean"].iloc[-1])
ax.legend([handles[labels.index(n)] for n in order], order, frameon=False, loc="lower right")
ax.set_xlabel("training posts", color=MUTED)
ax.set_ylabel("validation macro-F1 (text+image)", color=MUTED)
ax.grid(color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
for side in ["left", "bottom"]:
    ax.spines[side].set_color(GRID)
ax.tick_params(colors=MUTED)
ax.set_title("Learning curve by classifier (text+image)\nMLP: 3 runs per size, threshold tuned on val; band = min to max",
             loc="left", fontsize=11, color=INK)
fig.savefig(RESULTS_DIR / "figures" / "mlp_learning_curve.png", dpi=200, facecolor="white")
print(mlp.to_string(index=False))
