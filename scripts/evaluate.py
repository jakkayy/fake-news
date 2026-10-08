import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fakenews.data import ROOT
from fakenews.features import MODES, load_split_features

EMB_DIR = ROOT / "data" / "embeddings"
SPLIT_FILE = ROOT / "data" / "splits" / "split.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
CLASS_NAMES = ["fake (0)", "real (1)"]

X_test, y_test, ids = load_split_features(EMB_DIR, SPLIT_FILE, "test")

rows, cms, preds = [], {}, {"id": ids, "label": y_test}
for mode in MODES:
    clf = joblib.load(MODEL_DIR / f"logreg_{mode.replace('+', '_')}.joblib")
    prob = clf.predict_proba(X_test[mode])[:, 1]
    pred = (prob >= 0.5).astype(int)
    preds[f"prob_real_{mode}"] = prob.round(4)
    p, r, f, _ = precision_recall_fscore_support(y_test, pred, labels=[0, 1])
    rows.append({
        "mode": mode,
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "accuracy": accuracy_score(y_test, pred),
        "precision_fake": p[0], "recall_fake": r[0], "f1_fake": f[0],
        "precision_real": p[1], "recall_real": r[1], "f1_real": f[1],
    })
    cms[mode] = confusion_matrix(y_test, pred, labels=[0, 1])

metrics = pd.DataFrame(rows).round(4)
FIG_DIR.mkdir(parents=True, exist_ok=True)
metrics.to_csv(RESULTS_DIR / "test_metrics.csv", index=False)
pd.DataFrame(preds).to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
(RESULTS_DIR / "confusion_matrices.json").write_text(json.dumps(
    {"rows": "true label", "cols": "predicted label", "labels": CLASS_NAMES,
     "test_rows": int(len(y_test)), "matrices": {m: cm.tolist() for m, cm in cms.items()}}, indent=2) + "\n")

cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
for ax, mode in zip(axes, MODES):
    cm = cms[mode]
    share = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=12,
                    color="white" if share[i, j] > 0.55 else "#1f1f1f")
    f1 = metrics.loc[metrics["mode"] == mode, "macro_f1"].item()
    ax.set_title(f"{mode}  ·  macro-F1 {f1:.3f}", fontsize=12, color="#1f1f1f")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("predicted", color="#555555")
    ax.tick_params(length=0, colors="#555555")
    for s in ax.spines.values():
        s.set_visible(False)
axes[0].set_ylabel("true", color="#555555")
fig.suptitle(f"Confusion matrices on test set (n = {len(y_test):,}); cell shade = share of true class",
             fontsize=11, color="#555555")
fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=200, facecolor="white")

print(metrics.to_string(index=False))
