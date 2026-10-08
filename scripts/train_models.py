import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fakenews.data import ROOT
from fakenews.features import MODES, load_split_features

EMB_DIR = ROOT / "data" / "embeddings"
SPLIT_FILE = ROOT / "data" / "splits" / "split.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
SEED = 42


def make_model(C):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED)


X_train, y_train, _ = load_split_features(EMB_DIR, SPLIT_FILE, "train")
X_val, y_val, _ = load_split_features(EMB_DIR, SPLIT_FILE, "val")

rows = []
for mode in MODES:
    for C in C_GRID:
        clf = make_model(C).fit(X_train[mode], y_train)
        rows.append({
            "mode": mode,
            "C": C,
            "train_macro_f1": f1_score(y_train, clf.predict(X_train[mode]), average="macro"),
            "val_macro_f1": f1_score(y_val, clf.predict(X_val[mode]), average="macro"),
        })
        print(rows[-1])
grid = pd.DataFrame(rows)

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
chosen = {}
for mode in MODES:
    best = grid[grid["mode"] == mode].sort_values(["val_macro_f1", "C"], ascending=[False, True]).iloc[0]
    C = float(best["C"])
    clf = make_model(C).fit(X_train[mode], y_train)
    joblib.dump(clf, MODEL_DIR / f"logreg_{mode.replace('+', '_')}.joblib")
    chosen[mode] = {"C": C, "val_macro_f1": round(float(best["val_macro_f1"]), 4)}

grid.round(4).to_csv(RESULTS_DIR / "val_selection.csv", index=False)
(RESULTS_DIR / "chosen_params.json").write_text(json.dumps(
    {"model": "LogisticRegression", "class_weight": "balanced", "selected_on": "val macro-F1",
     "train_rows": int(len(y_train)), "val_rows": int(len(y_val)), "modes": chosen}, indent=2) + "\n")
print(json.dumps(chosen, indent=2))
