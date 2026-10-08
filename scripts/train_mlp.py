import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fakenews.data import ROOT
from fakenews.features import MODES, load_split_features

EMB_DIR = ROOT / "data" / "embeddings"
SPLIT_FILE = ROOT / "data" / "splits" / "split.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
HIDDEN = [(256,), (512,), (256, 128)]
ALPHAS = [1e-4, 1e-3, 1e-2]
THRESHOLDS = np.round(np.arange(0.2, 0.801, 0.02), 2)
SEED = 42


def make_model(hidden, alpha):
    return MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, early_stopping=True, validation_fraction=0.1,
                         n_iter_no_change=10, max_iter=300, random_state=SEED)


def best_threshold(y, prob):
    scores = [(f1_score(y, (prob >= t).astype(int), average="macro"), -abs(t - 0.5), t) for t in THRESHOLDS]
    f1, _, t = max(scores)
    return float(t), float(f1)


X_train, y_train, _ = load_split_features(EMB_DIR, SPLIT_FILE, "train")
X_val, y_val, _ = load_split_features(EMB_DIR, SPLIT_FILE, "val")

rows, chosen = [], {}
for mode in MODES:
    best = None
    for hidden in HIDDEN:
        for alpha in ALPHAS:
            t0 = time.time()
            clf = make_model(hidden, alpha).fit(X_train[mode], y_train)
            threshold, val_f1 = best_threshold(y_val, clf.predict_proba(X_val[mode])[:, 1])
            row = {"mode": mode, "hidden": str(hidden), "alpha": alpha, "threshold": threshold,
                   "val_macro_f1": round(val_f1, 4), "epochs": clf.n_iter_, "seconds": round(time.time() - t0, 1)}
            rows.append(row)
            print(row, flush=True)
            if best is None or val_f1 > best[0]:
                best = (val_f1, hidden, alpha, threshold, clf)
    val_f1, hidden, alpha, threshold, clf = best
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": clf, "threshold": threshold}, MODEL_DIR / f"mlp_{mode.replace('+', '_')}.joblib")
    chosen[mode] = {"hidden": list(hidden), "alpha": alpha, "threshold": threshold, "val_macro_f1": round(val_f1, 4)}

pd.DataFrame(rows).to_csv(RESULTS_DIR / "mlp_val_selection.csv", index=False)
(RESULTS_DIR / "mlp_chosen_params.json").write_text(json.dumps(
    {"model": "MLPClassifier (relu, adam, early stopping on 10% of train)",
     "selected_on": "val macro-F1, decision threshold also tuned on val", "modes": chosen}, indent=2) + "\n")
print(json.dumps(chosen, indent=2))
