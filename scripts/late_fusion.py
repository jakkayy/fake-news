import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fakenews.data import ROOT
from fakenews.features import load_split_features

RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"
EMB_DIR = ROOT / "data" / "embeddings"
SPLIT_FILE = ROOT / "data" / "splits" / "split.csv"
WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)
N_BOOT = 2000
SEED = 42

models = {m: joblib.load(MODEL_DIR / f"logreg_{m.replace('+', '_')}.joblib") for m in ["text", "image", "text+image"]}


def probs(split):
    X, y, _ = load_split_features(EMB_DIR, SPLIT_FILE, split)
    return {m: clf.predict_proba(X[m])[:, 1] for m, clf in models.items()}, y


def late(p, w):
    return w * p["text"] + (1 - w) * p["image"]


def macro_f1(y, prob):
    return f1_score(y, (prob >= 0.5).astype(int), average="macro")


p_val, y_val = probs("val")
val_scores = {float(w): macro_f1(y_val, late(p_val, w)) for w in WEIGHTS}
best_w = max(val_scores, key=lambda w: (val_scores[w], -abs(w - 0.5)))

p_test, y_test = probs("test")
pred = {
    "late_fusion_mean": (late(p_test, 0.5) >= 0.5).astype(int),
    f"late_fusion_w{best_w}": (late(p_test, best_w) >= 0.5).astype(int),
    "early_fusion": (p_test["text+image"] >= 0.5).astype(int),
}
rng = np.random.default_rng(SEED)
idx = rng.integers(0, len(y_test), size=(N_BOOT, len(y_test)))
a, b = pred[f"late_fusion_w{best_w}"], pred["early_fusion"]
diff = np.array([f1_score(y_test[k], a[k], average="macro") - f1_score(y_test[k], b[k], average="macro") for k in idx])
only_a, only_b = int(((a == y_test) & (b != y_test)).sum()), int(((a != y_test) & (b == y_test)).sum())

result = {
    "late_fusion": "P(real) = w * P_text + (1 - w) * P_image, w chosen on val macro-F1",
    "val_macro_f1_by_w": {str(w): round(s, 4) for w, s in val_scores.items()},
    "chosen_w": best_w,
    "test_macro_f1": {k: round(float(f1_score(y_test, v, average="macro")), 4) for k, v in pred.items()},
    "late_vs_early": {
        "macro_f1_diff": round(float(f1_score(y_test, a, average="macro") - f1_score(y_test, b, average="macro")), 4),
        "diff_ci95": [round(float(np.percentile(diff, 2.5)), 4), round(float(np.percentile(diff, 97.5)), 4)],
        "only_late_correct": only_a, "only_early_correct": only_b,
        "mcnemar_exact_p": float(f"{binomtest(only_a, only_a + only_b, 0.5).pvalue:.3g}"),
    },
}
(RESULTS_DIR / "late_fusion.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "val_macro_f1_by_w"}, indent=2))
