import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fakenews.data import ID_COL, ROOT
from fakenews.features import MODES

RESULTS_DIR = ROOT / "results"
N_BOOT = 2000
SEED = 42

clip = pd.read_csv(RESULTS_DIR / "test_predictions.csv")
base = pd.read_csv(RESULTS_DIR / "baseline_predictions.csv")
assert (clip[ID_COL] == base[ID_COL]).all()
y = clip["label"].to_numpy()
preds = {f"clip_{m}": (clip[f"prob_real_{m}"] >= 0.5).astype(int).to_numpy() for m in MODES}
preds["tfidf"] = base["pred_tfidf_logreg"].to_numpy()

rng = np.random.default_rng(SEED)
boot_idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))


def macro_f1(pred, idx=slice(None)):
    return f1_score(y[idx], pred[idx], average="macro")


boot = {name: np.array([macro_f1(p, idx) for idx in boot_idx]) for name, p in preds.items()}

intervals = pd.DataFrame([
    {"model": name, "macro_f1": macro_f1(p), "ci95_low": np.percentile(boot[name], 2.5),
     "ci95_high": np.percentile(boot[name], 97.5)}
    for name, p in preds.items()
]).round(4)

pairs = [("clip_text+image", "clip_text"), ("clip_text+image", "clip_image"),
         ("clip_text", "clip_image"), ("clip_text", "tfidf")]
rows = []
for a, b in pairs:
    diff = boot[a] - boot[b]
    ca, cb = preds[a] == y, preds[b] == y
    only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
    rows.append({
        "comparison": f"{a} vs {b}",
        "macro_f1_diff": macro_f1(preds[a]) - macro_f1(preds[b]),
        "diff_ci95_low": np.percentile(diff, 2.5),
        "diff_ci95_high": np.percentile(diff, 97.5),
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "mcnemar_exact_p": binomtest(only_a, only_a + only_b, 0.5).pvalue,
    })
tests = pd.DataFrame(rows)
tests[["macro_f1_diff", "diff_ci95_low", "diff_ci95_high"]] = tests[["macro_f1_diff", "diff_ci95_low", "diff_ci95_high"]].round(4)
tests["mcnemar_exact_p"] = tests["mcnemar_exact_p"].map(lambda p: float(f"{p:.2e}"))

intervals.to_csv(RESULTS_DIR / "macro_f1_ci.csv", index=False)
tests.to_csv(RESULTS_DIR / "significance_tests.csv", index=False)
print(f"bootstrap: {N_BOOT} resamples of {len(y)} test rows, seed {SEED}\n")
print(intervals.to_string(index=False))
print()
print(tests.to_string(index=False))
