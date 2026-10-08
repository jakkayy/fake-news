import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fakenews.data import ID_COL, ROOT
from fakenews.features import MODES, load_split_features

EMB_DIR = ROOT / "data" / "embeddings"
SPLIT_FILE = ROOT / "data" / "splits" / "split.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
N_BOOT = 2000
SEED = 42

X_test, y, ids = load_split_features(EMB_DIR, SPLIT_FILE, "test")
lr_pred = pd.read_csv(RESULTS_DIR / "test_predictions.csv")
assert (lr_pred[ID_COL].to_numpy() == ids).all()

rng = np.random.default_rng(SEED)
boot_idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))


def f1(pred, idx=slice(None)):
    return f1_score(y[idx], pred[idx], average="macro")


rows, preds = [], {ID_COL: ids, "label": y}
for mode in MODES:
    saved = joblib.load(MODEL_DIR / f"mlp_{mode.replace('+', '_')}.joblib")
    prob = saved["model"].predict_proba(X_test[mode])[:, 1]
    mlp = (prob >= saved["threshold"]).astype(int)
    lr = (lr_pred[f"prob_real_{mode}"].to_numpy() >= 0.5).astype(int)
    preds[f"prob_real_mlp_{mode}"] = prob.round(4)
    boot_mlp = np.array([f1(mlp, k) for k in boot_idx])
    boot_lr = np.array([f1(lr, k) for k in boot_idx])
    diff = boot_mlp - boot_lr
    only_mlp, only_lr = int(((mlp == y) & (lr != y)).sum()), int(((mlp != y) & (lr == y)).sum())
    rows.append({
        "mode": mode,
        "logreg_macro_f1": f1(lr), "logreg_ci95": f"{np.percentile(boot_lr, 2.5):.3f}-{np.percentile(boot_lr, 97.5):.3f}",
        "mlp_macro_f1": f1(mlp), "mlp_ci95": f"{np.percentile(boot_mlp, 2.5):.3f}-{np.percentile(boot_mlp, 97.5):.3f}",
        "mlp_accuracy": accuracy_score(y, mlp),
        "diff_mlp_minus_logreg": f1(mlp) - f1(lr),
        "diff_ci95_low": np.percentile(diff, 2.5), "diff_ci95_high": np.percentile(diff, 97.5),
        "only_mlp_correct": only_mlp, "only_logreg_correct": only_lr,
        "mcnemar_exact_p": float(f"{binomtest(only_mlp, only_mlp + only_lr, 0.5).pvalue:.3g}"),
    })

out = pd.DataFrame(rows).round(4)
out.to_csv(RESULTS_DIR / "mlp_vs_logreg.csv", index=False)
pd.DataFrame(preds).to_csv(RESULTS_DIR / "mlp_test_predictions.csv", index=False)
print(out.to_string(index=False))
