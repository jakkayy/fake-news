import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fakenews.data import ROOT
from fakenews.embed_store import load_embeddings

RESULTS_DIR = ROOT / "results"
EMB_DIR = ROOT / "data" / "embeddings"
SPLIT_FILE = ROOT / "data" / "splits" / "split.csv"
C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
N_BOOT = 2000
SEED = 42

data = {s: load_embeddings(EMB_DIR, SPLIT_FILE, s) for s in ["train", "val", "test"]}
sim = {s: (t * i).sum(axis=1, keepdims=True) for s, (t, i, _, _) in data.items()}
mu, sd = sim["train"].mean(), sim["train"].std()


def features(split, kind):
    t, i, _, _ = data[split]
    z = (sim[split] - mu) / sd
    if kind == "similarity":
        return z
    if kind == "text+image":
        return np.hstack([t, i])
    if kind == "text+image+similarity":
        return np.hstack([t, i, z])
    raise ValueError(kind)


y = {s: data[s][2] for s in data}
by_label = pd.DataFrame({"similarity": sim["train"][:, 0], "label": y["train"]}).groupby("label")["similarity"].describe()

rows, test_preds = [], {}
for kind in ["similarity", "text+image", "text+image+similarity"]:
    scores = []
    for C in C_GRID:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED)
        clf.fit(features("train", kind), y["train"])
        scores.append((f1_score(y["val"], clf.predict(features("val", kind)), average="macro"), -C, C))
    val_f1, _, C = max(scores)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED).fit(features("train", kind), y["train"])
    test_preds[kind] = clf.predict(features("test", kind))
    rows.append({"features": kind, "C": C, "val_macro_f1": val_f1,
                 "test_macro_f1": f1_score(y["test"], test_preds[kind], average="macro"),
                 "similarity_coef": float(clf.coef_[0, -1]) if "similarity" in kind else None})
results = pd.DataFrame(rows).round(4)

yt = y["test"]
rng = np.random.default_rng(SEED)
idx = rng.integers(0, len(yt), size=(N_BOOT, len(yt)))
a, b = test_preds["text+image+similarity"], test_preds["text+image"]
diff = np.array([f1_score(yt[k], a[k], average="macro") - f1_score(yt[k], b[k], average="macro") for k in idx])
only_a, only_b = int(((a == yt) & (b != yt)).sum()), int(((a != yt) & (b == yt)).sum())
comparison = {
    "comparison": "text+image+similarity vs text+image (test)",
    "macro_f1_diff": round(float(f1_score(yt, a, average="macro") - f1_score(yt, b, average="macro")), 4),
    "diff_ci95": [round(float(np.percentile(diff, 2.5)), 4), round(float(np.percentile(diff, 97.5)), 4)],
    "only_with_similarity_correct": only_a, "only_without_similarity_correct": only_b,
    "mcnemar_exact_p": float(f"{binomtest(only_a, only_a + only_b, 0.5).pvalue:.3g}") if only_a + only_b else 1.0,
    "similarity_auc_real_vs_fake_train": round(float(roc_auc_score(y["train"], sim["train"][:, 0])), 4),
    "similarity_by_label_train": {int(k): {"mean": round(float(v["mean"]), 4), "std": round(float(v["std"]), 4)}
                                  for k, v in by_label.iterrows()},
}
results.to_csv(RESULTS_DIR / "similarity_feature.csv", index=False)
(RESULTS_DIR / "similarity_feature_test.json").write_text(json.dumps(comparison, indent=2) + "\n")
print(results.to_string(index=False))
print(json.dumps(comparison, indent=2))

WARN_PERCENTILE = 15
train_sim = sim["train"][:, 0]
threshold = float(np.percentile(train_sim, WARN_PERCENTILE))
shuffled = (data["train"][0] * data["train"][1][rng.permutation(len(train_sim))]).sum(axis=1)
warning = {
    "rule": f"warn when cos(title, image) is below the {WARN_PERCENTILE}th percentile of the training split",
    "percentile": WARN_PERCENTILE,
    "threshold": round(threshold, 4),
    "train_percentiles": {str(q): round(float(np.percentile(train_sim, q)), 4) for q in [1, 5, 10, 15, 25, 50]},
    "share_of_real_train_posts_flagged": round(float((train_sim[y["train"] == 1] < threshold).mean()), 4),
    "share_of_random_mismatched_pairs_flagged": round(float((shuffled < threshold).mean()), 4),
}
(RESULTS_DIR / "similarity_warning.json").write_text(json.dumps(warning, indent=2) + "\n")
print(json.dumps(warning, indent=2))
