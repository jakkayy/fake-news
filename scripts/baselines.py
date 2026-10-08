import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fakenews.data import ID_COL, LABEL_COL, ROOT, TEXT_COL

RESULTS_DIR = ROOT / "results"
C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
SEED = 42

df = pd.read_csv(ROOT / "data" / "processed" / "sample.csv")
split = pd.read_csv(ROOT / "data" / "splits" / "split.csv")[[ID_COL, "split"]]
df = df.drop(columns="split").merge(split, on=ID_COL)
train, val, test = (df[df["split"] == s] for s in ["train", "val", "test"])
pred_order = pd.read_csv(RESULTS_DIR / "test_predictions.csv")[ID_COL]
test = test.set_index(ID_COL).loc[pred_order].reset_index()

rows, preds = [], {ID_COL: test[ID_COL], "label": test[LABEL_COL]}


def record(name, pred, params=None):
    preds[f"pred_{name}"] = pred
    rows.append({"model": name, "macro_f1": f1_score(test[LABEL_COL], pred, average="macro"),
                 "accuracy": accuracy_score(test[LABEL_COL], pred), "params": json.dumps(params or {})})


for strategy in ["most_frequent", "stratified"]:
    dummy = DummyClassifier(strategy=strategy, random_state=SEED).fit(train[[TEXT_COL]], train[LABEL_COL])
    record(f"dummy_{strategy}", dummy.predict(test[[TEXT_COL]]))

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_train = vec.fit_transform(train[TEXT_COL])
X_val, X_test = vec.transform(val[TEXT_COL]), vec.transform(test[TEXT_COL])
scores = []
for C in C_GRID:
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED).fit(X_train, train[LABEL_COL])
    scores.append((f1_score(val[LABEL_COL], clf.predict(X_val), average="macro"), -C, C))
best_val, _, best_C = max(scores)
clf = LogisticRegression(C=best_C, class_weight="balanced", max_iter=5000, random_state=SEED).fit(X_train, train[LABEL_COL])
record("tfidf_logreg", clf.predict(X_test),
       {"C": best_C, "val_macro_f1": round(best_val, 4), "ngram_range": [1, 2], "vocab_size": len(vec.vocabulary_)})

clip = pd.read_csv(RESULTS_DIR / "test_metrics.csv")
for _, r in clip.iterrows():
    rows.append({"model": f"clip_logreg_{r['mode']}", "macro_f1": r["macro_f1"], "accuracy": r["accuracy"], "params": "{}"})

out = pd.DataFrame(rows).round(4).sort_values("macro_f1")
out.to_csv(RESULTS_DIR / "baselines.csv", index=False)
pd.DataFrame(preds).to_csv(RESULTS_DIR / "baseline_predictions.csv", index=False)
print(out.to_string(index=False))
