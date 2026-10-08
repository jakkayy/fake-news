import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fakenews.data import ID_COL, LABEL_COL, ROOT, TEXT_COL, load_raw
from fakenews.embed import embed_texts, load_clip
from fakenews.text import light_clean

RESULTS_DIR = ROOT / "results"
EMB_DIR = ROOT / "data" / "embeddings"
SEED = 42

ids = pd.read_csv(EMB_DIR / "ids.csv")
df = ids.merge(pd.read_csv(ROOT / "data" / "processed" / "sample.csv"), on=ID_COL, how="left")
df = df.merge(load_raw("train", cols=[ID_COL, "title", "subreddit"]), on=ID_COL, how="left")
assert (df[ID_COL] == ids[ID_COL]).all()
t = df[TEXT_COL]

audit = {
    "rows": len(df),
    "empty": int((t.str.strip() == "").sum()),
    "uppercase": int(t.str.contains("[A-Z]").sum()),
    "punctuation": int(t.str.contains(r"[^\w\s]").sum()),
    "digits": int(t.str.contains(r"\d").sum()),
    "non_ascii": int(t.str.contains(r"[^\x00-\x7f]").sum()),
    "extra_whitespace": int((t.str.contains("  ") | t.ne(t.str.strip())).sum()),
    "url_fragments": int(t.str.contains(URL := r"\b(?:https?\w*|www\w*)").sum()),
    "one_word": int((t.str.split().str.len() == 1).sum()),
    "over_50_words": int((t.str.split().str.len() > 50).sum()),
}
raw_prefix = df["title"].str.match(r"^\s*psbattle", case=False)
prefix_by_sub = df[raw_prefix].groupby("subreddit").size().to_dict()
raw_digits = int(df["title"].str.contains(r"\d").sum())

df["light_clean"] = t.map(light_clean)
changed = df["light_clean"] != t

model, processor, device = load_clip()
text_clean = np.load(EMB_DIR / "text.npy")
text_raw = embed_texts(df["title"].fillna("").tolist(), model, processor, device)
text_light = text_clean.copy()
if changed.any():
    text_light[changed.to_numpy()] = embed_texts(df.loc[changed, "light_clean"].tolist(), model, processor, device)

C = json.loads((RESULTS_DIR / "chosen_params.json").read_text())["modes"]["text"]["C"]
y = df[LABEL_COL].to_numpy()
split = df["split"].to_numpy()
scores = {}
for name, X in [("raw_title", text_raw), ("clean_title", text_clean), ("clean_title_light_clean", text_light)]:
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED)
    clf.fit(X[split == "train"], y[split == "train"])
    scores[name] = {s: round(float(f1_score(y[split == s], clf.predict(X[split == s]), average="macro")), 4) for s in ["val", "test"]}
    print(name, scores[name])

result = {
    "clean_title_audit": audit,
    "raw_title_issues": {
        "psbattle_prefix_rows": int(raw_prefix.sum()),
        "psbattle_prefix_by_subreddit": prefix_by_sub,
        "raw_titles_with_digits": raw_digits,
    },
    "light_clean": {"rule": "remove URL fragments (http..., www...), collapse whitespace", "rows_changed": int(changed.sum()),
                    "examples": df.loc[changed, [TEXT_COL, "light_clean"]].head(3).values.tolist()},
    "text_only_macro_f1_by_title_version": scores,
    "text_model_C": C,
}
(RESULTS_DIR / "text_cleaning.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "light_clean"}, indent=2))
print(result["light_clean"])
