from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

ID_COL = "id"
TEXT_COL = "clean_title"
URL_COL = "image_url"
LABEL_COL = "2_way_label"
KEEP_COLS = [ID_COL, TEXT_COL, URL_COL, LABEL_COL]


def load_raw(name="train", cols=KEEP_COLS):
    return pd.read_csv(RAW_DIR / f"multimodal_{name}.tsv", sep="\t", usecols=cols)
