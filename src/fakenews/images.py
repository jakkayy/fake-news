import concurrent.futures as cf
import io
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

HEADERS = {"User-Agent": "Mozilla/5.0 (fakeddit course project)"}
MIN_SIDE = 32


def image_path(out_dir, post_id):
    return Path(out_dir) / f"{post_id}.jpg"


def _download_one(post_id, url, out_dir, timeout):
    path = image_path(out_dir, post_id)
    if path.exists():
        return post_id, "ok", "cached"
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code != 200:
            return post_id, "fail", f"http_{r.status_code}"
        img = Image.open(io.BytesIO(r.content))
        img.load()
        if min(img.size) < MIN_SIDE or "removed" in r.url:
            return post_id, "fail", "placeholder"
        img.convert("RGB").save(path, "JPEG", quality=90)
        return post_id, "ok", ""
    except requests.RequestException as e:
        return post_id, "fail", type(e).__name__
    except Exception as e:
        return post_id, "fail", f"bad_image:{type(e).__name__}"


def download_images(df, out_dir, id_col="id", url_col="image_url", workers=8, timeout=10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = [
            ex.submit(_download_one, pid, url, out_dir, timeout)
            for pid, url in zip(df[id_col], df[url_col])
        ]
        for f in tqdm(cf.as_completed(futs), total=len(futs), desc="download"):
            rows.append(f.result())
    return pd.DataFrame(rows, columns=[id_col, "status", "reason"])
