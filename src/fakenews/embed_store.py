from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .embed import embed_images, embed_texts
from .images import image_path


def extract_to_dir(df, img_dir, out_dir, model, processor, device, id_col="id", text_col="clean_title",
                   label_col="2_way_label", chunk_size=2000):
    out_dir = Path(out_dir)
    parts = out_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(df) + chunk_size - 1) // chunk_size
    for c in tqdm(range(n_chunks), desc="embed chunks"):
        part = parts / f"{c:04d}.npz"
        if part.exists():
            continue
        chunk = df.iloc[c * chunk_size:(c + 1) * chunk_size]
        text = embed_texts(chunk[text_col].tolist(), model, processor, device)
        image = embed_images([image_path(img_dir, i) for i in chunk[id_col]], model, processor, device)
        np.savez(part.with_suffix(".tmp.npz"), text=text, image=image, ids=chunk[id_col].to_numpy(dtype=str))
        part.with_suffix(".tmp.npz").replace(part)

    loaded = [np.load(parts / f"{c:04d}.npz") for c in range(n_chunks)]
    ids = np.concatenate([p["ids"] for p in loaded])
    if not (ids == df[id_col].to_numpy(dtype=str)).all():
        raise ValueError("embedding parts do not match the sample order; delete parts/ and rerun")
    np.save(out_dir / "text.npy", np.concatenate([p["text"] for p in loaded]))
    np.save(out_dir / "image.npy", np.concatenate([p["image"] for p in loaded]))
    df[[id_col]].to_csv(out_dir / "ids.csv", index=False)
    np.save(out_dir / "labels.npy", df[label_col].to_numpy())


def load_embeddings(emb_dir, split_file, split):
    emb_dir = Path(emb_dir)
    ids = pd.read_csv(emb_dir / "ids.csv")["id"]
    splits = pd.read_csv(split_file).set_index("id")["split"]
    mask = (ids.map(splits) == split).to_numpy()
    return (np.load(emb_dir / "text.npy")[mask], np.load(emb_dir / "image.npy")[mask],
            np.load(emb_dir / "labels.npy")[mask], ids[mask].to_numpy())
