from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image

from .embed import load_clip
from .features import MODES, build_features

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
CLASSIFIERS = ["logreg", "mlp"]


class Predictor:
    def __init__(self, model_dir=MODEL_DIR):
        self.model, self.processor, self.device = load_clip()
        self.classifiers, self.thresholds = {}, {}
        for kind in CLASSIFIERS:
            paths = {m: Path(model_dir) / f"{kind}_{m.replace('+', '_')}.joblib" for m in MODES}
            if not all(path.exists() for path in paths.values()):
                continue
            saved = {m: joblib.load(path) for m, path in paths.items()}
            self.classifiers[kind] = {m: s["model"] if isinstance(s, dict) else s for m, s in saved.items()}
            self.thresholds[kind] = {m: s["threshold"] if isinstance(s, dict) else 0.5 for m, s in saved.items()}

    @torch.no_grad()
    def _embed_text(self, title):
        inputs = self.processor(text=[title], padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
        return self._unit(self.model.get_text_features(**inputs))

    @torch.no_grad()
    def _embed_image(self, image):
        inputs = self.processor(images=[image.convert("RGB")], return_tensors="pt").to(self.device)
        return self._unit(self.model.get_image_features(**inputs))

    @staticmethod
    def _unit(x):
        if not isinstance(x, torch.Tensor):
            x = x.pooler_output
        return (x / x.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)

    def predict(self, title=None, image=None, classifier="logreg"):
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        text = self._embed_text(title) if title and title.strip() else None
        img = self._embed_image(image) if image is not None else None
        probs = {}
        for mode in MODES:
            if (mode != "image" and text is None) or (mode != "text" and img is None):
                continue
            X = build_features(mode, text, img)
            probs[mode] = float(self.classifiers[classifier][mode].predict_proba(X)[0, 1])
        return probs
