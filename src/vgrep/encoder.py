"""SigLIP 2 wrapper. Loaded lazily so `vgrep --help` stays instant."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .config import SETTINGS

# iPhone photos are HEIC and PIL cannot open them unaided.
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover
    pass


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Encoder:
    """Encodes images and text into the same vector space.

    Vectors are L2-normalised on the way out, so inner product == cosine similarity
    and FAISS IndexFlatIP gives us cosine ranking for free.
    """

    def __init__(self, model_name: str = SETTINGS.model, device: str | None = None):
        self.model_name = model_name
        self.device = device or pick_device()
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(
            self.model_name,
            dtype=torch.float16 if self.device != "cpu" else torch.float32,
        ).to(self.device).eval()

    @staticmethod
    def _as_tensor(out):
        """transformers 5 returns an output object here; older versions a raw tensor."""
        if hasattr(out, "float"):
            return out
        for attr in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
            v = getattr(out, attr, None)
            if v is not None:
                return v.mean(dim=1) if v.ndim == 3 else v
        raise TypeError(f"Unexpected output type: {type(out)}")

    @staticmethod
    def _normalise(v: "np.ndarray") -> np.ndarray:
        return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        self._load()
        inputs = self._processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        return self._normalise(self._as_tensor(feats).float().cpu().numpy())

    def encode_text(self, texts: list[str]) -> np.ndarray:
        import torch

        self._load()
        inputs = self._processor(
            text=texts, padding="max_length", truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        return self._normalise(self._as_tensor(feats).float().cpu().numpy())


def load_image(path: str) -> Image.Image | None:
    """Decode and downscale. Runs in worker threads; returns None on unreadable files."""
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        # Shrink before handing to the processor -- full-res decode is the main cost.
        img.thumbnail((512, 512), Image.Resampling.BILINEAR)
        return img
    except Exception:
        return None
