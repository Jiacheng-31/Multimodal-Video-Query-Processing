from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from .ann import l2_normalize


class CLIPBackbone:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cuda",
        dtype: str = "float16",
    ) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("Install the 'transformers' package to use CLIP retrieval encoders.") from exc

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        torch_dtype = torch.float16 if dtype in {"float16", "fp16"} and self.device.type == "cuda" else torch.float32
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)
        self.model = CLIPModel.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
        self.model.to(self.device)
        self.model.eval()

    def encode_text(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        import torch

        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                inputs = self.processor(text=batch, padding=True, truncation=True, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                feats = self.model.get_text_features(**inputs)
                feats = torch.nn.functional.normalize(feats.float(), dim=-1)
                outputs.append(feats.cpu().numpy().astype("float32"))
        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, self.hidden_size), dtype="float32")

    def encode_images(self, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        import torch

        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch = [image.convert("RGB") for image in images[start : start + batch_size]]
                inputs = self.processor(images=batch, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                feats = self.model.get_image_features(**inputs)
                feats = torch.nn.functional.normalize(feats.float(), dim=-1)
                outputs.append(feats.cpu().numpy().astype("float32"))
        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, self.hidden_size), dtype="float32")

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.projection_dim)


def open_image(path: Path) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def crop_image(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image | None:
    width, height = image.size
    x1, y1, x2, y2 = box
    left = max(0, min(width, int(round(x1))))
    top = max(0, min(height, int(round(y1))))
    right = max(left + 1, min(width, int(round(x2))))
    bottom = max(top + 1, min(height, int(round(y2))))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom)).convert("RGB")


def mean_pool_window(frame_features: np.ndarray, indices: Iterable[int]) -> np.ndarray:
    idx = list(indices)
    if not idx:
        raise ValueError("empty action window")
    pooled = frame_features[idx].mean(axis=0, keepdims=True)
    return l2_normalize(pooled)[0]
