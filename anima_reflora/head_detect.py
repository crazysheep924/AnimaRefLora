from __future__ import annotations

from pathlib import Path

from PIL import Image

from .head_geom import Box


def detect_head_box(image_path: str | Path, conf: float = 0.3) -> tuple[Box, tuple[int, int]]:
    """Detect the highest-confidence head box without exposing image contents."""

    from imgutils.detect import detect_heads

    path = Path(image_path)
    detections = detect_heads(str(path), conf_threshold=float(conf))
    if not detections:
        raise RuntimeError(f"No head detected: {path}")
    bbox, _label, _score = max(detections, key=lambda item: item[2])
    with Image.open(path) as image:
        size = image.size
    left, top, right, bottom = (float(value) for value in bbox)
    return Box(left, top, right, bottom), size


__all__ = ["detect_head_box"]
