from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Box:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        return max(0.0, self.bottom - self.top)


def expand_box(box: Box, image_size: tuple[int, int], margin: float = 0.6) -> Box:
    width, height = image_size
    cx = (box.left + box.right) * 0.5
    cy = (box.top + box.bottom) * 0.5
    half_w = box.width * (1.0 + margin) * 0.5
    half_h = box.height * (1.0 + margin) * 0.5
    return Box(
        left=max(0.0, cx - half_w),
        top=max(0.0, cy - half_h),
        right=min(float(width), cx + half_w),
        bottom=min(float(height), cy + half_h),
    )


def box_to_latent_mask(
    box: Box,
    image_size: tuple[int, int],
    latent_size: tuple[int, int],
    *,
    margin: float = 0.6,
    min_lat: int = 24,
    max_lat: int = 64,
) -> torch.Tensor:
    expanded = expand_box(box, image_size, margin=margin)
    image_w, image_h = image_size
    lat_h, lat_w = latent_size
    y0 = int(round(expanded.top / max(1, image_h) * lat_h))
    y1 = int(round(expanded.bottom / max(1, image_h) * lat_h))
    x0 = int(round(expanded.left / max(1, image_w) * lat_w))
    x1 = int(round(expanded.right / max(1, image_w) * lat_w))
    side = max(min_lat, min(max(y1 - y0, x1 - x0), max_lat))
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    y0 = max(0, min(lat_h - 1, cy - side // 2))
    x0 = max(0, min(lat_w - 1, cx - side // 2))
    y1 = min(lat_h, max(y0 + 1, y0 + side))
    x1 = min(lat_w, max(x0 + 1, x0 + side))
    mask = torch.zeros(lat_h, lat_w, dtype=torch.float32)
    mask[y0:y1, x0:x1] = 1.0
    return mask


__all__ = ["Box", "box_to_latent_mask", "expand_box"]
