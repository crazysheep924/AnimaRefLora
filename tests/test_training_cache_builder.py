import json

import torch
from PIL import Image

from anima_reflora.build_training_cache import build_training_cache
from anima_reflora.cache import LatentCacheDataset
from anima_reflora.config import parse_config
from anima_reflora.anima_caption import compute_bucket


class FakeVae:
    def encode_pixels_to_latents(self, pixels):
        batch, _channels, height, width = pixels.shape
        fill = pixels.float().mean().detach()
        return torch.ones(batch, 16, 1, height // 8, width // 8, device=pixels.device, dtype=pixels.dtype) * fill


class FakePromptEncoder:
    def encode(self, captions):
        encoded = {}
        for i, caption in enumerate(captions):
            encoded[caption] = {
                "prompt_embeds": torch.full((1, 4, 8), float(i), dtype=torch.bfloat16),
                "attn_mask": torch.ones(1, 4, dtype=torch.int32),
                "t5_input_ids": torch.ones(1, 4, dtype=torch.int32) * (i + 1),
                "t5_attn_mask": torch.ones(1, 4, dtype=torch.int32),
            }
        return encoded


def fake_detector(_path, conf_threshold=0.0):
    return [((8, 8, 56, 56), "head", max(0.9, conf_threshold))]


def test_build_training_cache_from_images_metadata(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    for name, color in [("one.webp", (255, 0, 0)), ("two.webp", (0, 255, 0))]:
        Image.new("RGB", (72, 96), color).save(image_root / name)

    metadata = tmp_path / "records.jsonl"
    rows = [
        {"path": "one.webp", "character": "char-a", "caption": "raw one", "change_caption": "change one", "ref_eligible": True},
        {"path": "two.webp", "character": "char-a", "caption": "raw two", "change_caption": "change two", "ref_eligible": True},
    ]
    metadata.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    cache_dir = tmp_path / "latcache"
    config = parse_config(["--storage", str(tmp_path / "storage"), "--backend", "tiny", "--dtype", "fp32"])
    report = build_training_cache(
        config,
        image_root=image_root,
        metadata=metadata,
        output_cache=cache_dir,
        shard_size=1,
        vae=FakeVae(),
        prompt_encoder=FakePromptEncoder(),
        detector=fake_detector,
    )

    assert report["built_full"] == 2
    assert report["built_head"] == 2
    assert report["captions"] == 4
    assert (cache_dir / "_cache_index.pkl").exists()
    assert (cache_dir / "shard_000.pt").exists()
    assert (cache_dir / "shard_001.pt").exists()

    dataset = LatentCacheDataset(cache_dir, frames=3)
    item = dataset[1]
    expected_bucket = compute_bucket(72, 96)[2]
    assert item["bucket"] == expected_bucket
    assert item["frames"].shape == (16, 3, *expected_bucket)
    assert item["caption_key"] == "change two"
    assert item["caption_source"] == "change_caption"
    assert item["prompt_embeds"].shape == (4, 8)
    assert item["head_available"] is True
