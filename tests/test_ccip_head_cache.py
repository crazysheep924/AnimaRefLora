import pickle
from pathlib import Path

import torch
from PIL import Image

from anima_reflora.ccip_head_cache import build_head_ccip_cache
from anima_reflora.config import parse_config
from anima_reflora.features import CcipEmbeddingCache


def write_ref_cache(storage: Path, records: list[tuple[Path, str]]) -> None:
    cache_dir = storage / "_latcache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    lat_idx = {}
    meta = {}
    bucket = (8, 8)
    for image_path, character in records:
        path = str(image_path)
        lat_idx[(path, bucket, "full")] = 0
        meta[image_path.stem] = {
            "path": path,
            "character": character,
            "caption": "",
            "ref_eligible": True,
            "has_head": False,
        }
    with (cache_dir / "_cache_index.pkl").open("wb") as fh:
        pickle.dump({"version": 3, "sig": [("shard_000.pt", 0, 0)], "lat_idx": lat_idx, "cap_idx": {}, "meta": meta}, fh)


def make_image(path: Path, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 96), color).save(path)
    return path


def fake_detector(_path: str, conf_threshold: float = 0.4):
    return [((16, 8, 80, 88), "head", max(conf_threshold, 0.9))]


def fake_encoder(images, size: int = 384, model: str = "fake"):
    features = []
    for image in images:
        r, g, b = image.resize((1, 1)).getpixel((0, 0))
        features.append(torch.tensor([r, g, b, size], dtype=torch.float32))
    return features


def config_for(tmp_path: Path):
    storage = tmp_path / "storage"
    return parse_config(["--storage", str(storage), "--out-dir", str(tmp_path / "runs")]), storage


def test_head_ccip_cache_writes_character_prototypes(tmp_path):
    config, storage = config_for(tmp_path)
    image_a = make_image(tmp_path / "images" / "alice_a.png", (255, 0, 0))
    image_b = make_image(tmp_path / "images" / "alice_b.png", (0, 255, 0))
    write_ref_cache(storage, [(image_a, "alice"), (image_b, "alice")])

    out = tmp_path / "head_ccip.pt"
    report = build_head_ccip_cache(config, output=out, detector=fake_detector, encoder=fake_encoder, batch_size=1)

    assert report == {"output": str(out), "records": 2, "built": 2, "failed": 0}
    cache = CcipEmbeddingCache(out)
    embeddings, valid = cache.gather([str(image_a), str(image_b)])
    assert valid.tolist() == [True, True]
    assert torch.allclose(embeddings[0], embeddings[1])


def test_head_ccip_cache_keeps_unknown_characters_per_path(tmp_path):
    config, storage = config_for(tmp_path)
    image_a = make_image(tmp_path / "images" / "unknown_a.png", (255, 0, 0))
    image_b = make_image(tmp_path / "images" / "unknown_b.png", (0, 255, 0))
    write_ref_cache(storage, [(image_a, "unknown"), (image_b, "unknown")])

    out = tmp_path / "head_ccip_unknown.pt"
    build_head_ccip_cache(config, output=out, detector=fake_detector, encoder=fake_encoder)

    cache = CcipEmbeddingCache(out)
    embeddings, valid = cache.gather([str(image_a), str(image_b)])
    assert valid.tolist() == [True, True]
    assert not torch.allclose(embeddings[0], embeddings[1])


def test_head_ccip_cache_supports_path_prototypes(tmp_path):
    config, storage = config_for(tmp_path)
    image_a = make_image(tmp_path / "images" / "alice_path_a.png", (255, 0, 0))
    image_b = make_image(tmp_path / "images" / "alice_path_b.png", (0, 255, 0))
    write_ref_cache(storage, [(image_a, "alice"), (image_b, "alice")])

    out = tmp_path / "head_ccip_path.pt"
    build_head_ccip_cache(config, output=out, prototype_by="path", detector=fake_detector, encoder=fake_encoder)

    cache = CcipEmbeddingCache(out)
    embeddings, valid = cache.gather([str(image_a), str(image_b)])
    assert valid.tolist() == [True, True]
    assert not torch.allclose(embeddings[0], embeddings[1])
