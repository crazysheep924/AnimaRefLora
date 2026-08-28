import torch

from anima_reflora.features import HeadRoiCache


def test_head_roi_cache_accepts_legacy_tuple_bbox_format(tmp_path):
    cache_path = tmp_path / "head_roi_cache.pt"
    image_path = "/path/to/dataset/images/sample.webp"
    torch.save({(image_path, (8, 10)): (1, 4, 2, 7)}, cache_path)

    cache = HeadRoiCache(cache_path)
    mask = cache.lookup(image_path)

    assert mask is not None
    assert mask.shape == (8, 10)
    assert mask.sum().item() == 15
    assert mask[1:4, 2:7].all()

    batch, valid = cache.gather([image_path], 8, 10, device=torch.device("cpu"), dtype=torch.float32)
    assert valid.tolist() == [True]
    assert batch.shape == (1, 1, 8, 10)
