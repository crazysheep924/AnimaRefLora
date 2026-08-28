import json
from pathlib import Path

import pytest
import torch
from torch import nn

from anima_reflora.bundle import (
    BUNDLE_FORMAT,
    bundle_features,
    bundle_rope_payload,
    bundle_sidecar_names,
    is_bundle,
    load_bundle_group,
    pack_bundle,
    read_safetensors_metadata,
    verify_bundle,
)
from anima_reflora.checkpoints import load_checkpoint_into, load_sidecar_into
from anima_reflora.local_ref_ab_infer import load_features
from anima_reflora.rope_refpos import load_refpos_for_checkpoint

safetensors = pytest.importorskip("safetensors.torch")

STEP = 12345
FEATURES = {"frames": 3, "rope_refpos": True, "rope_layout": "disjoint", "rope_shift": 1.0, "cpm": True}
ROPE = {
    "format": "anima_reflora_rope_refpos_v1",
    "name": "disjoint",
    "ref_frame_idx": [0, 1],
    "spatial_shift": 1.0,
    "temporal": [0, 1, 2],
    "per_frame_offsets": None,
    "frames": 3,
}


def make_module(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))


@pytest.fixture()
def ckpt_dir(tmp_path: Path) -> Path:
    d = tmp_path / "checkpoints"
    d.mkdir()
    lora = make_module(1)
    ref = make_module(2)
    crepa = make_module(3)
    safetensors.save_file(lora.state_dict(), str(d / f"lora_step_{STEP}.safetensors"), metadata={"format": "pt", "anima_reflora_step": str(STEP)})
    safetensors.save_file(ref.state_dict(), str(d / f"ref_conditioner_step_{STEP}.safetensors"))
    safetensors.save_file(crepa.state_dict(), str(d / f"crepa_projector_step_{STEP}.safetensors"))
    torch.save({"optimizer": {}}, d / f"optimizer_step_{STEP}.pt")
    (d / f"feature_config_step_{STEP}.json").write_text(json.dumps(FEATURES), encoding="utf-8")
    (d / f"rope_refpos_step_{STEP}.json").write_text(json.dumps(ROPE), encoding="utf-8")
    return d


@pytest.fixture()
def bundle(ckpt_dir: Path, tmp_path: Path) -> Path:
    return pack_bundle(ckpt_dir, STEP, tmp_path / "test.animaref.safetensors", name="test-bundle")


def test_pack_and_verify_bit_exact(ckpt_dir, bundle):
    verify_bundle(ckpt_dir, STEP, bundle)


def test_bundle_detection_and_metadata(bundle, ckpt_dir):
    assert is_bundle(bundle)
    assert not is_bundle(ckpt_dir / f"lora_step_{STEP}.safetensors")
    meta = read_safetensors_metadata(bundle)
    assert meta["format"] == BUNDLE_FORMAT
    assert meta["step"] == str(STEP)
    assert meta["name"] == "test-bundle"
    # original lora metadata carried under orig. prefix
    assert meta["orig.anima_reflora_step"] == str(STEP)
    assert sorted(bundle_sidecar_names(bundle)) == ["crepa_projector", "ref_conditioner"]


def test_group_loading_matches_sources(bundle, ckpt_dir):
    for group, fname in [
        ("lora", f"lora_step_{STEP}.safetensors"),
        ("ref_conditioner", f"ref_conditioner_step_{STEP}.safetensors"),
        ("crepa_projector", f"crepa_projector_step_{STEP}.safetensors"),
    ]:
        expected = safetensors.load_file(str(ckpt_dir / fname))
        got = load_bundle_group(bundle, group)
        assert set(expected) == set(got)
        for key in expected:
            assert torch.equal(expected[key], got[key])


def test_features_from_bundle(bundle):
    assert bundle_features(bundle) == FEATURES
    assert load_features(bundle) == FEATURES


def test_rope_from_bundle(bundle):
    assert bundle_rope_payload(bundle) == ROPE
    state = load_refpos_for_checkpoint(bundle)
    assert state is not None
    assert state.frames == 3
    assert state.scheme.name == "disjoint"
    assert list(state.scheme.ref_frame_idx) == [0, 1]


def test_rope_legacy_path_still_works(ckpt_dir):
    state = load_refpos_for_checkpoint(ckpt_dir / f"lora_step_{STEP}.safetensors")
    assert state is not None
    assert state.scheme.name == "disjoint"


def test_load_sidecar_into_bundle_and_legacy(bundle, ckpt_dir):
    for source in (bundle, ckpt_dir / f"lora_step_{STEP}.safetensors"):
        module = make_module(99)
        missing, unexpected = load_sidecar_into(module, source, "ref_conditioner", strict=True)
        assert missing == [] and unexpected == []
        expected = safetensors.load_file(str(ckpt_dir / f"ref_conditioner_step_{STEP}.safetensors"))
        for key, value in module.state_dict().items():
            assert torch.equal(value, expected[key])


def test_load_sidecar_missing_group_raises(bundle):
    module = make_module(7)
    with pytest.raises(FileNotFoundError):
        load_sidecar_into(module, bundle, "no_such_group")


def test_load_checkpoint_into_bundle(bundle, ckpt_dir):
    module = make_module(42)
    missing, unexpected = load_checkpoint_into(module, bundle, strict=True)
    assert missing == [] and unexpected == []
    expected = safetensors.load_file(str(ckpt_dir / f"lora_step_{STEP}.safetensors"))
    for key, value in module.state_dict().items():
        assert torch.equal(value, expected[key])


def test_optimizer_excluded(bundle):
    from safetensors import safe_open

    with safe_open(str(bundle), framework="pt", device="cpu") as fh:
        assert not any(k.startswith("optimizer.") for k in fh.keys())
