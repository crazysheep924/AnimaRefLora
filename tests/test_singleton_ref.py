import torch

from anima_reflora.cache import collate_latent_batch
from anima_reflora.noise import apply_reference_dropout, make_noised_batch


def _clean(batch=3, frames=3):
    return torch.randn(batch, 4, frames, 8, 8)


def test_force_blank_zeroes_all_ref_frames_and_marks_dropped():
    clean = _clean()
    force = torch.tensor([True, False, True])
    out, stats = apply_reference_dropout(
        clean, prob=0.0, mode="blank", t3_mode="structured",
        return_stats=True, force_blank=force,
    )
    dropped = stats["dropped_ref_frames"]
    for b in (0, 2):
        assert torch.all(out[b, :, 0] == 0)
        assert torch.all(out[b, :, 1] == 0)
        assert bool(dropped[b, 0]) and bool(dropped[b, 1])
        assert torch.equal(out[b, :, 2], clean[b, :, 2])  # target untouched
    assert torch.equal(out[1], clean[1])
    assert not bool(dropped[1].any())


def test_force_blank_applies_even_with_prob_zero_and_mode_none():
    clean = _clean(batch=2)
    force = torch.tensor([True, False])
    out, stats = apply_reference_dropout(
        clean, prob=0.5, mode="none", t3_mode="structured",
        generator=torch.Generator().manual_seed(0),
        return_stats=True, force_blank=force,
    )
    assert torch.all(out[0, :, 0] == 0) and torch.all(out[0, :, 1] == 0)
    assert bool(stats["dropped_ref_frames"][0, 0]) and bool(stats["dropped_ref_frames"][0, 1])


def test_no_force_no_prob_is_identity():
    clean = _clean(batch=2)
    out, stats = apply_reference_dropout(
        clean, prob=0.0, mode="blank", t3_mode="structured",
        return_stats=True, force_blank=torch.tensor([False, False]),
    )
    assert torch.equal(out, clean)
    assert not bool(stats["dropped_ref_frames"].any())


def test_make_noised_batch_threads_force_flag_to_diff_weight_gating():
    clean = _clean(batch=2)
    force = torch.tensor([True, False])
    noised = make_noised_batch(
        clean, high_sigma_mix_prob=0.0, high_sigma_min=0.9, high_sigma_max=1.0,
        ref_dropout_prob=0.0, generator=torch.Generator().manual_seed(1),
        force_ref_blank=force,
    )
    # full-ref frame (index frames-2) must be flagged dropped for sample 0 —
    # this is the key train.py's diff-weight gating reads.
    full_idx = clean.shape[2] - 2
    assert bool(noised["dropped_ref_frames"][0, full_idx])
    assert not bool(noised["dropped_ref_frames"][1, full_idx])


def test_collate_carries_force_ref_blank_with_default():
    def item(flag=None):
        d = {
            "frames": torch.zeros(4, 3, 8, 8), "prompt_embeds": torch.zeros(2, 4),
            "attn_mask": torch.ones(2), "t5_input_ids": torch.ones(2),
            "t5_attn_mask": torch.ones(2), "target_path": "t", "ref_path": "r",
            "caption_key": "c", "caption_source": "s", "bucket": (8, 8),
            "character": "x", "head_available": True,
            "frame_roles": ("head_ref", "full_ref", "target"),
        }
        if flag is not None:
            d["force_ref_blank"] = flag
        return d

    batch = collate_latent_batch([item(True), item(False), item(None)])
    assert batch["force_ref_blank"].tolist() == [True, False, False]
