import torch

from anima_reflora.noise import latent_reconstruction_l1, ref_diff_weight_map, target_frame_mse


def make_clean(diff_region_value: float = 3.0) -> torch.Tensor:
    # B=1, C=4, F=3 (head_ref, full_ref, target), H=W=8
    clean = torch.zeros(1, 4, 3, 8, 8)
    clean[:, :, 1] = 1.0  # full ref
    clean[:, :, 2] = 1.0  # target identical to ref...
    clean[:, :, 2, :, :4] = diff_region_value  # ...except the left half
    return clean


def test_ref_diff_weight_map_disabled():
    clean = make_clean()
    assert ref_diff_weight_map(clean, 0.0) is None
    single = torch.zeros(1, 4, 1, 8, 8)
    assert ref_diff_weight_map(single, 0.5) is None


def test_ref_diff_weight_map_upweights_changed_region():
    clean = make_clean()
    weight = ref_diff_weight_map(clean, 1.0, diff_weight_min=0.2)
    assert weight.shape == (1, 1, 8, 8)
    changed = weight[0, 0, :, :4].mean()
    unchanged = weight[0, 0, :, 4:].mean()
    assert changed > unchanged
    # identical region has zero diff -> hits the clamp floor
    assert torch.allclose(weight[0, 0, :, 4:], torch.full((8, 4), 0.2))


def test_ref_diff_weight_map_floor_binds_at_half_lambda():
    # Regression: the old 1 + lambda*(diff-1) form bottomed out at 1-lambda,
    # so diff_weight_min=0.2 was silently inert for lambda <= 0.8 (the
    # production recipe runs lambda=0.5).
    clean = make_clean()
    weight = ref_diff_weight_map(clean, 0.5, diff_weight_min=0.2)
    assert torch.allclose(weight[0, 0, :, 4:], torch.full((8, 4), 0.2))
    # up-weight side keeps the lambda slope: normalized diff 2 -> 1 + 0.5*(2-1)
    assert torch.allclose(weight[0, 0, :, :4], torch.full((8, 4), 1.5))


def test_ref_diff_weight_map_uniform_when_everything_differs():
    clean = make_clean()
    clean[:, :, 2] = 5.0  # target differs from ref everywhere by the same amount
    weight = ref_diff_weight_map(clean, 1.0)
    assert torch.allclose(weight, torch.ones_like(weight))


def test_ref_diff_weight_map_head_roi_never_below_one():
    clean = make_clean()
    head_mask = torch.zeros(1, 1, 8, 8)
    head_mask[..., :4, 4:] = 1.0  # head ROI inside the target==ref region
    weight = ref_diff_weight_map(clean, 1.0, diff_weight_min=0.2, head_mask=head_mask)
    # inside head ROI the identical region is exempt from down-weighting
    assert torch.all(weight[0, 0, :4, 4:] >= 1.0)
    # outside head ROI the identical region still hits the floor
    assert torch.allclose(weight[0, 0, 4:, 4:], torch.full((4, 4), 0.2))


def test_ref_diff_weight_map_caps_spiky_weights():
    clean = torch.zeros(1, 4, 3, 8, 8)
    clean[:, :, 1] = 1.0
    clean[:, :, 2] = 1.0
    clean[:, :, 2, 0, 0] = 100.0  # one extreme pixel dominates the diff
    weight = ref_diff_weight_map(clean, 1.0)
    assert float(weight.max()) <= 5.0


def test_target_frame_mse_extra_weight_map_redistributes_without_rescaling():
    torch.manual_seed(0)
    prediction = torch.randn(1, 4, 3, 8, 8)
    target = torch.zeros(1, 4, 8, 8)
    base = target_frame_mse(prediction, target)
    uniform = torch.ones(1, 1, 8, 8)
    weighted_uniform = target_frame_mse(prediction, target, extra_weight_map=uniform)
    assert torch.allclose(base, weighted_uniform)

    # zero weight on the right half -> only left-half errors count (after mean-normalization)
    half = torch.ones(1, 1, 8, 8)
    half[..., 4:] = 0.0
    weighted_half = target_frame_mse(prediction, target, extra_weight_map=half)
    left_mse = prediction[:, :, -1, :, :4].float().pow(2).mean()
    assert torch.allclose(weighted_half, left_mse, atol=1e-6)


def test_latent_reconstruction_l1_extra_weight_map():
    torch.manual_seed(0)
    prediction = torch.randn(1, 4, 3, 8, 8)
    noised_x = torch.randn(1, 4, 3, 8, 8)
    clean = torch.randn(1, 4, 3, 8, 8)
    sigmas = torch.tensor([0.5])
    base = latent_reconstruction_l1(prediction, noised_x, clean, sigmas)
    uniform = torch.ones(1, 1, 8, 8)
    weighted_uniform = latent_reconstruction_l1(prediction, noised_x, clean, sigmas, extra_weight_map=uniform)
    assert torch.allclose(base, weighted_uniform)

    # zero weight on the right half -> only left-half errors count (after mean-normalization)
    half = torch.ones(1, 1, 8, 8)
    half[..., 4:] = 0.0
    weighted_half = latent_reconstruction_l1(prediction, noised_x, clean, sigmas, extra_weight_map=half)
    pred_clean = noised_x[:, :, -1].float() - sigmas.view(-1, 1, 1, 1) * prediction[:, :, -1].float()
    left_l1 = (pred_clean - clean[:, :, -1].float()).abs()[..., :4].mean()
    assert torch.allclose(weighted_half, left_l1, atol=1e-6)


def test_target_frame_mse_head_and_diff_weights_compose():
    torch.manual_seed(0)
    prediction = torch.randn(2, 4, 3, 8, 8)
    target = torch.randn(2, 4, 8, 8)
    sigmas = torch.tensor([0.1, 0.3])
    head_mask = torch.zeros(2, 1, 8, 8)
    head_mask[..., :3, :3] = 1.0
    extra = torch.ones(2, 1, 8, 8)
    head_only = target_frame_mse(
        prediction, target, sigmas=sigmas, head_mask=head_mask, head_loss_weight=4.0
    )
    composed = target_frame_mse(
        prediction, target, sigmas=sigmas, head_mask=head_mask, head_loss_weight=4.0, extra_weight_map=extra
    )
    assert torch.allclose(head_only, composed)


def test_target_frame_mse_head_and_nonuniform_diff_hand_computed():
    # Real composition check with hand-derived expectation: head weight 4
    # (sigma 0 -> full decay -> w_eff=4) times a non-uniform extra map,
    # renormalized to mean 1 exactly once. A double-renorm (or renorm of only
    # one factor) produces a different value and fails this test.
    prediction = torch.zeros(1, 1, 2, 2, 2)
    prediction[0, 0, -1] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # sq err 1,4,9,16
    target = torch.zeros(1, 1, 2, 2)
    sigmas = torch.tensor([0.0])
    head_mask = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    extra = torch.tensor([[[[1.0, 1.0], [2.0, 2.0]]]])
    # combined = head(4,1,1,1) * extra = (4,1,2,2), mean 2.25
    # loss = mean([1,4,9,16] * [4,1,2,2] / 2.25) = (4+4+18+32)/4/2.25 = 58/9
    loss = target_frame_mse(
        prediction, target, sigmas=sigmas, head_mask=head_mask, head_loss_weight=4.0, extra_weight_map=extra
    )
    assert torch.allclose(loss, torch.tensor(58.0 / 9.0), atol=1e-6)


def test_ref_diff_weight_map_head_roi_at_half_lambda():
    # Production path: lambda=0.5, floor=0.2, head ROI overlapping the matched
    # region. Inside ROI matched pixels are exempt (1.0), outside they floor
    # at 0.2, changed pixels keep the lambda up-slope (1.5).
    clean = make_clean()
    head_mask = torch.zeros(1, 1, 8, 8)
    head_mask[..., :4, 4:] = 1.0
    weight = ref_diff_weight_map(clean, 0.5, diff_weight_min=0.2, head_mask=head_mask)
    assert torch.allclose(weight[0, 0, :4, 4:], torch.ones(4, 4))
    assert torch.allclose(weight[0, 0, 4:, 4:], torch.full((4, 4), 0.2))
    assert torch.allclose(weight[0, 0, :, :4], torch.full((8, 4), 1.5))


def test_ref_diff_weight_map_per_sample_normalization_independence():
    # Two samples with the same spatial diff pattern at 100x different raw
    # magnitude must produce identical weight maps: normalization is
    # per-sample. A regression to a batch-global mean would push sample 0's
    # normalized diff toward 0 (down branch) and sample 1's toward the cap.
    clean = torch.zeros(2, 4, 3, 8, 8)
    clean[:, :, 1] = 1.0
    clean[0, :, 2] = 1.0
    clean[0, :, 2, :, :4] = 3.0  # raw diff 2 on the left half
    clean[1, :, 2] = 1.0
    clean[1, :, 2, :, :4] = 201.0  # raw diff 200, same pattern
    weight = ref_diff_weight_map(clean, 0.5, diff_weight_min=0.2)
    assert torch.allclose(weight[0], weight[1])
    assert torch.allclose(weight[0, 0, :, :4], torch.full((8, 4), 1.5))
    assert torch.allclose(weight[0, 0, :, 4:], torch.full((8, 4), 0.2))
