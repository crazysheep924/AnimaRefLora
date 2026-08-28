import torch

from anima_reflora.noise import denoised_target_latent, make_noised_batch, target_frame_mse


def test_noising_keeps_reference_frames_clean():
    clean = torch.randn(2, 16, 3, 8, 8)
    out = make_noised_batch(clean, 0.0, 0.8, 1.0, ref_dropout_prob=0.0)
    assert torch.allclose(out["x"][:, :, 0], clean[:, :, 0])
    assert torch.allclose(out["x"][:, :, 1], clean[:, :, 1])
    assert not torch.allclose(out["x"][:, :, 2], clean[:, :, 2])
    assert torch.all(out["timesteps"][:, :-1] == 0)
    assert torch.all(out["timesteps"][:, -1] >= 0)


def test_noising_target_is_f2_for_reference_layout():
    clean = torch.zeros(1, 1, 3, 2, 2)
    clean[:, :, 0] = 10
    clean[:, :, 1] = 20
    clean[:, :, 2] = 30
    out = make_noised_batch(clean, 1.0, 1.0, 1.0, ref_dropout_prob=0.0)
    assert torch.all(out["x"][:, :, 0] == 10)
    assert torch.all(out["x"][:, :, 1] == 20)
    assert not torch.all(out["x"][:, :, 2] == 30)
    assert torch.all(out["timesteps"] == torch.tensor([[0.0, 0.0, 1.0]]))


def test_loss_uses_only_last_frame():
    pred = torch.zeros(1, 16, 3, 4, 4)
    target = torch.zeros(1, 16, 4, 4)
    pred[:, :, 0] = 100
    pred[:, :, 1] = -100
    loss = target_frame_mse(pred, target)
    assert loss.item() == 0.0
    pred[:, :, 2] = 1
    assert target_frame_mse(pred, target).item() == 1.0


def test_denoised_target_latent_recovers_clean_f2_from_velocity():
    clean = torch.full((1, 1, 3, 2, 2), 30.0)
    noise = torch.full((1, 1, 2, 2), 10.0)
    sigmas = torch.tensor([0.25])
    noised_x = clean.clone()
    noised_x[:, :, -1] = (1.0 - sigmas.view(1, 1, 1, 1)) * clean[:, :, -1] + sigmas.view(1, 1, 1, 1) * noise
    prediction = torch.zeros_like(clean)
    prediction[:, :, -1] = noise - clean[:, :, -1]

    assert torch.allclose(denoised_target_latent(prediction, noised_x, sigmas), clean[:, :, -1])


def test_head_roi_weight_redistributes_loss_with_mean_one_norm():
    target = torch.zeros(1, 1, 2, 2)
    mask = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    sigmas = torch.tensor([0.2])
    # Uniform target-frame error: mean-1 normalization preserves the total loss
    # scale (no secret LR amplification), so the weighted loss equals the plain mean.
    pred = torch.zeros(1, 1, 3, 2, 2)
    pred[:, :, 0] = 100
    pred[:, :, 1] = -100
    pred[:, :, 2] = 1
    loss_uniform = target_frame_mse(pred, target, sigmas=sigmas, head_mask=mask, head_loss_weight=3.0, head_sigma_cutoff=0.6)
    assert abs(loss_uniform.item() - 1.0) < 1e-6
    # Error concentrated inside the ROI gets up-weighted vs the unweighted mean (0.25).
    pred_roi = torch.zeros(1, 1, 3, 2, 2)
    pred_roi[:, :, 2, 0, 0] = 1.0
    loss_roi = target_frame_mse(pred_roi, target, sigmas=sigmas, head_mask=mask, head_loss_weight=3.0, head_sigma_cutoff=0.6)
    assert loss_roi.item() > 0.25
