from __future__ import annotations

import torch
import torch.nn.functional as F


# sigma stays away from 0 to avoid blow-up in sigma**-2 / min_snr weighting (A7).
SIGMA_CLAMP = (1e-3, 1.0)


def sample_sigmas(
    batch: int,
    high_sigma_mix_prob: float,
    high_sigma_min: float,
    high_sigma_max: float,
    device: torch.device,
    generator: torch.Generator | None = None,
    timestep_sampling: str = "sigmoid",
    sigmoid_scale: float = 1.0,
) -> torch.Tensor:
    # Matches the original noise_collate sampler: "sigmoid" gives a logit-normal
    # distribution (noise concentrated mid-range), "uniform" is sigma ~ U(0,1).
    if timestep_sampling == "sigmoid":
        base = torch.sigmoid(sigmoid_scale * torch.randn(batch, device=device, generator=generator))
    elif timestep_sampling == "uniform":
        base = torch.rand(batch, device=device, generator=generator)
    else:
        raise ValueError(f"unsupported timestep_sampling={timestep_sampling!r} (use 'sigmoid' or 'uniform')")
    if high_sigma_mix_prob > 0:
        choose_high = torch.rand(batch, device=device, generator=generator) < high_sigma_mix_prob
        high = torch.empty(batch, device=device).uniform_(high_sigma_min, high_sigma_max, generator=generator)
        base = torch.where(choose_high, high, base)
    return base.clamp(*SIGMA_CLAMP)


def apply_reference_dropout(
    clean: torch.Tensor,
    prob: float,
    mode: str,
    t3_mode: str,
    generator: torch.Generator | None = None,
    return_stats: bool = False,
    force_blank: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """force_blank: per-sample bool — blank ALL ref frames of those samples
    unconditionally (singleton-cell samples have no real ref; zeroing routes
    them into the unconditional branch and every downstream gate keyed on
    dropped_ref_frames stays consistent). Applies even when prob == 0 or
    mode == 'none'."""
    dropped = torch.zeros(clean.shape[0], clean.shape[2], device=clean.device, dtype=torch.bool)
    batch, _, frames, _, _ = clean.shape
    ref_frames = list(range(max(frames - 1, 0)))

    out = clean
    if force_blank is not None and bool(force_blank.any()):
        out = out.clone()
        forced = force_blank.to(device=out.device).bool()
        for frame in ref_frames:
            out[forced, :, frame] = 0.0
            dropped[forced, frame] = True

    if prob <= 0:
        return (out, {"dropped_ref_frames": dropped}) if return_stats else out
    if mode not in {"blank", "noise", "none"}:
        raise ValueError(f"Unsupported ref dropout mode: {mode}")
    if out is clean:
        out = clean.clone()
    mask = torch.rand(batch, device=out.device, generator=generator) < prob
    if not mask.any():
        return (out, {"dropped_ref_frames": dropped}) if return_stats else out
    if mode == "none":
        return (out, {"dropped_ref_frames": dropped}) if return_stats else out

    def replacement(frame: torch.Tensor) -> torch.Tensor:
        if mode == "blank":
            return torch.zeros_like(frame)
        return torch.randn(frame.shape, device=frame.device, dtype=frame.dtype, generator=generator)

    if frames == 3 and t3_mode == "structured":
        for b in torch.where(mask)[0].tolist():
            chosen = int(torch.randint(0, 3, (), device=out.device, generator=generator).item())
            drop_frames = (0,) if chosen == 0 else (1,) if chosen == 1 else (0, 1)
            for frame in drop_frames:
                out[b, :, frame] = replacement(out[b, :, frame])
                dropped[b, frame] = True
    else:
        for frame in ref_frames:
            out[mask, :, frame] = replacement(out[mask, :, frame])
            dropped[mask, frame] = True
    return (out, {"dropped_ref_frames": dropped}) if return_stats else out


def make_noised_batch(
    clean: torch.Tensor,
    high_sigma_mix_prob: float,
    high_sigma_min: float,
    high_sigma_max: float,
    ref_dropout_prob: float = 0.0,
    ref_dropout_mode: str = "blank",
    ref_dropout_t3_mode: str = "structured",
    generator: torch.Generator | None = None,
    timestep_sampling: str = "sigmoid",
    sigmoid_scale: float = 1.0,
    force_ref_blank: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    clean, dropout = apply_reference_dropout(
        clean,
        ref_dropout_prob,
        ref_dropout_mode,
        ref_dropout_t3_mode,
        generator=generator,
        return_stats=True,
        force_blank=force_ref_blank,
    )
    batch, channels, frames, height, width = clean.shape
    sigmas = sample_sigmas(
        batch,
        high_sigma_mix_prob,
        high_sigma_min,
        high_sigma_max,
        clean.device,
        generator,
        timestep_sampling=timestep_sampling,
        sigmoid_scale=sigmoid_scale,
    )
    noise = torch.randn(clean[:, :, -1].shape, device=clean.device, dtype=clean.dtype, generator=generator)
    x = clean.clone()
    sigma_view = sigmas.view(batch, 1, 1, 1)
    x[:, :, -1] = (1.0 - sigma_view) * clean[:, :, -1] + sigma_view * noise
    timesteps = torch.zeros(batch, frames, device=clean.device, dtype=torch.float32)
    timesteps[:, -1] = sigmas
    target = noise - clean[:, :, -1]
    return {"x": x, "target": target, "noise": noise, "sigmas": sigmas, "timesteps": timesteps, "clean": clean, **dropout}


def target_frame_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sigmas: torch.Tensor | None = None,
    weighting_scheme: str = "none",
    min_snr_gamma: float = 5.0,
    head_mask: torch.Tensor | None = None,
    head_loss_weight: float = 1.0,
    head_sigma_cutoff: float = 0.6,
    extra_weight_map: torch.Tensor | None = None,
) -> torch.Tensor:
    pred_target = prediction[:, :, -1].float()
    target = target.float()
    loss_map = (pred_target - target).pow(2)
    if weighting_scheme == "min_snr" and sigmas is not None:
        snr = ((1.0 - sigmas).clamp_min(1e-4) / sigmas.clamp_min(1e-4)).pow(2)
        weights = torch.minimum(snr, torch.full_like(snr, min_snr_gamma)) / snr.clamp_min(1e-4)
        loss_map = loss_map * weights.view(-1, 1, 1, 1)
    weight_map = None
    if head_mask is not None and sigmas is not None and head_loss_weight != 1.0:
        if head_mask.shape[-2:] != loss_map.shape[-2:]:
            head_mask = F.interpolate(head_mask.float(), size=loss_map.shape[-2:], mode="nearest")
        head_mask = head_mask.float()
        # Linear sigma fade: weighting fades to 1.0 as sigma -> cutoff (the face is
        # unrecoverable under heavy noise).
        sig = sigmas.float().view(-1, 1, 1, 1)
        if head_sigma_cutoff > 0:
            decay = (1.0 - sig / head_sigma_cutoff).clamp_min(0.0)
        else:
            decay = torch.ones_like(sig)
        w_eff = 1.0 + (head_loss_weight - 1.0) * decay
        weight_map = 1.0 + (w_eff - 1.0) * head_mask
    if extra_weight_map is not None:
        extra = extra_weight_map.float()
        if extra.shape[-2:] != loss_map.shape[-2:]:
            extra = F.interpolate(extra, size=loss_map.shape[-2:], mode="nearest")
        weight_map = extra if weight_map is None else weight_map * extra
    if weight_map is not None:
        # Renormalize each sample's combined weight map so mean()==1, which
        # redistributes the loss WITHOUT inflating the total loss scale
        # (i.e. without secretly amplifying the effective LR).
        mean = weight_map.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        loss_map = loss_map * (weight_map / mean)
    return loss_map.mean()


def ref_diff_weight_map(
    clean: torch.Tensor,
    diff_loss_lambda: float,
    diff_weight_min: float = 0.2,
    head_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Spatial MSE weight from |target - full_ref| in latent space.

    Regions where the target genuinely matches the reference get down-weighted
    (copying there is correct), regions that differ get up-weighted (that is
    where caption->change binding lives). Per-sample normalization makes
    unrelated-pose pairs degrade to ~uniform weighting.

    Two independent knobs: below diff=1 the weight interpolates linearly from
    diff_weight_min (at diff=0) up to 1, so the floor is actually reached on
    fully-matched regions regardless of lambda; above diff=1 lambda alone sets
    the up-weight slope. (The old single-slope form 1 + lambda*(diff-1) never
    reached the floor for lambda <= 0.8.)

    Computed from pre-dropout clean latents; callers gate samples whose full-ref
    input frame was blanked back to uniform 1.0 (the model cannot see that ref,
    so a weight derived from it would bias the unconditional branch).

    Inside the head ROI the weight never drops below 1.0: the head SHOULD match
    the reference (identity), so diff down-weighting there would cancel the
    head-loss boost instead of discouraging body copying.
    """
    if diff_loss_lambda <= 0 or clean.shape[2] < 2:
        return None
    full_idx = max(clean.shape[2] - 2, 0)
    diff = (clean[:, :, -1].float() - clean[:, :, full_idx].float()).abs().mean(dim=1, keepdim=True)
    diff = diff / diff.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    # Cap the normalized diff so tiny residuals on near-identical pairs cannot
    # blow up into spiky weights after per-sample normalization.
    diff = diff.clamp_max(5.0)
    floor = float(diff_weight_min)
    down = floor + (1.0 - floor) * diff
    up = 1.0 + float(diff_loss_lambda) * (diff - 1.0)
    weight = torch.where(diff < 1.0, down, up).clamp_min(floor)
    if head_mask is not None:
        mask = head_mask.float()
        if mask.shape[-2:] != weight.shape[-2:]:
            mask = F.interpolate(mask, size=weight.shape[-2:], mode="nearest")
        weight = torch.where(mask > 0, weight.clamp_min(1.0), weight)
    return weight


def denoised_target_latent(prediction: torch.Tensor, noised_x: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    pred_target_v = prediction[:, :, -1].float()
    noised_target = noised_x[:, :, -1].float()
    sigma = sigmas.float().view(-1, 1, 1, 1)
    return noised_target - sigma * pred_target_v


def latent_reconstruction_l1(
    prediction: torch.Tensor,
    noised_x: torch.Tensor,
    clean: torch.Tensor,
    sigmas: torch.Tensor,
    extra_weight_map: torch.Tensor | None = None,
) -> torch.Tensor:
    pred_clean = denoised_target_latent(prediction, noised_x, sigmas)
    loss_map = (pred_clean - clean[:, :, -1].float()).abs()
    if extra_weight_map is not None:
        # Same diff map as the MSE term: without it the uniform L1 restores
        # full-strength "copy the ref" reward exactly where the MSE was
        # down-weighted. mean-1 renorm keeps the term's scale (and thus
        # LATENT_RECON_LOSS_WEIGHT's meaning) unchanged.
        extra = extra_weight_map.float()
        if extra.shape[-2:] != loss_map.shape[-2:]:
            extra = F.interpolate(extra, size=loss_map.shape[-2:], mode="nearest")
        mean = extra.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        loss_map = loss_map * (extra / mean)
    return loss_map.mean()


def f1_anti_copy_loss(
    prediction: torch.Tensor,
    noised_x: torch.Tensor,
    clean: torch.Tensor,
    sigmas: torch.Tensor,
    dropped_ref_frames: torch.Tensor,
    head_mask: torch.Tensor,
    *,
    margin: float = 0.35,
    sigma_cutoff: float = 0.6,
    head_roi_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    if clean.shape[2] < 2:
        return prediction.sum() * 0.0
    full_idx = max(clean.shape[2] - 2, 0)
    valid = ~dropped_ref_frames[:, full_idx].to(device=prediction.device).bool()
    if head_roi_valid is not None:
        # Samples without a cached head ROI have an all-zero head_mask; without
        # this gate the penalty would cover the face and fight identity losses.
        valid = valid & head_roi_valid.to(device=prediction.device).bool()
    if sigma_cutoff > 0:
        valid = valid & (sigmas.to(device=prediction.device).float() <= float(sigma_cutoff))
    if not bool(valid.any()):
        return prediction.sum() * 0.0

    pred_clean = denoised_target_latent(prediction, noised_x, sigmas)
    full_ref = clean[:, :, full_idx].float()
    mask = 1.0 - head_mask.float()
    if mask.shape[-2:] != pred_clean.shape[-2:]:
        mask = F.interpolate(mask, size=pred_clean.shape[-2:], mode="nearest")
    mask = mask.clamp(0, 1)
    flat_mask = mask.expand(-1, pred_clean.shape[1], -1, -1).reshape(pred_clean.shape[0], -1)
    pred_flat = (pred_clean * mask).reshape(pred_clean.shape[0], -1)
    ref_flat = (full_ref * mask).reshape(full_ref.shape[0], -1)
    enough = flat_mask.sum(dim=1) > 0
    valid = valid & enough
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    sim = F.cosine_similarity(pred_flat[valid], ref_flat[valid], dim=1)
    return (sim - float(margin)).clamp_min(0).mean()


def focal_frequency_loss(prediction: torch.Tensor, target: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    pred_fft = torch.fft.rfft2(prediction[:, :, -1].float(), norm="ortho")
    target_fft = torch.fft.rfft2(target.float(), norm="ortho")
    diff = pred_fft - target_fft
    weight = diff.abs().detach().pow(alpha)
    weight = weight / weight.mean().clamp_min(1e-6)
    return (weight * diff.abs().pow(2)).mean()
