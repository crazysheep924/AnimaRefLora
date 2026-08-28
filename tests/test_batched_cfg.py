import torch

from anima_reflora.local_ref_ab_infer import sample_target


class DummyModel:
    """Deterministic stand-in: velocity depends linearly on x, the caption
    embedding mean, and the ccip embedding, so cond/uncond genuinely differ and
    any batching mistake (wrong split, wrong broadcast) changes the output."""

    def __init__(self):
        self.calls = 0
        self.max_batch = 0

    def __call__(self, x, timesteps, caption_embeds=None, attention_mask=None, batch=None, config=None):
        self.calls += 1
        self.max_batch = max(self.max_batch, x.shape[0])
        cap = caption_embeds.float().mean(dim=(1, 2)).view(-1, 1, 1, 1, 1)
        ccip = batch["ccip_embeddings"].float().mean(dim=1).view(-1, 1, 1, 1, 1)
        t = timesteps[:, -1].view(-1, 1, 1, 1, 1)
        return 0.1 * x.float() + cap + 0.01 * ccip + 0.001 * t


def _prompt(fill: float, tokens: int = 8, dim: int = 4):
    return {
        "prompt_embeds": torch.full((1, tokens, dim), fill),
        "attn_mask": torch.ones(1, tokens, dtype=torch.int32),
        "t5_input_ids": torch.arange(tokens, dtype=torch.int32).unsqueeze(0),
        "t5_attn_mask": torch.ones(1, tokens, dtype=torch.int32),
    }


def _run(batched_cfg: bool, guidance: float = 4.5):
    torch.manual_seed(0)
    model = DummyModel()
    ref_latents = [torch.randn(16, 8, 8), torch.randn(16, 8, 8)]
    out = sample_target(
        model,
        config=None,
        ref_latents=[r.clone() for r in ref_latents],
        prompt=_prompt(0.5),
        negative=_prompt(-0.25),
        ccip_embedding=torch.randn(768),
        cpm_valid=True,
        steps=4,
        flow_shift=3.0,
        guidance_scale=guidance,
        seed=1234,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batched_cfg=batched_cfg,
    )
    return out, model


def test_batched_matches_sequential_exactly():
    torch.manual_seed(7)
    out_seq, m_seq = _run(batched_cfg=False)
    torch.manual_seed(7)
    out_bat, m_bat = _run(batched_cfg=True)
    assert torch.equal(out_seq, out_bat)
    # sequential: 2 calls/step at batch 1; batched: 1 call/step at batch 2
    assert m_seq.calls == 8 and m_seq.max_batch == 1
    assert m_bat.calls == 4 and m_bat.max_batch == 2


def test_no_cfg_path_unaffected():
    torch.manual_seed(7)
    out_a, m_a = _run(batched_cfg=True, guidance=1.0)
    torch.manual_seed(7)
    out_b, m_b = _run(batched_cfg=False, guidance=1.0)
    assert torch.equal(out_a, out_b)
    # guidance<=1: single cond call per step regardless of the flag
    assert m_a.calls == 4 and m_a.max_batch == 1
    assert m_b.calls == 4 and m_b.max_batch == 1
