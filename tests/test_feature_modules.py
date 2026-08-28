import torch
from torch import nn

from anima_reflora.crepa import CrepaHiddenCapture, CrepaProjector, crepa_hidden_loss
from anima_reflora.noise import apply_reference_dropout
from anima_reflora.ref_conditioning import AdapterGateBlock
from anima_reflora.rope_refpos import RefPosScheme, install_refpos


def test_structured_dropout_can_drop_full_reference_and_both():
    clean = torch.ones(64, 1, 3, 2, 2)
    generator = torch.Generator().manual_seed(3)
    _out, stats = apply_reference_dropout(
        clean,
        prob=1.0,
        mode="blank",
        t3_mode="structured",
        generator=generator,
        return_stats=True,
    )
    dropped = stats["dropped_ref_frames"]
    assert dropped[:, 0].any()
    assert dropped[:, 1].any()
    assert (dropped[:, 0] & dropped[:, 1]).any()
    ccip_valid = torch.ones(dropped.shape[0], dtype=torch.bool)
    full_ref_idx = 1
    cpm_valid = ccip_valid & ~dropped[:, full_ref_idx].cpu()
    assert not cpm_valid.all()


def test_adapter_gate_updates_target_frame_only():
    torch.manual_seed(0)
    block = AdapterGateBlock(hidden_dim=4, adapter_dim=4, num_heads=1, adapter_enabled=True)
    block.adapter_gate.data.fill_(1.0)
    x = torch.randn(2, 3, 2, 2, 4, requires_grad=True)
    out = block(x)
    assert torch.allclose(out[:, 0], x[:, 0])
    assert torch.allclose(out[:, 1], x[:, 1])
    assert not torch.allclose(out[:, 2], x[:, 2])
    out[:, 2].sum().backward()
    assert block.adapter_gate.grad is not None


class FakeRopeEmbedder:
    enable_fps_modulation = False
    h_ntk_factor = 1.0
    w_ntk_factor = 1.0
    t_ntk_factor = 1.0
    max_h = 16
    max_w = 16
    max_t = 8

    def __init__(self):
        self.dim_spatial_range = torch.arange(0, 4, 2).float() / 8
        self.dim_temporal_range = torch.arange(0, 4, 2).float() / 8
        self.seq = torch.arange(32).float()

    def generate_embeddings(self, shape, **_kwargs):
        _b, frames, height, width, _c = shape
        frame_values = []
        h_freqs = 1.0 / (10000.0 ** self.dim_spatial_range)
        w_freqs = 1.0 / (10000.0 ** self.dim_spatial_range)
        t_freqs = 1.0 / (10000.0 ** self.dim_temporal_range)
        for frame in range(frames):
            et = torch.outer(self.seq[frame : frame + 1], t_freqs)
            eh = torch.outer(self.seq[:height], h_freqs)
            ew = torch.outer(self.seq[:width], w_freqs)
            half = torch.cat(
                [
                    et.view(1, 1, -1).expand(height, width, -1),
                    eh.view(height, 1, -1).expand(height, width, -1),
                    ew.view(1, width, -1).expand(height, width, -1),
                ],
                dim=-1,
            )
            frame_values.append(torch.cat([half, half], dim=-1))
        out = torch.stack(frame_values)
        return out.reshape(frames * height * width, 1, 1, out.shape[-1])


def test_rope_refpos_identity_and_disjoint_patch():
    embedder = FakeRopeEmbedder()
    shape = torch.Size([1, 3, 4, 4, 12])
    baseline = embedder.generate_embeddings(shape)
    restore = install_refpos(embedder, RefPosScheme.for_layout("identity", 3))
    try:
        assert torch.allclose(embedder.generate_embeddings(shape), baseline)
    finally:
        restore()
    restore = install_refpos(embedder, RefPosScheme.for_layout("disjoint", 3, shift=1.0))
    try:
        shifted = embedder.generate_embeddings(shape)
    finally:
        restore()
    frame_tokens = 4 * 4
    assert not torch.allclose(shifted[:frame_tokens], baseline[:frame_tokens])
    assert not torch.allclose(shifted[frame_tokens : 2 * frame_tokens], baseline[frame_tokens : 2 * frame_tokens])
    assert torch.allclose(shifted[2 * frame_tokens :], baseline[2 * frame_tokens :])


class CaptureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4)])

    def forward(self, x):
        return self.blocks[0](x)


def test_crepa_hidden_hook_and_projector_receive_gradients():
    model = CaptureModel()
    capture = CrepaHiddenCapture(model, 0)
    x = torch.randn(2, 3, 2, 2, 4, requires_grad=True)
    hidden = model(x)
    assert capture.hidden is hidden
    projector = CrepaProjector(in_dim=4, embedding_dim=8)
    embeddings = torch.randn(2, 8)
    valid = torch.tensor([True, True])
    loss, metrics = crepa_hidden_loss(projector, capture.hidden, embeddings, valid, frames=3)
    loss.backward()
    assert metrics["crepa_valid_fraction"] == 1.0
    assert any(param.grad is not None for param in projector.parameters())
    assert x.grad is not None


def test_crepa_sigma_cutoff_keeps_low_sigma_samples():
    projector = nn.Identity()
    hidden = torch.zeros(2, 3, 1, 1, 4)
    hidden[0, -1, 0, 0, 0] = 1.0
    hidden[1, -1, 0, 0, 1] = 1.0
    embeddings = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]])
    valid = torch.tensor([True, True])
    sigmas = torch.tensor([0.2, 0.8])

    loss, metrics = crepa_hidden_loss(projector, hidden, embeddings, valid, frames=3, sigmas=sigmas, sigma_cutoff=0.5)

    assert metrics["crepa_valid_fraction"] == 0.5
    assert loss.item() == 0.0
