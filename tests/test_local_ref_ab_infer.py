import torch
from torch import nn
import pytest

from anima_reflora.local_ref_ab_infer import apply_ref_frame_mode, condition_seed, set_ref_conditioner_components, zero_lora_network


def test_condition_seed_can_hold_noise_fixed_for_paired_comparisons():
    assert condition_seed(42, 1, "correct") == 142
    assert condition_seed(42, 1, "blank") == 144
    assert condition_seed(42, 1, "blank", same_condition_seed=True) == 142


def test_apply_ref_frame_mode_masks_expected_reference_frames():
    head = torch.ones(1, 2, 2)
    full = torch.full((1, 2, 2), 2.0)

    out, cpm = apply_ref_frame_mode([head, full], "both")
    assert cpm is True
    assert torch.equal(out[0], head)
    assert torch.equal(out[1], full)

    out, cpm = apply_ref_frame_mode([head, full], "head_only")
    assert cpm is False
    assert torch.equal(out[0], head)
    assert torch.equal(out[1], torch.zeros_like(full))

    out, cpm = apply_ref_frame_mode([head, full], "full_only")
    assert cpm is True
    assert torch.equal(out[0], torch.zeros_like(head))
    assert torch.equal(out[1], full)

    out, cpm = apply_ref_frame_mode([head, full], "blank")
    assert cpm is False
    assert torch.equal(out[0], torch.zeros_like(head))
    assert torch.equal(out[1], torch.zeros_like(full))


def test_apply_ref_frame_mode_single_frame_checkpoint():
    full = torch.full((1, 2, 2), 2.0)

    out, cpm = apply_ref_frame_mode([full], "both")
    assert cpm is True
    assert torch.equal(out[0], full)

    out, cpm = apply_ref_frame_mode([full], "full_only")
    assert cpm is True
    assert torch.equal(out[0], full)

    out, cpm = apply_ref_frame_mode([full], "blank")
    assert cpm is False
    assert torch.equal(out[0], torch.zeros_like(full))

    with pytest.raises(ValueError):
        apply_ref_frame_mode([full], "head_only")


def test_zero_lora_network_only_zeros_network_params():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Linear(2, 2, bias=False)
            self.ref_conditioner = nn.Linear(2, 2, bias=False)

    model = Model()
    with torch.no_grad():
        model.network.weight.fill_(1.0)
        model.ref_conditioner.weight.fill_(2.0)

    assert zero_lora_network(model) == 4
    assert torch.equal(model.network.weight, torch.zeros_like(model.network.weight))
    assert torch.equal(model.ref_conditioner.weight, torch.full_like(model.ref_conditioner.weight, 2.0))


def test_set_ref_conditioner_components_zeros_only_requested_gates():
    class GateBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter_gate = nn.Parameter(torch.tensor(2.0))
            self.cpm_gate = nn.Parameter(torch.tensor(3.0))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.dit = nn.Module()
            self.dit.ref_conditioner = nn.Module()
            self.dit.ref_conditioner.blocks = nn.ModuleDict({"4": GateBlock(), "12": GateBlock()})

    model = Model()
    counts = set_ref_conditioner_components(model, frame_adapter_on=False, cpm_on=True)
    assert counts == {"frame_adapter_gates_zeroed": 2, "cpm_gates_zeroed": 0}
    for block in model.dit.ref_conditioner.blocks.values():
        assert block.adapter_gate.item() == 0.0
        assert block.cpm_gate.item() == 3.0
