from __future__ import annotations

import json
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass
class RefPosScheme:
    name: str = "identity"
    ref_frame_idx: list[int] = field(default_factory=list)
    spatial_shift: float = 0.0
    temporal: list[int] | None = None
    per_frame_offsets: list[tuple[float, float]] | None = None

    @classmethod
    def for_layout(cls, name: str, frames: int, *, shift: float = 1.0) -> "RefPosScheme":
        if name in {"identity", "off"}:
            return cls(name="identity")
        target_idx = frames - 1
        if name in {"shifted", "shift_refs"}:
            return cls(name=f"shift_refs(s={shift})", ref_frame_idx=list(range(target_idx)), spatial_shift=float(shift))
        if name == "packed":
            return cls(name=f"packed(s={shift})", per_frame_offsets=[(float(i) * shift, 0.0) for i in range(frames)])
        if name == "disjoint":
            offsets: list[tuple[float, float]] = [(0.0, 0.0)] * frames
            if target_idx == 1:
                offsets[0] = (float(shift), float(shift))
            else:
                for i in range(target_idx):
                    offsets[i] = (float(shift), 0.0) if i % 2 == 0 else (0.0, float(shift))
            return cls(name=f"disjoint(s={shift})", per_frame_offsets=offsets)
        raise ValueError(f"Unsupported RoPE ref-position layout: {name}")

    def resolve(self, frames: int, height: int, width: int) -> tuple[list[int], list[int], list[int]]:
        temporal = list(self.temporal) if self.temporal is not None else list(range(frames))
        if len(temporal) != frames:
            raise ValueError(f"temporal length {len(temporal)} != frames {frames}")
        h_offsets = [0] * frames
        w_offsets = [0] * frames
        if self.per_frame_offsets is not None:
            if len(self.per_frame_offsets) != frames:
                raise ValueError(f"per_frame_offsets length {len(self.per_frame_offsets)} != frames {frames}")
            for frame, (h_mult, w_mult) in enumerate(self.per_frame_offsets):
                h_offsets[frame] = int(round(float(h_mult) * height))
                w_offsets[frame] = int(round(float(w_mult) * width))
            return temporal, h_offsets, w_offsets
        h_shift = int(round(float(self.spatial_shift) * height))
        w_shift = int(round(float(self.spatial_shift) * width))
        for frame in self.ref_frame_idx:
            if frame < 0 or frame >= frames:
                raise ValueError(f"ref_frame_idx {frame} out of range for {frames} frames")
            h_offsets[frame] = h_shift
            w_offsets[frame] = w_shift
        return temporal, h_offsets, w_offsets


@dataclass(frozen=True)
class RefPosSidecar:
    path: Path
    scheme: RefPosScheme
    frames: int | None


def _make_patched_generate_embeddings(scheme: RefPosScheme):
    def generate_embeddings(
        self,
        B_T_H_W_C,
        fps=None,
        h_ntk_factor=None,
        w_ntk_factor=None,
        t_ntk_factor=None,
    ):
        if getattr(self, "enable_fps_modulation", False):
            raise NotImplementedError("RoPE ref-position patch supports enable_fps_modulation=False only")

        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_freqs = 1.0 / (h_theta ** self.dim_spatial_range)
        w_freqs = 1.0 / (w_theta ** self.dim_spatial_range)
        t_freqs = 1.0 / (t_theta ** self.dim_temporal_range)

        _batch, frames, height, width, _channels = B_T_H_W_C
        max_h = int(getattr(self, "max_h"))
        max_w = int(getattr(self, "max_w"))
        max_t = int(getattr(self, "max_t"))
        t_index, h_offsets, w_offsets = scheme.resolve(frames, height, width)
        if max(t_index) >= max_t:
            raise ValueError(f"RoPE temporal index {max(t_index)} >= max_t {max_t}")
        if max(h_offsets) + height > max_h:
            raise ValueError(f"RoPE h offset {max(h_offsets)} + {height} > max_h {max_h}")
        if max(w_offsets) + width > max_w:
            raise ValueError(f"RoPE w offset {max(w_offsets)} + {width} > max_w {max_w}")

        seq = self.seq
        embeddings = []
        for frame in range(frames):
            et = torch.outer(seq[t_index[frame] : t_index[frame] + 1], t_freqs)
            eh = torch.outer(seq[h_offsets[frame] : h_offsets[frame] + height], h_freqs)
            ew = torch.outer(seq[w_offsets[frame] : w_offsets[frame] + width], w_freqs)
            et_b = et.view(1, 1, -1).expand(height, width, -1)
            eh_b = eh.view(height, 1, -1).expand(height, width, -1)
            ew_b = ew.view(1, width, -1).expand(height, width, -1)
            half = torch.cat([et_b, eh_b, ew_b], dim=-1)
            embeddings.append(torch.cat([half, half], dim=-1))
        out = torch.stack(embeddings, dim=0)
        return out.reshape(frames * height * width, 1, 1, out.shape[-1]).float()

    return generate_embeddings


def install_refpos(pos_embedder, scheme: RefPosScheme):
    if hasattr(pos_embedder, "_anima_refpos_restore"):
        installed = getattr(pos_embedder, "_refpos_scheme", None)
        if installed is not None and not schemes_equivalent(installed, scheme):
            raise RuntimeError(f"Different RoPE ref-position scheme already installed: {describe_scheme(installed)}")
        return pos_embedder._anima_refpos_restore
    original = pos_embedder.generate_embeddings
    pos_embedder.generate_embeddings = types.MethodType(_make_patched_generate_embeddings(scheme), pos_embedder)
    pos_embedder._refpos_scheme = scheme

    def restore():
        pos_embedder.generate_embeddings = original
        if hasattr(pos_embedder, "_refpos_scheme"):
            del pos_embedder._refpos_scheme
        if hasattr(pos_embedder, "_anima_refpos_restore"):
            del pos_embedder._anima_refpos_restore

    pos_embedder._anima_refpos_restore = restore
    return restore


def install_on_anima(anima, scheme: RefPosScheme):
    pos_embedder = getattr(anima, "pos_embedder", None)
    if pos_embedder is None:
        raise AttributeError("Anima model has no .pos_embedder for RoPE ref-position patch")
    return install_refpos(pos_embedder, scheme)


def rope_state_path(lora_path: str | Path) -> Path:
    path = Path(lora_path)
    if path.name.startswith("lora_step_"):
        return path.with_name(path.name.replace("lora_step_", "rope_refpos_step_", 1)).with_suffix(".json")
    return path.with_name(f"{path.stem}.rope_refpos.json")


def save_refpos_scheme(scheme: RefPosScheme, path: str | Path, *, frames: int) -> str:
    payload = {
        "format": "anima_reflora_rope_refpos_v1",
        "name": scheme.name,
        "ref_frame_idx": list(scheme.ref_frame_idx),
        "spatial_shift": float(scheme.spatial_shift),
        "temporal": list(scheme.temporal) if scheme.temporal is not None else None,
        "per_frame_offsets": [list(item) for item in scheme.per_frame_offsets] if scheme.per_frame_offsets else None,
        "frames": int(frames),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def load_refpos_sidecar(path: str | Path) -> RefPosSidecar:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return refpos_from_payload(payload, path)


def refpos_from_payload(payload: dict, path: Path) -> RefPosSidecar:
    scheme = RefPosScheme(
        name=payload.get("name", "identity"),
        ref_frame_idx=list(payload.get("ref_frame_idx", [])),
        spatial_shift=float(payload.get("spatial_shift", 0.0)),
        temporal=list(payload["temporal"]) if payload.get("temporal") is not None else None,
        per_frame_offsets=[tuple(item) for item in payload["per_frame_offsets"]]
        if payload.get("per_frame_offsets") is not None
        else None,
    )
    frames = payload.get("frames")
    return RefPosSidecar(path=path, scheme=scheme, frames=int(frames) if frames is not None else None)


def load_refpos_for_checkpoint(lora_path: str | Path) -> Optional[RefPosSidecar]:
    """Resolve the RoPE scheme shipped with a checkpoint: from bundle metadata
    for single-file bundles, from the step-matched JSON sidecar otherwise.
    Returns None when the checkpoint carries no scheme."""
    from .bundle import bundle_rope_payload, is_bundle

    path = Path(lora_path)
    if is_bundle(path):
        payload = bundle_rope_payload(path)
        return refpos_from_payload(payload, path) if payload else None
    sidecar = rope_state_path(path)
    if not sidecar.exists():
        return None
    return load_refpos_sidecar(sidecar)


def _scheme_signature(scheme: RefPosScheme):
    return (
        tuple(scheme.ref_frame_idx),
        float(scheme.spatial_shift),
        tuple(scheme.temporal) if scheme.temporal is not None else None,
        tuple((float(h), float(w)) for h, w in scheme.per_frame_offsets)
        if scheme.per_frame_offsets is not None
        else None,
    )


def schemes_equivalent(left: RefPosScheme, right: RefPosScheme) -> bool:
    return _scheme_signature(left) == _scheme_signature(right)


def describe_scheme(scheme: RefPosScheme) -> str:
    return (
        f"{scheme.name} ref={scheme.ref_frame_idx} shift={scheme.spatial_shift:g} "
        f"temporal={scheme.temporal} offsets={scheme.per_frame_offsets}"
    )


def assert_sidecar_compatible(
    sidecar: RefPosSidecar,
    *,
    expected_frames: Optional[int] = None,
    expected_scheme: Optional[RefPosScheme] = None,
    context: str = "RoPE sidecar",
) -> None:
    if expected_frames is not None and sidecar.frames != int(expected_frames):
        raise RuntimeError(f"{context} frames mismatch: sidecar={sidecar.frames} requested={expected_frames}")
    if expected_scheme is not None and not schemes_equivalent(sidecar.scheme, expected_scheme):
        raise RuntimeError(
            f"{context} scheme mismatch:\n  sidecar: {describe_scheme(sidecar.scheme)}\n"
            f"  requested: {describe_scheme(expected_scheme)}"
        )


# ---------------------------------------------------------------------------
# centralized fail-safe application: one place every checkpoint loader uses, so a
# new/forgetful eval/infer loader fails LOUD instead of silently scoring a
# RoPE-trained checkpoint at identity positions.
# ---------------------------------------------------------------------------
def is_refpos_installed(anima) -> bool:
    """True if a RefPosScheme is already installed on this model's pos_embedder."""
    return hasattr(getattr(anima, "pos_embedder", None), "_refpos_scheme")


def maybe_apply_sidecar(
    anima,
    lora_path: str | Path | None,
    *,
    expected_frames: Optional[int] = None,
    expected_scheme: Optional[RefPosScheme] = None,
) -> Optional[RefPosScheme]:
    """Install the checkpoint's RoPE sidecar scheme if present (idempotent).

    Returns the scheme in effect (existing or newly applied) or None. Skips if a
    scheme is already installed (e.g. train()'s config-driven install) so callers
    can layer this defensively without double-relocating.
    """
    if not lora_path:
        return None
    state = load_refpos_for_checkpoint(lora_path)
    if state is None:
        return None
    assert_sidecar_compatible(
        state,
        expected_frames=expected_frames,
        expected_scheme=expected_scheme,
        context=f"checkpoint {Path(lora_path).name}",
    )
    if is_refpos_installed(anima):
        installed = getattr(anima.pos_embedder, "_refpos_scheme", None)
        if installed is not None and not schemes_equivalent(installed, state.scheme):
            raise RuntimeError(
                f"checkpoint {Path(lora_path).name} carries RoPE sidecar "
                f"{state.path.name}, but a different scheme is already installed.\n"
                f"  sidecar: {describe_scheme(state.scheme)}\n"
                f"  installed: {describe_scheme(installed)}"
            )
        return installed
    install_on_anima(anima, state.scheme)
    print(f"[rope-refpos] applied sidecar {state.path.name} -> {state.scheme.name}", flush=True)
    return state.scheme


def assert_sidecar_applied(
    anima,
    lora_path: str | Path | None,
    *,
    expected_frames: Optional[int] = None,
    expected_scheme: Optional[RefPosScheme] = None,
) -> None:
    """Loud guard: refuse to proceed if a RoPE sidecar exists but no scheme is
    installed (which would mean eval/infer positions diverge from training)."""
    if not lora_path:
        return
    state = load_refpos_for_checkpoint(lora_path)
    if state is None:
        return
    assert_sidecar_compatible(
        state,
        expected_frames=expected_frames,
        expected_scheme=expected_scheme,
        context=f"checkpoint {Path(lora_path).name}",
    )
    if not is_refpos_installed(anima):
        raise RuntimeError(
            f"checkpoint {Path(lora_path).name} carries a RoPE relocation sidecar "
            f"({state.path.name}) but no scheme was installed — eval/infer "
            f"positions would diverge from training. This loader must apply the "
            f"sidecar (the default) or explicitly opt out."
        )
    installed = getattr(anima.pos_embedder, "_refpos_scheme", None)
    if installed is not None and not schemes_equivalent(installed, state.scheme):
        raise RuntimeError(
            f"checkpoint {Path(lora_path).name} carries RoPE sidecar "
            f"{state.path.name}, but the installed scheme differs.\n"
            f"  sidecar: {describe_scheme(state.scheme)}\n"
            f"  installed: {describe_scheme(installed)}"
        )


__all__ = [
    "RefPosScheme",
    "RefPosSidecar",
    "assert_sidecar_applied",
    "assert_sidecar_compatible",
    "describe_scheme",
    "install_on_anima",
    "install_refpos",
    "is_refpos_installed",
    "load_refpos_sidecar",
    "maybe_apply_sidecar",
    "rope_state_path",
    "save_refpos_scheme",
    "schemes_equivalent",
]
