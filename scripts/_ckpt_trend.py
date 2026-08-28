import glob, os, re
import torch
from safetensors.torch import load_file

CKPT = "/ckpt"
STEPS = [45000, 50000, 55000, 60000, 65000, 70000]
GROUPS = ["lora", "ref_conditioner", "crepa_projector"]

def load(g, s):
    p = os.path.join(CKPT, f"{g}_step_{s}.safetensors")
    return load_file(p, device="cpu") if os.path.exists(p) else None

def fnorm(t):
    return float(t.detach().to(torch.float32).norm(2))

def total_norm(d):
    return sum(fnorm(t)**2 for t in d.values() if torch.is_floating_point(t))**0.5

cache = {(g, s): load(g, s) for g in GROUPS for s in STEPS}

# 1) total norm trajectory
print("="*90)
print("1) TOTAL NORM per group across snapshots")
print("="*90)
print("group".ljust(18) + "".join(f"{s:>11d}" for s in STEPS))
for g in GROUPS:
    row = g.ljust(18)
    for s in STEPS:
        d = cache[(g, s)]
        row += f"{total_norm(d):11.5g}" if d else f"{'-':>11}"
    print(row)

# 2) per-block gates across snapshots
print("\n" + "="*90)
print("2) ref_conditioner per-block GATES across snapshots")
print("="*90)
# collect gate keys from first available
sample = next(cache[("ref_conditioner", s)] for s in STEPS if cache[("ref_conditioner", s)])
gate_keys = sorted(k for k, v in sample.items() if k.endswith("gate") and v.numel() <= 4)
for gk in gate_keys:
    print(f"\n  {gk}")
    print("    " + "".join(f"{s:>11d}" for s in STEPS))
    row = "    "
    vals = []
    for s in STEPS:
        d = cache[("ref_conditioner", s)]
        if d and gk in d:
            v = float(d[gk].detach().to(torch.float32).flatten()[0])
            vals.append(v)
            row += f"{v:11.5g}"
        else:
            row += f"{'-':>11}"
    print(row)
    if len(vals) >= 2:
        print(f"    growth 45k->70k: {vals[-1]-vals[0]:+.5g}  ({100*(vals[-1]/vals[0]-1):+.1f}%)")

# 3) per-step movement (consecutive) per group, rel%
print("\n" + "="*90)
print("3) CONSECUTIVE movement rel% (||Wb-Wa|| / ||Wa||)")
print("="*90)
def delta(a, b):
    sq = 0.0
    for k, tb in b.items():
        ta = a.get(k)
        if ta is None or ta.shape != tb.shape or not torch.is_floating_point(tb):
            continue
        sq += fnorm(tb.to(torch.float32) - ta.to(torch.float32))**2
    return sq**0.5
for g in GROUPS:
    print(f"\n  [{g}]")
    for a, b in zip(STEPS, STEPS[1:]):
        da, db = cache[(g, a)], cache[(g, b)]
        if not da or not db:
            continue
        d = delta(da, db)
        base = total_norm(da)
        print(f"    {a}->{b}: abs={d:9.4g}  rel={100*d/base:7.3g}%")

# 4) LoKr per-block MLP vs attention movement 45k->70k
print("\n" + "="*90)
print("4) LoKr movement by module TYPE (45k->70k)")
print("="*90)
la, lb = cache[("lora", 45000)], cache[("lora", 70000)]
by_type = {}
for k, tb in lb.items():
    ta = la.get(k)
    if ta is None or ta.shape != tb.shape or not torch.is_floating_point(tb):
        continue
    d = fnorm(tb.to(torch.float32) - ta.to(torch.float32))
    typ = "mlp" if "mlp" in k else ("attn" if "attn" in k or "attention" in k else "other")
    by_type.setdefault(typ, 0.0)
    by_type[typ] += d**2
for typ, sq in sorted(by_type.items(), key=lambda kv: -kv[1]):
    print(f"    {typ:8s} aggregate abs move = {sq**0.5:.4g}")
