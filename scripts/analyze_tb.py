#!/usr/bin/env python3
"""Dump scalar trends from a TB event dir (run inside the docker image)."""
import sys
from tensorboard.backend.event_processing import event_accumulator

path = sys.argv[1]
ea = event_accumulator.EventAccumulator(
    path, size_guidance={event_accumulator.SCALARS: 0}
)
ea.Reload()
tags = ea.Tags().get("scalars", [])
print(f"# scalar tags ({len(tags)}):")
for t in tags:
    print(f"  {t}")
print()

def summarize(tag):
    ev = ea.Scalars(tag)
    if not ev:
        return
    steps = [e.step for e in ev]
    vals = [e.value for e in ev]
    n = len(vals)
    print(f"== {tag}  (n={n}, step {steps[0]}..{steps[-1]}) ==")
    # sample ~12 evenly spaced points
    idxs = sorted(set(int(round(i * (n - 1) / 11)) for i in range(12)))
    for i in idxs:
        print(f"   step {steps[i]:>7}: {vals[i]:.5f}")
    # tail mean vs early mean
    k = max(1, n // 10)
    early = sum(vals[:k]) / k
    late = sum(vals[-k:]) / k
    print(f"   early(mean {k})={early:.5f}  late(mean {k})={late:.5f}  delta={late-early:+.5f}")
    print()

for t in tags:
    summarize(t)
