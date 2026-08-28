from pathlib import Path
import runpy

import pytest


SCRIPT = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "validate_finished_run.py"))
discover_steps = SCRIPT["discover_steps"]
prompts = SCRIPT["prompts"]


def test_discover_steps_and_prompt_parser(tmp_path: Path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for step in (300, 100, 200):
        (checkpoints / f"lora_step_{step}.safetensors").touch()
    (checkpoints / "optimizer_step_300.pt").touch()

    assert discover_steps(tmp_path) == [100, 200, 300]
    assert prompts(["pose=running, outdoors"]) == [("pose", "running, outdoors")]
    with pytest.raises(ValueError):
        prompts(["missing separator"])
