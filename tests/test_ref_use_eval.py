import json

from anima_reflora.config import parse_config
from anima_reflora.ref_use_eval import run_ref_use_eval, write_eval_output


def test_ref_use_eval_synthetic_tiny(tmp_path):
    out = tmp_path / "eval"
    config = parse_config(
        [
            "--stage", "rope-smoke",
            "--run-name", "eval-test",
            "--out-dir", str(tmp_path / "runs"),
            "--storage", str(tmp_path / "storage"),
            "--backend", "tiny",
            "--synthetic-data",
            "--dtype", "fp32",
            "--from-scratch",
            "--ref-eval-refs", "3",
            "--ref-eval-seeds", "0,1",
        ]
    )
    report = run_ref_use_eval(config)
    assert "summary" in report
    assert "results" in report
    assert report["frames"] == 3
    assert report["backend"] == "tiny"
    summary = report["summary"]
    assert "correct_mse_mean" in summary
    assert "wrong_mse_mean" in summary
    assert "blank_mse_mean" in summary
    assert "gap_correct_vs_wrong" in summary

    metrics_path = write_eval_output(report, out)
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "summary" in payload
    assert payload["conditions"] == ["correct", "wrong", "blank"]


def test_ref_use_eval_t2_synthetic(tmp_path):
    config = parse_config(
        [
            "--stage", "plan",
            "--run-name", "eval-t2",
            "--out-dir", str(tmp_path / "runs"),
            "--storage", str(tmp_path / "storage"),
            "--backend", "tiny",
            "--synthetic-data",
            "--dtype", "fp32",
            "--from-scratch",
            "--frames", "2",
        ]
    )
    report = run_ref_use_eval(config)
    assert report["frames"] == 2
    assert "summary" in report
    assert len(report["results"]) > 0
