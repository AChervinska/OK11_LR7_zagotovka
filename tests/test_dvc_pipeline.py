"""
Тести структури DVC-пайплайну і CI-конфігурації — крок 10-11
(поглиблений рівень). Не запускають `dvc repro` (це робить verify.sh),
а перевіряють, що конвеєр описано коректно.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytestmark = pytest.mark.advanced

ROOT = Path(__file__).resolve().parent.parent


def _dvc_yaml() -> dict:
    return yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))


def test_pipeline_has_prepare_train_evaluate_stages():
    stages = _dvc_yaml()["stages"]
    assert set(stages) == {"prepare", "train", "evaluate"}


def test_train_stage_is_parametrised():
    train = _dvc_yaml()["stages"]["train"]
    assert "train.C" in train["params"]
    assert "models/pipeline" in train["outs"]


def test_evaluate_stage_declares_metrics_file():
    evaluate = _dvc_yaml()["stages"]["evaluate"]
    assert any("metrics.json" in str(entry) for entry in evaluate["metrics"])


def test_params_yaml_matches_train_stage():
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    assert {"data_version", "ngram_max", "C"} <= set(params["train"])


def test_ci_workflow_runs_tests_and_reproduces_pipeline():
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    [job] = workflow["jobs"].values()
    steps = job["steps"]
    assert len(steps) >= 5, "конвеєр має містити щонайменше 5 кроків"

    commands = " ".join(str(step.get("run", "")) for step in steps)
    uses = " ".join(str(step.get("uses", "")) for step in steps)
    assert "actions/checkout" in uses
    assert "pytest" in commands
    assert "dvc repro" in commands
    assert "metrics diff" in commands
