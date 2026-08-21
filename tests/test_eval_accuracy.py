"""Unit tests for eval_accuracy() in eval/score.py.

These tests verify the graceful-degradation contract without requiring a live
model endpoint: with no endpoint configured, eval_accuracy must return a
zero-score, non-passing, non-raising result matching the standard eval dict
contract.
"""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

# Load eval/score.py by path ("eval" is a builtin name and not a package).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCORE_PATH = os.path.join(_REPO_ROOT, "eval", "score.py")
_spec = importlib.util.spec_from_file_location("its_eval_score", _SCORE_PATH)
score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score)

_CONTRACT_KEYS = {"name", "score", "weight", "passed", "details"}


@pytest.fixture
def no_endpoint(monkeypatch):
    monkeypatch.delenv("ITS_ENDPOINT", raising=False)
    monkeypatch.delenv("OPENAI_ENDPOINT", raising=False)


def test_no_endpoint_returns_zero_score(no_endpoint):
    result = score.eval_accuracy()
    assert result["score"] == 0.0


def test_no_endpoint_not_passed(no_endpoint):
    result = score.eval_accuracy()
    assert result["passed"] is False


def test_no_endpoint_does_not_raise(no_endpoint):
    # Must never raise regardless of environment.
    score.eval_accuracy()


def test_result_matches_contract(no_endpoint):
    result = score.eval_accuracy()
    assert set(result) == _CONTRACT_KEYS
    assert result["name"] == "accuracy"
    assert isinstance(result["score"], float)
    assert isinstance(result["weight"], float)
    assert isinstance(result["passed"], bool)
    assert isinstance(result["details"], str)
    assert result["details"]  # non-empty, actionable message


def test_details_mention_endpoint(no_endpoint):
    result = score.eval_accuracy()
    assert "ITS_ENDPOINT" in result["details"]


def test_invalid_budget_rejected(monkeypatch):
    # Even with an endpoint set, an out-of-allowlist budget degrades cleanly.
    monkeypatch.setenv("ITS_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.setenv("ITS_BUDGET", "7")
    result = score.eval_accuracy()
    assert result["score"] == 0.0
    assert result["passed"] is False
    assert "ITS_BUDGET" in result["details"]


def test_allowed_budget_values():
    assert {4, 8} == score._ALLOWED_BUDGETS


def test_accuracy_excluded_from_bare_bundle():
    # The expensive GPU accuracy benchmark must NOT run in the default bundle;
    # it is scored on its own via `--dimension accuracy`.
    assert score.eval_accuracy not in score.EVALS


def test_accuracy_in_dimension_registry():
    assert score._DIMENSIONS["accuracy"] is score.eval_accuracy


def _run_score(args, env=None):
    """Invoke eval/score.py as a subprocess (as the factory runner does)."""
    run_env = dict(os.environ)
    # Never let an ambient endpoint turn this into a live GPU run.
    run_env.pop("ITS_ENDPOINT", None)
    run_env.pop("OPENAI_ENDPOINT", None)
    if env:
        run_env.update(env)
    return subprocess.run(
        [sys.executable, _SCORE_PATH, *args],
        capture_output=True,
        text=True,
        env=run_env,
        cwd=_REPO_ROOT,
        timeout=120,
    )


def test_dimension_accuracy_no_endpoint_emits_top_level_score():
    # `--dimension accuracy` with no endpoint must print a JSON object with a
    # top-level float 'score' == 0.0 and a string 'details', and must not raise.
    proc = _run_score(["--dimension", "accuracy"])
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert isinstance(data["score"], float)
    assert data["score"] == 0.0
    assert isinstance(data["details"], str)
    assert data["details"]
    assert data["name"] == "accuracy"


def test_bare_bundle_excludes_accuracy():
    # The bare bundle (`python eval/score.py`, no args) prints {"results": [...]}
    # and must NOT contain an 'accuracy' entry — accuracy is scored on its own
    # via `--dimension accuracy` so the default bundle never triggers a GPU run.
    # Checked in-process against the real EVALS list (mapped to dimension names
    # through the registry) so this stays fast: actually shelling out to the
    # bundle would run the full pytest suite twice via eval_tests/eval_coverage.
    name_by_fn = {fn: name for name, fn in score._DIMENSIONS.items()}
    bundle_names = {name_by_fn[fn] for fn in score.EVALS}
    assert "accuracy" not in bundle_names
    # Other dimensions remain in the default bundle.
    assert {"tests", "lint", "coverage", "observability"} <= bundle_names
