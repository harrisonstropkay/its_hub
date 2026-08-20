"""Unit tests for eval_accuracy() in eval/score.py.

These tests verify the graceful-degradation contract without requiring a live
model endpoint: with no endpoint configured, eval_accuracy must return a
zero-score, non-passing, non-raising result matching the standard eval dict
contract.
"""

import importlib.util
import os

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


def test_eval_accuracy_registered():
    assert score.eval_accuracy in score.EVALS
