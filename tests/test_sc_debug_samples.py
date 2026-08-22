"""Tests for C1 guarded per-sample draw logging (H1, exp-20).

Covers the ``ITS_SC_DEBUG_SAMPLES`` diagnostic that persists all K per-sample
draws + their ``_aggregate_logprob`` values + the vote keys/mode + the resolved
``selected_index`` to one JSON file per item:

  (a) guard UNSET => no file written AND the returned winner is byte-identical
      to the plurality/confidence baseline (no selection-path perturbation);
  (b) guard SET => a file is written containing the K draws + logprobs + vote
      metadata;
  (c) the write is fail-safe -- an unwritable directory never raises into the
      self-consistency path and leaves the winner unchanged.

New tests only -- no existing tests are modified.
"""

import json
import os
import random

import pytest

from its_hub.core.algorithms.self_consistency import (
    _DEBUG_SAMPLES_ENV_VAR,
    SelfConsistency,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _response(content: str, mean_logprob: float | None) -> dict:
    """Build a response whose ``_aggregate_logprob`` equals ``mean_logprob``.

    A single-token logprob content list makes the mean exactly the given value.
    ``mean_logprob=None`` produces a response with no logprob metadata.
    """
    resp = {"role": "assistant", "content": content}
    if mean_logprob is not None:
        resp["_logprobs"] = {"content": [{"logprob": mean_logprob}]}
    return resp


def _draws() -> list[dict]:
    """Four draws: a clear plurality winner ("A") plus one minority ("B")."""
    return [
        _response("The answer is \\boxed{A}", -0.10),
        _response("The answer is \\boxed{A}", -0.30),
        _response("The answer is \\boxed{A}", -0.20),
        _response("The answer is \\boxed{B}", -0.05),
    ]


# ---------------------------------------------------------------------------
# (a) guard UNSET => no write + byte-identical winner
# ---------------------------------------------------------------------------


class TestGuardUnset:
    def test_no_file_written_when_unset(self, monkeypatch, tmp_path):
        monkeypatch.delenv(_DEBUG_SAMPLES_ENV_VAR, raising=False)
        sc = SelfConsistency()
        sc._process_responses(_draws(), return_response_only=False)
        # Nothing should have been created anywhere under tmp_path.
        assert list(tmp_path.iterdir()) == []

    def test_empty_value_is_treated_as_off(self, monkeypatch, tmp_path):
        monkeypatch.setenv(_DEBUG_SAMPLES_ENV_VAR, "   ")
        sc = SelfConsistency()
        sc._process_responses(_draws(), return_response_only=False)
        assert list(tmp_path.iterdir()) == []

    def test_winner_identical_with_and_without_guard_plurality(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.delenv("ITS_SC_VOTE", raising=False)
        # The clear plurality group has 3 members, so member selection uses
        # random.choice; seed identically before both calls so any difference is
        # attributable to the logging side-effect (which must not touch random).
        monkeypatch.delenv(_DEBUG_SAMPLES_ENV_VAR, raising=False)
        random.seed(0)
        baseline = SelfConsistency()._process_responses(
            _draws(), return_response_only=False
        )
        # Same inputs + same RNG state with logging ON must not change selection.
        monkeypatch.setenv(_DEBUG_SAMPLES_ENV_VAR, str(tmp_path))
        random.seed(0)
        with_logging = SelfConsistency()._process_responses(
            _draws(), return_response_only=False
        )
        assert with_logging.selected_index == baseline.selected_index
        assert with_logging.the_one == baseline.the_one
        # The clear plurality winner is one of the "A" draws (indices 0..2).
        assert baseline.selected_index in (0, 1, 2)

    def test_winner_identical_with_and_without_guard_confidence(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("ITS_SC_VOTE", "confidence")
        monkeypatch.delenv(_DEBUG_SAMPLES_ENV_VAR, raising=False)
        baseline = SelfConsistency()._process_responses(
            _draws(), return_response_only=False
        )
        monkeypatch.setenv(_DEBUG_SAMPLES_ENV_VAR, str(tmp_path))
        with_logging = SelfConsistency()._process_responses(
            _draws(), return_response_only=False
        )
        assert with_logging.selected_index == baseline.selected_index
        assert with_logging.the_one == baseline.the_one


# ---------------------------------------------------------------------------
# (b) guard SET => file with K draws + logprobs + vote metadata
# ---------------------------------------------------------------------------


class TestGuardSet:
    def test_file_written_with_all_draws_and_metadata(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ITS_SC_VOTE", raising=False)
        monkeypatch.setenv(_DEBUG_SAMPLES_ENV_VAR, str(tmp_path))
        responses = _draws()
        result = SelfConsistency()._process_responses(
            responses, return_response_only=False
        )

        files = list(tmp_path.glob("*_samples.json"))
        assert len(files) == 1
        payload = json.loads(files[0].read_text(encoding="utf-8"))

        # All K draws persisted, aligned by index.
        assert payload["num_draws"] == len(responses)
        assert len(payload["draws"]) == len(responses)
        assert [d["index"] for d in payload["draws"]] == list(range(len(responses)))

        # Contents + logprobs persisted per draw.
        assert payload["draws"][0]["content"] == "The answer is \\boxed{A}"
        assert payload["draws"][0]["aggregate_logprob"] == pytest.approx(-0.10)
        assert payload["draws"][3]["content"] == "The answer is \\boxed{B}"
        assert payload["draws"][3]["aggregate_logprob"] == pytest.approx(-0.05)

        # Vote metadata persisted and consistent with the returned result.
        assert payload["vote_mode"] == "plurality"
        assert payload["selected_index"] == result.selected_index
        assert isinstance(payload["vote_keys"], list)
        assert len(payload["vote_keys"]) == len(payload["eligible_indices"])

    def test_vote_mode_recorded_as_confidence(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ITS_SC_VOTE", "confidence")
        monkeypatch.setenv(_DEBUG_SAMPLES_ENV_VAR, str(tmp_path))
        SelfConsistency()._process_responses(_draws(), return_response_only=False)
        payload = json.loads(
            next(tmp_path.glob("*_samples.json")).read_text(encoding="utf-8")
        )
        assert payload["vote_mode"] == "confidence"

    def test_filename_is_deterministic_for_same_draws(self, monkeypatch, tmp_path):
        monkeypatch.setenv(_DEBUG_SAMPLES_ENV_VAR, str(tmp_path))
        SelfConsistency()._process_responses(_draws(), return_response_only=False)
        first = {p.name for p in tmp_path.glob("*_samples.json")}
        # Re-running the same item overwrites the same file (no duplicate).
        SelfConsistency()._process_responses(_draws(), return_response_only=False)
        second = {p.name for p in tmp_path.glob("*_samples.json")}
        assert first == second
        assert len(second) == 1

    def test_missing_logprobs_recorded_as_null(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ITS_SC_VOTE", raising=False)
        monkeypatch.setenv(_DEBUG_SAMPLES_ENV_VAR, str(tmp_path))
        responses = [
            _response("The answer is \\boxed{A}", None),
            _response("The answer is \\boxed{A}", None),
            _response("The answer is \\boxed{B}", None),
        ]
        SelfConsistency()._process_responses(responses, return_response_only=False)
        payload = json.loads(
            next(tmp_path.glob("*_samples.json")).read_text(encoding="utf-8")
        )
        assert all(d["aggregate_logprob"] is None for d in payload["draws"])

    def test_nested_directory_is_created(self, monkeypatch, tmp_path):
        target = tmp_path / "cache" / "debug"
        monkeypatch.setenv(_DEBUG_SAMPLES_ENV_VAR, str(target))
        SelfConsistency()._process_responses(_draws(), return_response_only=False)
        assert target.is_dir()
        assert len(list(target.glob("*_samples.json"))) == 1


# ---------------------------------------------------------------------------
# (c) fail-safe: logging failure never raises into the SC path
# ---------------------------------------------------------------------------


class TestFailSafe:
    def test_unwritable_dir_does_not_raise_and_winner_unchanged(
        self, monkeypatch, tmp_path
    ):
        # Point the guard at a path that cannot be a directory (a regular file),
        # so os.makedirs fails -- the SC path must still return the winner.
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("x", encoding="utf-8")
        monkeypatch.delenv("ITS_SC_VOTE", raising=False)

        monkeypatch.delenv(_DEBUG_SAMPLES_ENV_VAR, raising=False)
        baseline = SelfConsistency()._process_responses(
            _draws(), return_response_only=False
        )

        monkeypatch.setenv(_DEBUG_SAMPLES_ENV_VAR, str(blocker / "sub"))
        result = SelfConsistency()._process_responses(
            _draws(), return_response_only=False
        )
        assert result.selected_index == baseline.selected_index
        # The blocker file is untouched (no partial write leaked).
        assert blocker.read_text(encoding="utf-8") == "x"

    def test_helper_returns_none_when_off(self, monkeypatch):
        monkeypatch.delenv(_DEBUG_SAMPLES_ENV_VAR, raising=False)
        out = SelfConsistency._maybe_dump_debug_samples(
            responses=_draws(),
            eligible_indices=[0, 1, 2, 3],
            vote_keys=["a", "a", "a", "b"],
            vote_mode="plurality",
            selected_index=0,
        )
        assert out is None

    def test_helper_returns_path_when_on(self, monkeypatch, tmp_path):
        monkeypatch.setenv(_DEBUG_SAMPLES_ENV_VAR, str(tmp_path))
        out = SelfConsistency._maybe_dump_debug_samples(
            responses=_draws(),
            eligible_indices=[0, 1, 2, 3],
            vote_keys=["a", "a", "a", "b"],
            vote_mode="plurality",
            selected_index=0,
        )
        assert out is not None
        assert os.path.isfile(out)
