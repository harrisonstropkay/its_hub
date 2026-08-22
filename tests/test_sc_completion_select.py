"""Tests for completion-status-aware selection (H1) and telemetry (H2), cycle-3.

Exercises ONLY the Builder's own diff logic in ``self_consistency.py``:
- the structural completion signal (``_has_real_boxed`` / ``_is_completed_response``);
- the ``ITS_SC_COMPLETION_SELECT`` / ``ITS_SC_TRUNC_CAP_CHARS`` / ``ITS_SC_LOG_COMPLETION``
  env resolvers;
- the ``_project_responses`` completion filter and its three preserved guarantees:
    (a) all-samples-complete OR flag-OFF  -> BYTE-IDENTICAL to baseline,
    (b) all-samples-truncate              -> completed subset empty -> full set
        (never-zero-candidate uniform fallback),
    (c) correct index mapping through BOTH the flat and tuple/hierarchical paths;
- the H2 per-sample completion-status structured log record.

New tests only -- no existing tests are modified.
"""

import json
import logging

import pytest

from its_hub.core.algorithms.self_consistency import (
    _DEFAULT_TRUNC_CAP_CHARS,
    SelfConsistency,
    _has_real_boxed,
    _resolve_completion_select,
    _resolve_log_completion,
    _resolve_trunc_cap,
    create_regex_projection_function,
)

# Regex projection matching the boxed-answer shape used on the scored math path;
# returns a single-element tuple -> exercises the tuple/hierarchical path.
_BOXED_PROJ = create_regex_projection_function(r"\\boxed\{([^}]+)\}")


def _completed(letter: str) -> dict:
    """A short, finished-and-boxed response (completed under the default cap)."""
    return {"content": f"Reasoning here. The answer is \\boxed{{{letter}}}."}


def _truncated_no_box(letter: str, n: int = 5000) -> dict:
    """A long, un-boxed essay (truncation regime). Projection is still non-empty."""
    return {"content": (letter + " ") * n}


def _truncated_with_box(letter: str, n: int = 9000) -> dict:
    """A long essay that DOES contain a box but over-runs the cap.

    Guards the 'false completion' failure mode: a box emitted early followed by a
    truncated over-run must NOT count as completed (has_box True, len >= cap).
    """
    return {"content": f"\\boxed{{{letter}}} " + ("x" * n)}


# --------------------------------------------------------------------------- #
# structural completion signal
# --------------------------------------------------------------------------- #
class TestHasRealBoxed:
    @pytest.mark.parametrize(
        "content",
        [
            r"\boxed{42}",
            r"final: \boxed{A}.",
            r"\boxed{\frac{1}{2}}",  # nested braces balance correctly
            r"\boxed{C} and later \boxed{D}",  # last one still closes
        ],
    )
    def test_real_closed_box_detected(self, content):
        assert _has_real_boxed(content) is True

    @pytest.mark.parametrize(
        "content",
        [
            "",
            "no box at all, salvaged letter C",
            r"cut off mid-box \boxed{",  # opened but never closed -> truncated
            r"\boxed{unfinished nested {1",  # unbalanced -> not closed
        ],
    )
    def test_missing_or_unclosed_box_not_detected(self, content):
        assert _has_real_boxed(content) is False


class TestIsCompletedResponse:
    def test_box_and_short_is_completed(self):
        assert SelfConsistency._is_completed_response(_completed("A"), cap=8000)

    def test_box_but_over_cap_is_not_completed(self):
        # False completion: real box present but content over-runs the cap.
        r = _truncated_with_box("B")
        assert _has_real_boxed(r["content"]) is True
        assert not SelfConsistency._is_completed_response(r, cap=8000)

    def test_no_box_is_not_completed(self):
        assert not SelfConsistency._is_completed_response(_truncated_no_box("B"), 8000)


# --------------------------------------------------------------------------- #
# env resolvers
# --------------------------------------------------------------------------- #
class TestEnvResolvers:
    def test_completion_select_default_off(self, monkeypatch):
        monkeypatch.delenv("ITS_SC_COMPLETION_SELECT", raising=False)
        assert _resolve_completion_select() is False

    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on", " On "])
    def test_completion_select_truthy(self, monkeypatch, val):
        monkeypatch.setenv("ITS_SC_COMPLETION_SELECT", val)
        assert _resolve_completion_select() is True

    @pytest.mark.parametrize("val", ["0", "false", "", "nope"])
    def test_completion_select_falsy(self, monkeypatch, val):
        monkeypatch.setenv("ITS_SC_COMPLETION_SELECT", val)
        assert _resolve_completion_select() is False

    def test_log_completion_follows_select(self, monkeypatch):
        monkeypatch.delenv("ITS_SC_LOG_COMPLETION", raising=False)
        monkeypatch.setenv("ITS_SC_COMPLETION_SELECT", "1")
        assert _resolve_log_completion() is True

    def test_log_completion_dedicated_gate(self, monkeypatch):
        monkeypatch.delenv("ITS_SC_COMPLETION_SELECT", raising=False)
        monkeypatch.setenv("ITS_SC_LOG_COMPLETION", "1")
        assert _resolve_log_completion() is True

    def test_log_completion_default_off(self, monkeypatch):
        monkeypatch.delenv("ITS_SC_COMPLETION_SELECT", raising=False)
        monkeypatch.delenv("ITS_SC_LOG_COMPLETION", raising=False)
        assert _resolve_log_completion() is False

    def test_trunc_cap_default(self, monkeypatch):
        monkeypatch.delenv("ITS_SC_TRUNC_CAP_CHARS", raising=False)
        assert _resolve_trunc_cap() == _DEFAULT_TRUNC_CAP_CHARS

    @pytest.mark.parametrize("bad", ["not-an-int", "0", "-5", ""])
    def test_trunc_cap_malformed_falls_back(self, monkeypatch, bad):
        monkeypatch.setenv("ITS_SC_TRUNC_CAP_CHARS", bad)
        assert _resolve_trunc_cap() == _DEFAULT_TRUNC_CAP_CHARS

    def test_trunc_cap_override(self, monkeypatch):
        monkeypatch.setenv("ITS_SC_TRUNC_CAP_CHARS", "1234")
        assert _resolve_trunc_cap() == 1234


# --------------------------------------------------------------------------- #
# guarantee (c): flag OFF -> byte-identical projection (both projection paths)
# --------------------------------------------------------------------------- #
class TestFlagOffByteIdentical:
    def _projections_equal(self, sc, responses):
        return sc._project_responses(responses)

    def test_flag_off_flat_unchanged(self, monkeypatch):
        monkeypatch.delenv("ITS_SC_COMPLETION_SELECT", raising=False)
        sc = SelfConsistency()
        responses = [_completed("A"), _truncated_no_box("B"), _truncated_no_box("B")]
        idx, proj = sc._project_responses(responses)
        # No filtering: all three answer-bearing content responses remain eligible.
        assert idx == [0, 1, 2]
        assert len(proj) == 3

    def test_flag_off_tuple_unchanged(self, monkeypatch):
        monkeypatch.delenv("ITS_SC_COMPLETION_SELECT", raising=False)
        sc = SelfConsistency(consistency_space_projection_func=_BOXED_PROJ)
        responses = [_completed("A"), _truncated_with_box("B")]
        idx, proj = sc._project_responses(responses)
        assert idx == [0, 1]
        assert proj == [("A",), ("B",)]


# --------------------------------------------------------------------------- #
# guarantee (a): all-complete -> subset == full set -> byte-identical
# --------------------------------------------------------------------------- #
class TestAllCompleteByteIdentical:
    def test_all_complete_flag_on_equals_flag_off_flat(self, monkeypatch):
        responses = [_completed("A"), _completed("A"), _completed("B")]

        monkeypatch.delenv("ITS_SC_COMPLETION_SELECT", raising=False)
        off = SelfConsistency()._project_responses(responses)

        monkeypatch.setenv("ITS_SC_COMPLETION_SELECT", "1")
        on = SelfConsistency()._project_responses(responses)

        assert on == off  # no filtering when every sample is completed

    def test_all_complete_flag_on_equals_flag_off_tuple(self, monkeypatch):
        responses = [_completed("A"), _completed("A"), _completed("B")]

        def make_sc():
            return SelfConsistency(consistency_space_projection_func=_BOXED_PROJ)

        monkeypatch.delenv("ITS_SC_COMPLETION_SELECT", raising=False)
        off = make_sc()._project_responses(responses)

        monkeypatch.setenv("ITS_SC_COMPLETION_SELECT", "1")
        on = make_sc()._project_responses(responses)

        assert on == off


# --------------------------------------------------------------------------- #
# guarantee (b): all-truncate -> empty completed subset -> full set retained
# --------------------------------------------------------------------------- #
class TestAllTruncateFallsBackToFull:
    def test_all_truncate_flag_on_keeps_full_set_flat(self, monkeypatch):
        monkeypatch.setenv("ITS_SC_COMPLETION_SELECT", "1")
        sc = SelfConsistency()
        responses = [_truncated_no_box("A"), _truncated_no_box("B")]
        idx, proj = sc._project_responses(responses)
        # No completed sample -> proper-subset condition false -> full set kept.
        assert idx == [0, 1]
        assert len(proj) == 2

    def test_all_truncate_with_box_over_cap_keeps_full_set_tuple(self, monkeypatch):
        monkeypatch.setenv("ITS_SC_COMPLETION_SELECT", "1")
        sc = SelfConsistency(consistency_space_projection_func=_BOXED_PROJ)
        responses = [_truncated_with_box("A"), _truncated_with_box("B")]
        idx, proj = sc._project_responses(responses)
        assert idx == [0, 1]
        assert proj == [("A",), ("B",)]


# --------------------------------------------------------------------------- #
# mixed proper subset -> restrict to completed subset (the H1 mechanism)
# --------------------------------------------------------------------------- #
class TestMixedProperSubsetFilters:
    def test_flat_restricts_to_completed_and_maps_index(self, monkeypatch):
        monkeypatch.setenv("ITS_SC_COMPLETION_SELECT", "1")
        sc = SelfConsistency()
        # 1 completed ("A") vs 3 truncated (identical "B ..." essays).
        responses = [
            _completed("A"),
            _truncated_no_box("B"),
            _truncated_no_box("B"),
            _truncated_no_box("B"),
        ]
        idx, _proj = sc._project_responses(responses)
        assert idx == [0]  # only the completed response survives

        # End-to-end: without the filter the 3 truncated "B" essays would win the
        # plurality; with it, the finished "A" carries the vote.
        result = sc._process_responses(responses, return_response_only=False)
        assert result.selected_index == 0
        assert result.the_one is responses[0]

    def test_tuple_length_guard_and_index_mapping(self, monkeypatch):
        monkeypatch.setenv("ITS_SC_COMPLETION_SELECT", "1")
        sc = SelfConsistency(consistency_space_projection_func=_BOXED_PROJ)
        # index 1 is completed; indices 0 and 2 carry a real box but over-run the
        # cap -> not completed. Proper subset -> restrict to [1], mapping the
        # tuple projection index back to the original response position.
        responses = [
            _truncated_with_box("B"),
            _completed("A"),
            _truncated_with_box("B"),
        ]
        idx, proj = sc._project_responses(responses)
        assert idx == [1]
        assert proj == [("A",)]

        result = sc._process_responses(responses, return_response_only=False)
        assert result.selected_index == 1
        assert result.the_one is responses[1]

    def test_custom_cap_narrows_completed_subset(self, monkeypatch):
        monkeypatch.setenv("ITS_SC_COMPLETION_SELECT", "1")
        # With a tiny cap even the short "completed" response is over-cap -> no
        # completed sample -> full set kept (never-zero-candidate).
        monkeypatch.setenv("ITS_SC_TRUNC_CAP_CHARS", "5")
        sc = SelfConsistency()
        responses = [_completed("A"), _truncated_no_box("B")]
        idx, _ = sc._project_responses(responses)
        assert idx == [0, 1]


# --------------------------------------------------------------------------- #
# H2 telemetry: one structured record with the correct partition counts
# --------------------------------------------------------------------------- #
class TestCompletionStatusLogging:
    def _extract_record(self, caplog):
        recs = [
            r.getMessage()
            for r in caplog.records
            if r.getMessage().startswith("sc_completion_status ")
        ]
        assert len(recs) == 1, f"expected exactly one record, got {len(recs)}"
        return json.loads(recs[0][len("sc_completion_status ") :])

    def test_no_log_when_gates_off(self, monkeypatch, caplog):
        monkeypatch.delenv("ITS_SC_COMPLETION_SELECT", raising=False)
        monkeypatch.delenv("ITS_SC_LOG_COMPLETION", raising=False)
        sc = SelfConsistency()
        with caplog.at_level(logging.INFO):
            sc._process_responses([_completed("A")], return_response_only=False)
        assert not [
            r for r in caplog.records if "sc_completion_status" in r.getMessage()
        ]

    def test_record_counts_mixed(self, monkeypatch, caplog):
        # Dedicated log gate ON, selection filter effectively OFF for this check.
        monkeypatch.delenv("ITS_SC_COMPLETION_SELECT", raising=False)
        monkeypatch.setenv("ITS_SC_LOG_COMPLETION", "1")
        sc = SelfConsistency()
        responses = [
            _completed("A"),
            _truncated_no_box("B"),
            _truncated_with_box("C"),
        ]
        with caplog.at_level(logging.INFO):
            result = sc._process_responses(responses, return_response_only=False)
        rec = self._extract_record(caplog)
        assert rec["n_samples"] == 3
        assert rec["n_completed"] == 1
        assert rec["n_truncated"] == 2
        assert rec["n_empty_projection"] == 0
        assert rec["selected_index"] == result.selected_index
        assert len(rec["per_sample"]) == 3
        assert rec["per_sample"][0]["has_box"] is True
        assert rec["per_sample"][1]["has_box"] is False
        assert rec["per_sample"][2]["has_box"] is True
        assert rec["cap_chars"] == _DEFAULT_TRUNC_CAP_CHARS

    def test_selected_completed_flag(self, monkeypatch, caplog):
        monkeypatch.setenv("ITS_SC_COMPLETION_SELECT", "1")
        sc = SelfConsistency()
        responses = [
            _completed("A"),
            _truncated_no_box("B"),
            _truncated_no_box("B"),
        ]
        with caplog.at_level(logging.INFO):
            sc._process_responses(responses, return_response_only=False)
        rec = self._extract_record(caplog)
        # Filter restricts to the completed sample -> selected sample IS completed.
        assert rec["selected_completed"] is True
        assert rec["selected_index"] == 0
