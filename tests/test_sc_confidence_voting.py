"""Tests for full confidence-weighted voting (H5, CISC arXiv:2502.06233).

Covers the ``ITS_SC_VOTE=confidence`` upgrade from tie-break-only confidence to
full confidence-weighted voting in the self-consistency family:

  (a) the default/plurality path is byte-unchanged vs current selection;
  (b) confidence mode picks the higher-aggregate-confidence group where it
      legitimately differs from plurality;
  (c) the negative-logprob small-group-bias pitfall is ABSENT -- a large group
      of moderately-confident answers beats a single hyper-confident outlier;
  (d) ``None`` tiebreak_scores => confidence mode falls back to plurality;
  (e) determinism -- confidence decisions never depend on ``random``;
  (f) both the flat and hierarchical selectors honor the flag, end to end.

New tests only -- no existing tests are modified.
"""

import math

import pytest

from its_hub.core.algorithms._sc_voting import (
    _confidence_weight,
    _has_confidence_scores,
    _resolve_vote_mode,
    _select_by_confidence_weight,
    _select_hierarchical_most_common_or_random,
    _select_most_common_or_random,
)
from its_hub.core.algorithms.self_consistency import SelfConsistency

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


# ---------------------------------------------------------------------------
# _resolve_vote_mode / helpers
# ---------------------------------------------------------------------------


class TestResolveVoteMode:
    def test_unset_defaults_to_plurality(self, monkeypatch):
        monkeypatch.delenv("ITS_SC_VOTE", raising=False)
        assert _resolve_vote_mode() == "plurality"

    def test_confidence_value(self, monkeypatch):
        monkeypatch.setenv("ITS_SC_VOTE", "confidence")
        assert _resolve_vote_mode() == "confidence"

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("ITS_SC_VOTE", "  CONFIDENCE  ")
        assert _resolve_vote_mode() == "confidence"

    @pytest.mark.parametrize("value", ["plurality", "garbage", "", "conf", "1"])
    def test_unrecognized_values_are_plurality(self, monkeypatch, value):
        monkeypatch.setenv("ITS_SC_VOTE", value)
        assert _resolve_vote_mode() == "plurality"


class TestConfidenceWeight:
    def test_is_positive_and_bounded(self):
        # mean logprob is <= 0 => weight in (0, 1].
        assert _confidence_weight(0.0) == pytest.approx(1.0)
        assert 0.0 < _confidence_weight(-0.7) < 1.0
        assert _confidence_weight(-5.0) == pytest.approx(math.exp(-5.0))

    def test_monotonic_in_confidence(self):
        assert _confidence_weight(-0.1) > _confidence_weight(-1.0)


class TestHasConfidenceScores:
    def test_none(self):
        assert not _has_confidence_scores(None)

    def test_all_none(self):
        assert not _has_confidence_scores([None, None])

    def test_some_present(self):
        assert _has_confidence_scores([None, -0.5])


# ---------------------------------------------------------------------------
# (a) Default/plurality path is byte-unchanged
# ---------------------------------------------------------------------------


class TestPluralityPathUnchanged:
    @pytest.mark.parametrize(
        "keys,scores",
        [
            (["a", "b", "a", "c", "a"], [-1.0, -2.0, -1.5, -0.5, -0.9]),
            (["a", "b", "a", "b", "c"], [-0.1, -0.2, -0.3, -0.4, -0.5]),
            (["a", "b", "c", "d"], None),
        ],
    )
    def test_default_matches_explicit_plurality(self, keys, scores):
        # Omitting vote_mode == passing vote_mode="plurality" (default arg).
        counts_default, idx_default = _select_most_common_or_random(
            keys, scores, vote_keys=keys
        )
        counts_plur, idx_plur = _select_most_common_or_random(
            keys, scores, vote_keys=keys, vote_mode="plurality"
        )
        assert counts_default == counts_plur
        # For a clear single-group majority the selection is deterministic and
        # must match; the ["a","b","c","d"] all-tie case is excluded from the
        # index equality check because plurality breaks it randomly.
        if keys != ["a", "b", "c", "d"]:
            assert keys[idx_default] == keys[idx_plur]

    def test_clear_majority_plurality_selects_that_group(self):
        keys = ["a", "b", "a", "c", "a"]
        counts, idx = _select_most_common_or_random(keys, None, vote_keys=keys)
        assert counts == {"a": 3, "b": 1, "c": 1}
        assert keys[idx] == "a"

    def test_process_responses_default_is_plurality(self, monkeypatch):
        # With the flag unset, a clear frequency majority wins even when a
        # minority answer is far more confident.
        monkeypatch.delenv("ITS_SC_VOTE", raising=False)
        sc = SelfConsistency()
        responses = [
            _response("A", -5.0),
            _response("A", -5.0),
            _response("B", -0.01),
        ]
        result = sc._process_responses(responses, return_response_only=False)
        assert result.the_one["content"] == "A"
        assert result.response_counts["A"] == 2


# ---------------------------------------------------------------------------
# (b) Confidence mode differs from plurality when confidence warrants it
# ---------------------------------------------------------------------------


class TestConfidenceDiffersFromPlurality:
    def test_flat_high_confidence_minority_wins(self):
        keys = ["A", "A", "B"]
        scores = [-5.0, -5.0, -0.01]  # A weak (2x), B very strong (1x)
        # Plurality picks A.
        _, idx_plur = _select_most_common_or_random(
            keys, scores, vote_keys=keys, vote_mode="plurality"
        )
        assert keys[idx_plur] == "A"
        # Confidence picks B: exp(-0.01) ~ 0.99 > 2 * exp(-5) ~ 0.013.
        _, idx_conf = _select_most_common_or_random(
            keys, scores, vote_keys=keys, vote_mode="confidence"
        )
        assert keys[idx_conf] == "B"

    def test_select_by_confidence_weight_directly(self):
        keys = ["A", "A", "B"]
        scores = [-5.0, -5.0, -0.01]
        assert _select_by_confidence_weight(keys, scores) == 2

    def test_returned_counts_stay_frequency_based(self):
        keys = ["A", "A", "B"]
        scores = [-5.0, -5.0, -0.01]
        counts, _ = _select_most_common_or_random(
            keys, scores, vote_keys=keys, vote_mode="confidence"
        )
        # Reporting contract unchanged: counts are raw frequencies, not weights.
        assert counts == {"A": 2, "B": 1}

    def test_process_responses_confidence_flag(self, monkeypatch):
        monkeypatch.setenv("ITS_SC_VOTE", "confidence")
        sc = SelfConsistency()
        responses = [
            _response("A", -5.0),
            _response("A", -5.0),
            _response("B", -0.01),
        ]
        result = sc._process_responses(responses, return_response_only=False)
        assert result.the_one["content"] == "B"
        # Frequency counts still reported.
        assert result.response_counts["A"] == 2


# ---------------------------------------------------------------------------
# (c) The negative-logprob small-group-bias pitfall is ABSENT
# ---------------------------------------------------------------------------


class TestSmallGroupBiasAbsent:
    def test_large_moderate_group_beats_single_confident_outlier(self):
        # Three moderately-confident "A" votes vs one hyper-confident "B".
        keys = ["A", "A", "A", "B"]
        scores = [-0.7, -0.7, -0.7, -0.001]
        # Sanity: naive SUM of raw NEGATIVE logprobs would pick B (the pitfall):
        #   sum(A) = -2.1  <  sum(B) = -0.001  => argmax raw-sum = B (WRONG).
        raw_sum_a = -0.7 * 3
        raw_sum_b = -0.001
        assert raw_sum_b > raw_sum_a  # documents the trap we must NOT fall into
        # Correct exp-weighted aggregation picks A:
        #   3 * exp(-0.7) ~ 1.49  >  exp(-0.001) ~ 0.999.
        idx = _select_by_confidence_weight(keys, scores)
        assert keys[idx] == "A"

        _, idx2 = _select_most_common_or_random(
            keys, scores, vote_keys=keys, vote_mode="confidence"
        )
        assert keys[idx2] == "A"

    def test_process_responses_no_small_group_bias(self, monkeypatch):
        monkeypatch.setenv("ITS_SC_VOTE", "confidence")
        sc = SelfConsistency()
        responses = [
            _response("A", -0.7),
            _response("A", -0.7),
            _response("A", -0.7),
            _response("B", -0.001),
        ]
        result = sc._process_responses(responses, return_response_only=False)
        assert result.the_one["content"] == "A"


# ---------------------------------------------------------------------------
# (d) None / absent scores => confidence mode falls back to plurality
# ---------------------------------------------------------------------------


class TestGracefulFallback:
    def test_none_scores_falls_back_to_plurality(self):
        keys = ["A", "A", "B"]
        counts, idx = _select_most_common_or_random(
            keys, None, vote_keys=keys, vote_mode="confidence"
        )
        assert counts == {"A": 2, "B": 1}
        assert keys[idx] == "A"  # plurality winner

    def test_all_none_scores_falls_back_to_plurality(self):
        keys = ["A", "A", "B"]
        _, idx = _select_most_common_or_random(
            keys, [None, None, None], vote_keys=keys, vote_mode="confidence"
        )
        assert keys[idx] == "A"

    def test_process_responses_confidence_without_logprobs(self, monkeypatch):
        # Confidence flag set but no logprobs anywhere => plurality behavior.
        monkeypatch.setenv("ITS_SC_VOTE", "confidence")
        sc = SelfConsistency()
        responses = [
            _response("A", None),
            _response("A", None),
            _response("B", None),
        ]
        result = sc._process_responses(responses, return_response_only=False)
        assert result.the_one["content"] == "A"

    def test_partial_none_scores_contribute_zero(self):
        # A candidate lacking a score still belongs to its group but contributes
        # zero weight; it must not error and must not swing the vote.
        keys = ["A", "A", "B"]
        scores = [-0.7, None, -0.001]
        # Group A weight = exp(-0.7) ~ 0.497 (one scored member) > B ~ 0.999?
        # No: B ~ 0.999 > 0.497, so B wins here -- and that is correct given A's
        # second member carries no confidence evidence.
        idx = _select_by_confidence_weight(keys, scores)
        assert keys[idx] == "B"


# ---------------------------------------------------------------------------
# (e) Determinism -- confidence decisions never touch ``random``
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_no_random_dependence(self, monkeypatch):
        # Poison random.choice: the confidence path must never call it.
        import its_hub.core.algorithms._sc_voting as voting

        def _boom(*_args, **_kwargs):
            raise AssertionError("random.choice must not be called in confidence mode")

        monkeypatch.setattr(voting.random, "choice", _boom)
        keys = ["A", "A", "B"]
        scores = [-5.0, -5.0, -0.01]
        _, idx = _select_most_common_or_random(
            keys, scores, vote_keys=keys, vote_mode="confidence"
        )
        assert keys[idx] == "B"

    def test_group_weight_tie_is_deterministic(self):
        # Two groups with identical aggregate weight -> lowest-first-index group.
        keys = ["A", "B"]
        scores = [-0.5, -0.5]
        results = {_select_by_confidence_weight(keys, scores) for _ in range(50)}
        assert results == {0}  # always A (first-seen group), never random

    def test_within_group_member_is_deterministic(self):
        # Winning group has two members; the higher-logprob, lowest-index member
        # is chosen deterministically.
        keys = ["A", "A", "B"]
        scores = [-2.0, -0.5, -9.0]  # A wins; within A, index 1 is more confident
        results = {_select_by_confidence_weight(keys, scores) for _ in range(50)}
        assert results == {1}


# ---------------------------------------------------------------------------
# (f) Both flat and hierarchical selectors honor the flag
# ---------------------------------------------------------------------------


class TestHierarchicalHonorsFlag:
    def test_hierarchical_default_is_plurality(self):
        keys = [("A",), ("A",), ("B",)]
        _, idx = _select_hierarchical_most_common_or_random(
            keys, [-5.0, -5.0, -0.01], vote_keys=keys, vote_mode="plurality"
        )
        assert keys[idx] == ("A",)

    def test_hierarchical_confidence_high_conf_minority_wins(self):
        keys = [("A",), ("A",), ("B",)]
        _, idx = _select_hierarchical_most_common_or_random(
            keys, [-5.0, -5.0, -0.01], vote_keys=keys, vote_mode="confidence"
        )
        assert keys[idx] == ("B",)

    def test_hierarchical_large_moderate_group_beats_outlier(self):
        keys = [("A", "1"), ("A", "1"), ("A", "1"), ("B", "2")]
        scores = [-0.7, -0.7, -0.7, -0.001]
        _, idx = _select_hierarchical_most_common_or_random(
            keys, scores, vote_keys=keys, vote_mode="confidence"
        )
        assert keys[idx] == ("A", "1")

    def test_hierarchical_confidence_falls_back_without_scores(self):
        keys = [("A",), ("A",), ("B",)]
        _, idx = _select_hierarchical_most_common_or_random(
            keys, None, vote_keys=keys, vote_mode="confidence"
        )
        assert keys[idx] == ("A",)

    def test_hierarchical_default_matches_omitted_vote_mode(self):
        keys = [("a", "1"), ("a", "2"), ("b", "1"), ("c", "1")]
        counts_default, _ = _select_hierarchical_most_common_or_random(keys)
        counts_plur, _ = _select_hierarchical_most_common_or_random(
            keys, vote_mode="plurality"
        )
        assert counts_default == counts_plur

    def test_process_responses_hierarchical_confidence(self, monkeypatch):
        # Route through the hierarchical selector via a tuple projection.
        monkeypatch.setenv("ITS_SC_VOTE", "confidence")

        def proj(content: str) -> tuple:
            return (content,)

        sc = SelfConsistency(consistency_space_projection_func=proj)
        responses = [
            _response("A", -5.0),
            _response("A", -5.0),
            _response("B", -0.01),
        ]
        result = sc._process_responses(responses, return_response_only=False)
        assert result.the_one["content"] == "B"
