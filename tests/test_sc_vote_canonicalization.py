"""Tests for formatting-invariant vote-key canonicalization (H1-c3).

These cover the pure ``_canonicalize_vote_key`` helper, the ``vote_keys``
grouping path threaded through the self-consistency selectors, and an
end-to-end ``SelfConsistency._process_responses`` check that formatting-variant
answers merge into one vote group while the full selected response is preserved.

New tests only -- no existing tests are modified.
"""

import pytest

from its_hub.core.algorithms._sc_voting import (
    _select_hierarchical_most_common_or_random,
    _select_most_common_or_random,
)
from its_hub.core.algorithms.self_consistency import SelfConsistency
from its_hub.core.utils import _canonicalize_vote_key


class TestCanonicalizeVoteKey:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            # LaTeX formatting wrappers strip to the bare option letter.
            (r"\text{C}", "C"),
            (r"\mathrm{C}", "C"),
            (r"\mathbf{C}", "C"),
            (r"\textbf{C}", "C"),
            (r"\text{\mathbf{C}}", "C"),  # nested unwrap
            # Option-letter decorations normalize to a bare uppercase letter.
            ("(C)", "C"),
            ("C.", "C"),
            ("C)", "C"),
            ("[C]", "C"),
            ("c", "C"),  # case collapse only for the isolated option letter
            (" C ", "C"),
            ("$C$", "C"),
            (r"$\text{C}$", "C"),
        ],
    )
    def test_option_letter_variants_collapse(self, raw, expected):
        assert _canonicalize_vote_key(raw) == expected

    def test_all_option_letter_forms_agree(self):
        forms = [r"\text{C}", "(C)", "C.", "C)", "C", "c", " C "]
        assert len({_canonicalize_vote_key(f) for f in forms}) == 1

    @pytest.mark.parametrize(
        "a,b",
        [
            ("1/2", "0.5"),  # distinct numeric forms must NOT merge
            ("x=2", "2"),
            ("42", "24"),
            ("{1,2}", "1,2"),  # structured content: brace guard keeps them apart
            ("A", "B"),  # different option letters stay distinct
        ],
    )
    def test_distinct_answers_stay_distinct(self, a, b):
        assert _canonicalize_vote_key(a) != _canonicalize_vote_key(b)

    @pytest.mark.parametrize(
        "raw",
        ["1/2", "0.5", "x = 2", "42", "\\frac{1}{2}", "1,2", "{1,2}"],
    )
    def test_free_form_content_is_not_rewritten(self, raw):
        # Free-form numeric/expression content is left as-is (only stripped).
        assert _canonicalize_vote_key(raw) == raw.strip()

    def test_idempotent(self):
        for raw in [r"\text{C}", "(C)", "1/2", "{1,2}", "answer: 42"]:
            once = _canonicalize_vote_key(raw)
            assert _canonicalize_vote_key(once) == once

    def test_non_string_passthrough(self):
        assert _canonicalize_vote_key(None) is None
        assert _canonicalize_vote_key(42) == 42

    def test_tuple_canonicalized_elementwise(self):
        assert _canonicalize_vote_key((r"\text{C}", "1/2")) == ("C", "1/2")
        assert _canonicalize_vote_key(("algebra", None)) == ("algebra", None)

    def test_never_reduced_to_empty(self):
        # Wrapper stripping that would empty a non-empty answer keeps the raw.
        assert _canonicalize_vote_key("$$") == "$$"
        assert _canonicalize_vote_key("") == ""


class TestSelectorsWithVoteKeys:
    def test_flat_grouping_merges_variants(self):
        raw = [r"\text{C}", "C", "A", "B"]
        keys = [_canonicalize_vote_key(x) for x in raw]
        counts, idx = _select_most_common_or_random(raw, vote_keys=keys)
        assert counts["C"] == 2
        assert raw[idx] in {r"\text{C}", "C"}

    def test_flat_no_vote_keys_is_unchanged(self):
        raw = ["a", "a", "b"]
        counts, idx = _select_most_common_or_random(raw)
        assert counts == {"a": 2, "b": 1}
        assert raw[idx] == "a"

    def test_hierarchical_grouping_merges_variants(self):
        raw = [(r"\text{C}",), ("C",), ("A",)]
        keys = [_canonicalize_vote_key(x) for x in raw]
        counts, idx = _select_hierarchical_most_common_or_random(raw, vote_keys=keys)
        assert counts[("C",)] == 2
        assert raw[idx] in {(r"\text{C}",), ("C",)}


class TestProcessResponsesCanonicalization:
    def test_formatting_variants_form_plurality(self):
        # Two formatting-variant "C" answers vs one "A" and one "B": without
        # canonicalization this is a 4-way (or 1-1-1-1) split; with it, "C" wins.
        sc = SelfConsistency()
        responses = [
            {"content": r"\text{C}"},
            {"content": "C"},
            {"content": "A"},
            {"content": "B"},
        ]
        result = sc._process_responses(responses, return_response_only=False)
        assert result.response_counts["C"] == 2
        # the_one must be a full, unmodified original response.
        assert result.the_one in responses
        assert result.the_one["content"] in {r"\text{C}", "C"}

    def test_distinct_numeric_answers_not_merged(self):
        # 1/2 and 0.5 must remain separate vote groups (no false plurality).
        sc = SelfConsistency()
        responses = [
            {"content": "1/2"},
            {"content": "0.5"},
            {"content": "0.5"},
        ]
        result = sc._process_responses(responses, return_response_only=False)
        assert result.response_counts["0.5"] == 2
        assert result.response_counts["1/2"] == 1
        assert result.the_one["content"] == "0.5"
