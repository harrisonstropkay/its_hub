"""Unit tests for the H4.9 GPQA measurement-correctness fix (exp_id=12).

Two coordinated changes are covered here, both hermetic (no network, no dataset):

  (a) Per-benchmark self-consistency vote-key projection. GPQA-Diamond is
      multiple-choice, so its projection parses the model's OWN chosen A-D letter
      and feeds that as the vote key (voting over {A,B,C,D}); MATH500 / AIME-2024
      keep the existing numeric ``_extract_boxed`` path completely unchanged. This
      fixes the degenerate case where Qwen2.5-Math never emits ``\\boxed{letter}``
      on MCQ items, so every draw projected to ``""`` and collapsed into one empty
      vote group (no self-consistency voting occurred).

  (b) Reproducible plurality: ``ITS_SEED`` is threaded into Python's global
      ``random`` in the benchmark subprocess, so the ``random.choice`` plurality
      tie-break is deterministic across repeats under a fixed seed.

New tests only — no existing tests are modified.
"""

import importlib.util
import os
import random

import pytest

from its_hub.core.algorithms._sc_voting import _select_most_common_or_random
from its_hub.core.algorithms.self_consistency import SelfConsistency

# Load benchmarking/benchmark.py by path (it is not an importable package),
# mirroring tests/test_gpqa_grader.py.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BENCH_PATH = os.path.join(_REPO_ROOT, "benchmarking", "benchmark.py")
_spec = importlib.util.spec_from_file_location("its_benchmark_mcq", _BENCH_PATH)
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)

GPQA = benchmark.BenchmarkDataset.GPQA_DIAMOND
MATH500 = benchmark.BenchmarkDataset.MATH500
AIME = benchmark.BenchmarkDataset.AIME_2024

_extract_boxed = benchmark._extract_boxed
_extract_choice_letter_or_empty = benchmark._extract_choice_letter_or_empty
_projection_func_for = benchmark._projection_func_for
_seed_global_random_from_env = benchmark._seed_global_random_from_env
_normalize_gpqa_row = benchmark._normalize_gpqa_row


# --- (a) GPQA MCQ extraction yields non-empty vote keys ---------------------


class TestGpqaVoteKeyProjection:
    @pytest.mark.parametrize(
        "response,expected",
        [
            (r"After reasoning, Answer: B\n\boxed{B}", "B"),
            (r"I pick \boxed{C}", "C"),
            ("The answer is D.", "D"),
            ("answer: (a)", "A"),  # case-insensitive -> uppercased
            ("hence (D).", "D"),
            ("Comparing options, the correct choice is A", "A"),
        ],
    )
    def test_representative_outputs_yield_a_letter(self, response, expected):
        assert _extract_choice_letter_or_empty(response) == expected

    def test_unparseable_output_is_empty_string(self):
        # No A-D choice -> empty string (ineligible), never None (the projection
        # contract is str, so _project_responses can treat it as an empty key).
        assert _extract_choice_letter_or_empty("I am not sure.") == ""
        assert _extract_choice_letter_or_empty("") == ""

    def test_fixes_degenerate_all_empty_case(self):
        # Qwen2.5-Math on MCQ rarely emits \boxed{letter}: the OLD numeric
        # projection collapses these to all-empty (one degenerate vote group, no
        # voting). The NEW MCQ projection recovers real per-letter vote groups.
        outputs = [
            "After analysis, Answer: B",
            "I conclude the answer is B.",
            "The answer is A.",
            "hence (C).",
        ]
        old_keys = [_extract_boxed(o) for o in outputs]
        assert old_keys == ["", "", "", ""]  # degenerate under the numeric path
        new_keys = [_extract_choice_letter_or_empty(o) for o in outputs]
        assert new_keys == ["B", "B", "A", "C"]  # real vote groups {A,B,C}

    def test_selector_returns_mcq_extractor_for_gpqa(self):
        assert _projection_func_for(GPQA) is _extract_choice_letter_or_empty


class TestGpqaSelfConsistencyEndToEnd:
    def test_non_empty_vote_keys_produce_real_plurality(self):
        # Responses WITHOUT \boxed (the failure mode) still vote over letters.
        sc = SelfConsistency(_extract_choice_letter_or_empty)
        responses = [
            {"content": "After analysis, Answer: B"},
            {"content": "I conclude the answer is B."},
            {"content": "The answer is A."},
            {"content": "hence (C)."},
        ]
        result = sc._process_responses(responses, return_response_only=False)
        # B wins 2 vs 1 vs 1 — actual voting occurred (not a degenerate group).
        assert result.response_counts["B"] == 2
        assert result.response_counts["A"] == 1
        assert result.response_counts["C"] == 1
        assert _extract_choice_letter_or_empty(result.the_one["content"]) == "B"

    def test_all_empty_gpqa_would_be_single_group_under_numeric_path(self):
        # Contrast: the numeric projection on the same non-boxed outputs yields a
        # single degenerate "" group (one vote key) — no self-consistency.
        sc_numeric = SelfConsistency(_extract_boxed)
        responses = [
            {"content": "After analysis, Answer: B"},
            {"content": "The answer is A."},
        ]
        result = sc_numeric._process_responses(responses, return_response_only=False)
        assert list(result.response_counts.keys()) == [""]

    def test_mcq_projection_reads_model_text_not_gold(self):
        # The projection parses the model's OWN choice — even a WRONG one. If the
        # model answers A, the vote key is A regardless of any gold letter. This
        # proves the extractor never consults an answer key.
        assert _extract_choice_letter_or_empty("I am confident: Answer: A") == "A"
        # It also drives voting toward whatever the model actually picked:
        sc = SelfConsistency(_extract_choice_letter_or_empty)
        responses = [
            {"content": "Answer: A"},
            {"content": "Answer: A"},
            {"content": "Answer: B"},  # gold could be B, but A has the plurality
        ]
        result = sc._process_responses(responses, return_response_only=False)
        assert _extract_choice_letter_or_empty(result.the_one["content"]) == "A"


# --- (a) MATH500 / AIME projection is UNCHANGED -----------------------------


class TestMathAimeProjectionUnchanged:
    def test_selector_returns_extract_boxed_for_math_and_aime(self):
        assert _projection_func_for(MATH500) is _extract_boxed
        assert _projection_func_for(AIME) is _extract_boxed

    def test_selector_defaults_to_extract_boxed(self):
        # No benchmark supplied -> numeric path (backward compatible).
        assert _projection_func_for(None) is _extract_boxed

    @pytest.mark.parametrize(
        "response,expected",
        [
            (r"The answer is \boxed{42}", "42"),
            (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
            ("no box here", ""),
        ],
    )
    def test_extract_boxed_behavior_is_intact(self, response, expected):
        assert _extract_boxed(response) == expected

    def test_init_algorithm_wires_numeric_projection_for_math(self):
        alg = benchmark.init_algorithm(
            benchmark.ScalingAlgorithm.SELF_CONSISTENCY,
            "Qwen/Qwen2.5-Math-7B-Instruct",
            "rm",
            "cpu",
            "model",
            benchmark=MATH500,
        )
        assert alg.consistency_space_projection_func is _extract_boxed

    def test_init_algorithm_wires_mcq_projection_for_gpqa(self):
        alg = benchmark.init_algorithm(
            benchmark.ScalingAlgorithm.SELF_CONSISTENCY,
            "Qwen/Qwen2.5-Math-7B-Instruct",
            "rm",
            "cpu",
            "model",
            benchmark=GPQA,
        )
        assert alg.consistency_space_projection_func is _extract_choice_letter_or_empty


# --- (b) Seeded global random makes the plurality tie-break deterministic ----


class TestSeededPluralityDeterminism:
    def test_random_choice_tiebreak_is_reproducible_under_fixed_seed(self):
        # A single top group with multiple members (the GPQA-like case once real
        # letters exist) routes to random.choice among the members. Seeding the
        # global RNG makes the selected member reproducible across repeats.
        keys = ["A", "A", "B"]  # A is the single top group (2 members)

        def one_run(seed):
            random.seed(seed)
            picks = []
            for _ in range(20):
                _, idx = _select_most_common_or_random(keys, tiebreak_scores=None)
                picks.append(idx)
            return picks

        assert one_run(123) == one_run(123)  # same seed -> identical sequence

    def test_seed_env_helper_applies_its_seed(self, monkeypatch):
        monkeypatch.setenv("ITS_SEED", "777")
        assert _seed_global_random_from_env() == 777
        # After seeding, a fresh reseed reproduces the same draw sequence.
        random.seed(777)
        expected = [random.random() for _ in range(5)]
        _seed_global_random_from_env()
        assert [random.random() for _ in range(5)] == expected

    def test_seed_env_helper_noop_when_unset(self, monkeypatch):
        monkeypatch.delenv("ITS_SEED", raising=False)
        assert _seed_global_random_from_env() is None

    def test_seed_env_helper_noop_when_invalid(self, monkeypatch):
        monkeypatch.setenv("ITS_SEED", "not-an-int")
        assert _seed_global_random_from_env() is None

    def test_seeded_self_consistency_selection_is_reproducible(self, monkeypatch):
        # End-to-end: with ITS_SEED set and the global RNG seeded, the response
        # that SelfConsistency returns from a multi-member top group is stable.
        monkeypatch.setenv("ITS_SEED", "2024")
        sc = SelfConsistency(_extract_choice_letter_or_empty)
        responses = [
            {"content": "Answer: A"},
            {"content": "Answer: A"},
            {"content": "Answer: A"},
            {"content": "Answer: B"},
        ]

        def one_selection():
            _seed_global_random_from_env()
            return sc._process_responses(
                responses, return_response_only=False
            ).selected_index

        assert one_selection() == one_selection()


# --- (a) GPQA MCQ prompt instructs a parseable discrete choice ---------------


class TestGpqaPromptEmitsParseableChoiceInstruction:
    def _mock_row(self, record_id="rec-1"):
        return {
            "Question": "What is the capital of France?",
            "Correct Answer": "Paris",
            "Incorrect Answer 1": "London",
            "Incorrect Answer 2": "Berlin",
            "Incorrect Answer 3": "Madrid",
            "Record ID": record_id,
        }

    def test_prompt_asks_for_parseable_choice(self):
        out = _normalize_gpqa_row(self._mock_row(), 0)
        problem = out["problem"]
        # Instructs BOTH a parseable "Answer: X" and a \boxed{X} discrete choice.
        assert "Answer:" in problem
        assert "\\boxed" in problem
        assert out["answer"] in {"A", "B", "C", "D"}

    def test_prompt_does_not_reveal_gold_letter(self):
        # The instruction must not leak which option is correct.
        out = _normalize_gpqa_row(self._mock_row(), 0)
        problem = out["problem"]
        gold = out["answer"]
        # The gold letter must not be named in the trailing instruction sentence.
        instruction = problem.split("\n\n")[-1]
        assert f"correct option is {gold}" not in instruction
        assert f"answer is {gold}" not in instruction.lower()
