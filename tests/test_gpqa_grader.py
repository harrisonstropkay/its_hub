"""Unit tests for the GPQA-Diamond multiple-choice grader in benchmarking/benchmark.py.

These tests are hermetic: they never hit the network or load the real dataset.
The gold letter is mocked directly and only the pure-Python grader / extraction
logic is exercised.
"""

import importlib.util
import os

import pytest

# Load benchmarking/benchmark.py by path (it is not an importable package).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BENCH_PATH = os.path.join(_REPO_ROOT, "benchmarking", "benchmark.py")
_spec = importlib.util.spec_from_file_location("its_benchmark", _BENCH_PATH)
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)

GPQA = benchmark.BenchmarkDataset.GPQA_DIAMOND
grade_response = benchmark.grade_response
_extract_choice_letter = benchmark._extract_choice_letter
_normalize_gpqa_row = benchmark._normalize_gpqa_row


def test_correct_boxed_letter_is_true():
    assert grade_response(GPQA, "C", r"After reasoning, \boxed{C}") is True


def test_wrong_boxed_letter_is_false():
    assert grade_response(GPQA, "C", r"After reasoning, \boxed{A}") is False


@pytest.mark.parametrize(
    "response,gold,expected",
    [
        (r"\boxed{B}", "B", True),
        (r"\boxed{D}", "B", False),
        ("The answer is D.", "D", True),
        ("answer: (B)", "B", True),
        ("answer: (b)", "B", True),  # case-insensitive extraction
        ("Final answer is C", "C", True),
        ("Comparing options, the correct choice is A", "A", True),
        ("hence the option (D) is correct.", "D", True),
        ("The answer is D.", "A", False),
    ],
)
def test_extraction_format_variants(response, gold, expected):
    assert grade_response(GPQA, gold, response) is expected


def test_no_letter_in_response_is_false():
    assert grade_response(GPQA, "A", "I am not sure about this one.") is False


def test_empty_response_is_false():
    assert grade_response(GPQA, "A", "") is False


def test_gold_wrapped_in_boxed_is_tolerated():
    # gold arriving wrapped (e.g. "\boxed{B}") should still match a plain "B".
    assert grade_response(GPQA, r"\boxed{B}", "answer: B") is True


def test_extract_choice_letter_returns_none_when_absent():
    assert _extract_choice_letter("no idea") is None
    assert _extract_choice_letter("") is None


def test_extract_choice_letter_uppercases():
    assert _extract_choice_letter(r"\boxed{c}") == "C"


def test_gpqa_grader_does_not_import_math_verify():
    # The GPQA path must never route through math_verify (choices are letters,
    # not math expressions). math_verify is not installed in the base env, so if
    # the GPQA branch tried to import it this call would raise.
    assert grade_response(GPQA, "B", r"\boxed{B}") is True


def _mock_gpqa_row(record_id="rec-1"):
    return {
        "Question": "What is the capital of France?",
        "Correct Answer": "Paris",
        "Incorrect Answer 1": "London",
        "Incorrect Answer 2": "Berlin",
        "Incorrect Answer 3": "Madrid",
        "Record ID": record_id,
    }


def test_normalize_gpqa_row_letter_points_to_correct_answer():
    out = _normalize_gpqa_row(_mock_gpqa_row(), 0)
    assert set(out) == {"problem", "answer", "unique_id"}
    assert out["answer"] in {"A", "B", "C", "D"}
    # the labeled option for the returned letter must contain the correct answer
    option_lines = [
        ln for ln in out["problem"].splitlines() if ln[:2] in ("A)", "B)", "C)", "D)")
    ]
    assert len(option_lines) == 4
    correct_line = next(ln for ln in option_lines if ln.startswith(out["answer"] + ")"))
    assert "Paris" in correct_line


def test_normalize_gpqa_row_is_deterministic():
    a = _normalize_gpqa_row(_mock_gpqa_row("rec-x"), 0)
    b = _normalize_gpqa_row(_mock_gpqa_row("rec-x"), 0)
    assert a == b


def test_normalize_gpqa_row_shuffle_varies_by_item():
    # Different unique_ids should not all place the correct answer identically.
    letters = {
        _normalize_gpqa_row(_mock_gpqa_row(f"rec-{i}"), i)["answer"] for i in range(20)
    }
    assert len(letters) > 1


def test_normalize_gpqa_row_falls_back_to_index_uid():
    row = _mock_gpqa_row()
    del row["Record ID"]
    out = _normalize_gpqa_row(row, 7)
    assert out["unique_id"] == "7"
