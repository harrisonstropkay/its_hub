"""Unit tests for eval_leakage_check() in eval/score.py (H4, exp_id=9).

The contamination gate is exercised entirely with a MOCKED LM and injected
items — no GPU, no network, no dataset download. The tests pin the four
behaviours required by the H4 test contract:

  (i)   verbatim reproduction        -> high rate -> passed=False (contaminated)
  (ii)  unrelated model output       -> rate 0    -> passed=True  (clean)
  (iii) seeded item sampling         -> deterministic (same seed -> same items)
  (iv)  threshold boundary           -> rate == threshold passes; above fails

It also confirms the gate is purely additive: it is absent from the scored
bundle (EVALS), carries weight 0.0, and leaves the existing accuracy dimension
and composite untouched.
"""

import importlib.util
import os

# Load eval/score.py by path ("eval" is a builtin name and not a package).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCORE_PATH = os.path.join(_REPO_ROOT, "eval", "score.py")
_spec = importlib.util.spec_from_file_location("its_eval_score_leakage", _SCORE_PATH)
score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score)

_CONTRACT_KEYS = {"name", "score", "weight", "passed", "details"}


# --- Mock LMs ---------------------------------------------------------------


class VerbatimLM:
    """A 'contaminated' model: recites the gold answer for every probe.

    The probe prompt embeds the full problem statement, so a model that has
    memorized the answer key can echo the gold answer back. We simulate that by
    looking the gold answer up from the embedded problem text.
    """

    def __init__(self, items):
        # Map problem text -> gold answer, mimicking a memorized answer key.
        self._key = {it["problem"]: it["answer"] for it in items}

    def generate(self, messages):
        prompt = messages[-1]["content"]
        for problem, answer in self._key.items():
            if problem in prompt:
                return {"role": "assistant", "content": answer}
        return {"role": "assistant", "content": "UNKNOWN"}


class UnrelatedLM:
    """A 'clean' model: never reproduces the gold answer."""

    def generate(self, messages):
        return {"role": "assistant", "content": "UNKNOWN"}


class AsyncVerbatimLM:
    """Same as VerbatimLM but via the real async agenerate_single interface."""

    def __init__(self, items):
        self._key = {it["problem"]: it["answer"] for it in items}

    async def agenerate_single(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        for problem, answer in self._key.items():
            if problem in prompt:
                return {"role": "assistant", "content": answer}
        return {"role": "assistant", "content": "UNKNOWN"}


def _items(n=10):
    return [
        {"problem": f"Problem number {i}: what is {i} + {i}?", "answer": str(2 * i), "unique_id": f"id-{i}"}
        for i in range(n)
    ]


# --- (i) verbatim reproduction -> contamination detected --------------------


def test_verbatim_reproduction_flags_contamination():
    items = _items()
    result = score.eval_leakage_check(lm=VerbatimLM(items), items=items, n=5, seed=0)
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert "contamination detected" in result["details"]


def test_verbatim_reproduction_async_interface():
    items = _items()
    result = score.eval_leakage_check(lm=AsyncVerbatimLM(items), items=items, n=5, seed=0)
    assert result["passed"] is False
    assert "contamination detected" in result["details"]


# --- (ii) unrelated output -> clean -----------------------------------------


def test_unrelated_output_passes_clean():
    items = _items()
    result = score.eval_leakage_check(lm=UnrelatedLM(), items=items, n=5, seed=0)
    assert result["passed"] is True
    assert result["score"] == 1.0
    assert "no contamination" in result["details"]
    assert "rate=0.000" in result["details"]


# --- (iii) seeded sampling is deterministic ---------------------------------


def _sampled_ids(items, seed, n=4):
    """Recover which items were probed for a given seed via a recording LM."""
    seen = []

    class RecordingLM:
        def generate(self, messages):
            prompt = messages[-1]["content"]
            for it in items:
                if it["problem"] in prompt:
                    seen.append(it["unique_id"])
            return {"role": "assistant", "content": "UNKNOWN"}

    score.eval_leakage_check(lm=RecordingLM(), items=items, n=n, seed=seed)
    return seen


def test_seeded_sampling_is_deterministic():
    items = _items(20)
    first = _sampled_ids(items, seed=123)
    second = _sampled_ids(items, seed=123)
    assert first == second
    assert len(first) == 4


def test_different_seeds_select_different_items():
    items = _items(50)
    a = _sampled_ids(items, seed=1)
    b = _sampled_ids(items, seed=2)
    # Overwhelmingly likely to differ for disjoint seeds over 50 items.
    assert a != b


def test_sampled_indices_are_sorted_and_in_range():
    items = _items(20)
    ids = _sampled_ids(items, seed=7)
    idxs = [int(x.split("-")[1]) for x in ids]
    assert idxs == sorted(idxs)
    assert all(0 <= i < 20 for i in idxs)


# --- (iv) threshold boundary behaviour --------------------------------------


class HalfMatchLM:
    """Recites the gold answer for exactly half of the items (even indices)."""

    def __init__(self, items):
        self._match = {
            it["problem"]: it["answer"] for i, it in enumerate(items) if i % 2 == 0
        }

    def generate(self, messages):
        prompt = messages[-1]["content"]
        for problem, answer in self._match.items():
            if problem in prompt:
                return {"role": "assistant", "content": answer}
        return {"role": "assistant", "content": "UNKNOWN"}


def test_rate_equal_to_threshold_passes():
    # 2 of 4 items recited -> rate 0.5; threshold 0.5 -> rate == threshold passes.
    items = _items(4)
    result = score.eval_leakage_check(
        lm=HalfMatchLM(items), items=items, n=4, seed=0, threshold=0.5
    )
    assert result["passed"] is True
    assert "rate=0.500" in result["details"]


def test_rate_above_threshold_fails():
    items = _items(4)
    result = score.eval_leakage_check(
        lm=HalfMatchLM(items), items=items, n=4, seed=0, threshold=0.49
    )
    assert result["passed"] is False
    assert "contamination detected" in result["details"]


def test_all_match_low_threshold_fails():
    items = _items(4)
    result = score.eval_leakage_check(
        lm=VerbatimLM(items), items=items, n=4, seed=0, threshold=0.0
    )
    assert result["passed"] is False


# --- return-shape / contract ------------------------------------------------


def test_result_matches_contract():
    items = _items()
    result = score.eval_leakage_check(lm=UnrelatedLM(), items=items, n=3, seed=0)
    assert set(result) == _CONTRACT_KEYS
    assert result["name"] == "leakage_check"
    assert isinstance(result["score"], float)
    assert isinstance(result["weight"], float)
    assert isinstance(result["passed"], bool)
    assert isinstance(result["details"], str) and result["details"]


def test_n_larger_than_pool_is_clamped():
    items = _items(3)
    result = score.eval_leakage_check(lm=UnrelatedLM(), items=items, n=99, seed=0)
    assert result["passed"] is True
    # Only 3 items exist, so at most 3 are probed despite n=99.
    assert "0/3 items" in result["details"]


def test_no_items_degrades_gracefully():
    result = score.eval_leakage_check(lm=UnrelatedLM(), items=[], n=5, seed=0)
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert "could not run" in result["details"]


def test_no_lm_and_no_endpoint_degrades(monkeypatch):
    monkeypatch.delenv("ITS_ENDPOINT", raising=False)
    monkeypatch.delenv("OPENAI_ENDPOINT", raising=False)
    result = score.eval_leakage_check(items=_items(), n=5, seed=0)
    assert result["passed"] is False
    assert "could not run" in result["details"]


def test_probe_prompt_never_leaks_gold_answer():
    # The recite probe must NOT hand the model the gold answer.
    item = {"problem": "What is 2 + 2?", "answer": "42UNIQUEGOLD", "unique_id": "x"}
    prompt = score._leakage_probe_prompt(item)
    assert item["problem"] in prompt
    assert item["answer"] not in prompt


# --- purely-additive guarantees ---------------------------------------------


def test_leakage_check_excluded_from_bare_bundle():
    # Must not run in the default bundle (it drives a live model, like accuracy).
    assert score.eval_leakage_check not in score.EVALS


def test_leakage_check_weight_is_zero():
    # Weight 0.0 guarantees it can never perturb the weighted composite.
    assert score._LEAKAGE_WEIGHT == 0.0
    result = score.eval_leakage_check(lm=UnrelatedLM(), items=_items(), n=2, seed=0)
    assert result["weight"] == 0.0


def test_leakage_check_registered_as_dimension():
    assert score._DIMENSIONS["leakage_check"] is score.eval_leakage_check


def test_existing_accuracy_dimension_unchanged():
    # The additive gate must not disturb the existing accuracy dimension.
    assert score._DIMENSIONS["accuracy"] is score.eval_accuracy
    assert score._ACCURACY_WEIGHT == 0.50
    assert score.eval_accuracy not in score.EVALS
