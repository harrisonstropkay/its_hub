"""Unit tests for eval_leakage_check() in eval/score.py (H4, exp_id=9).

The contamination gate measures VERBATIM RECALL (memorization), not answer
correctness (capability). It dispatches on the benchmark type:

  * open-ended (math500, aime-2024) -> Track A: guided-instruction quiz
    completion with a WITH/WITHOUT-dataset-context ROUGE-L baseline delta.
  * multiple-choice (gpqa-diamond)  -> Track B: TS-Guessing masked wrong-option
    reconstruction scored by normalized exact-match rate.

Everything is exercised with a MOCKED LM and injected items — no GPU, no
network, no dataset download. These tests strengthen the original H4 scaffold
and add REGRESSION tests for the live failure that sank commit 5f1b429:

  * ANTI-INVERSION: a clean, non-reciting model on single-letter GPQA golds
    (C/D/B) must PASS (rate ~0), not the old substring-inversion rate=1.000.
  * incidental single-digit golds inside CoT must NOT match (word-boundary).
  * a genuinely memorizing model (recites the masked option / exact
    continuation) IS flagged contaminated.
  * MATH track baseline-delta logic: high overlap WITH context + low WITHOUT
    -> contaminated; equal overlap -> clean.

It also confirms the gate stays purely additive (absent from EVALS, weight 0.0,
accuracy dimension untouched) and degrades gracefully.
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


# --- item factories ---------------------------------------------------------


def _open_items(n=8):
    """Open-ended (math-style) items with multi-word problems for Track A."""
    return [
        {
            "problem": (
                f"Problem index {i}: compute the value of the long expression "
                f"below and report the final reduced answer for scenario number "
                f"{i} using careful algebraic manipulation throughout the work"
            ),
            "answer": str(2 * i),
            "unique_id": f"open-{i}",
        }
        for i in range(n)
    ]


def _mc_items(n=6):
    """GPQA-style multiple-choice items matching the real loader schema.

    Each `problem` embeds four `X) text` option lines and the boxed-answer
    instruction, exactly as benchmarking/benchmark.py::_normalize_gpqa_row
    builds them; `answer` is the correct LETTER (single char).
    """
    golds = ["C", "D", "B", "A", "C", "D", "B", "A", "C", "D"]
    letters = ["A", "B", "C", "D"]
    items = []
    for i in range(n):
        opts = {
            L: f"distinct option {L} statement about phenomenon {i}-{j}"
            for j, L in enumerate(letters)
        }
        problem = (
            f"Question stem for item {i}: which statement is correct?\n\n"
            + "\n".join(f"{L}) {opts[L]}" for L in letters)
            + "\n\nPlease reason step by step, and put the letter of the "
            "correct option (A, B, C, or D) in \\boxed{}."
        )
        items.append(
            {"problem": problem, "answer": golds[i % len(golds)], "unique_id": f"g-{i}"}
        )
    return items


# --- Track A (open-ended) mock LMs ------------------------------------------


class MemorizingOpenLM:
    """Recites the exact continuation ONLY when the dataset is named.

    Mimics a contaminated model: with the WITH-context probe (which mentions the
    benchmark) it recalls the held-out continuation verbatim; on the neutral
    baseline probe it cannot, so the WITH/WITHOUT delta is large.
    """

    def __init__(self, items):
        self._map = {}
        for it in items:
            p = it["problem"]
            split = len(p) // 2
            self._map[p[:split]] = p[split:]

    def generate(self, messages):
        prompt = messages[-1]["content"]
        with_context = "benchmark" in prompt.lower()
        for prefix, cont in self._map.items():
            if prefix in prompt:
                return {"content": cont if with_context else "no idea sorry"}
        return {"content": "no idea sorry"}


class EqualOverlapOpenLM:
    """Recalls the continuation on BOTH probes (equal overlap -> delta ~0).

    A strong/fluent model that completes the prefix regardless of framing must
    read as clean, because the baseline cancels capability.
    """

    def __init__(self, items):
        self._map = {}
        for it in items:
            p = it["problem"]
            split = len(p) // 2
            self._map[p[:split]] = p[split:]

    def generate(self, messages):
        prompt = messages[-1]["content"]
        for prefix, cont in self._map.items():
            if prefix in prompt:
                return {"content": cont}
        return {"content": "no idea sorry"}


class CleanOpenLM:
    """Never recalls the continuation on either probe -> low overlap, clean."""

    def generate(self, messages):
        return {"content": "I will not guess the hidden text at all here today"}


class AsyncMemorizingOpenLM(MemorizingOpenLM):
    """Same as MemorizingOpenLM but via the real async agenerate_single seam."""

    async def agenerate_single(self, messages, **kwargs):
        return self.generate(messages)


# --- Track B (multiple-choice) mock LMs -------------------------------------


class MemorizingMCLM:
    """Reconstructs the masked wrong-option text verbatim (contaminated)."""

    def __init__(self, items):
        self._parsed = [score._parse_mc_problem(it["problem"]) for it in items]

    def generate(self, messages):
        import re

        prompt = messages[-1]["content"]
        m = re.search(r"option ([A-D])[.:]?\s*$", prompt.strip())
        masked = m.group(1) if m else None
        for stem, options in self._parsed:
            if stem and stem in prompt and masked in options:
                return {"content": options[masked]}
        return {"content": "unknown"}


class CleanMCLM:
    """A clean model: emits reasoning that never recites the masked option.

    The reasoning even contains the letters A/B/C/D — the anti-inversion guard:
    a letter appearing in prose must NOT count as a match.
    """

    def generate(self, messages):
        return {
            "content": (
                "Based on my analysis of options A, B, C and D, I believe the "
                "correct choice is C, but I do not recall the masked text."
            )
        }


class AsyncMemorizingMCLM(MemorizingMCLM):
    async def agenerate_single(self, messages, **kwargs):
        return self.generate(messages)


# --- Track A: memorization detection ----------------------------------------


def test_open_memorizing_model_flagged_contaminated():
    items = _open_items()
    result = score.eval_leakage_check(
        benchmark="math500", lm=MemorizingOpenLM(items), items=items, n=6, seed=0
    )
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert "contamination detected" in result["details"]
    assert "ROUGE-L delta" in result["details"]


def test_open_memorizing_async_interface():
    items = _open_items()
    result = score.eval_leakage_check(
        benchmark="math500", lm=AsyncMemorizingOpenLM(items), items=items, n=6, seed=0
    )
    assert result["passed"] is False
    assert "contamination detected" in result["details"]


def test_open_equal_overlap_is_clean():
    # High overlap WITH and WITHOUT context -> delta ~0 -> capability, not leak.
    items = _open_items()
    result = score.eval_leakage_check(
        benchmark="math500", lm=EqualOverlapOpenLM(items), items=items, n=6, seed=0
    )
    assert result["passed"] is True
    assert result["score"] == 1.0
    assert "no contamination" in result["details"]


def test_open_clean_model_passes():
    items = _open_items()
    result = score.eval_leakage_check(
        benchmark="math500", lm=CleanOpenLM(), items=items, n=6, seed=0
    )
    assert result["passed"] is True
    assert "no contamination" in result["details"]


def test_open_baseline_prompt_has_no_dataset_framing():
    # The WITHOUT-context probe must not name the dataset or say "benchmark".
    prefix = "compute the value of"
    with_ctx = score._open_probe_prompt_with_context("math500", prefix)
    no_ctx = score._open_probe_prompt_no_context(prefix)
    assert "MATH500" in with_ctx and "benchmark" in with_ctx.lower()
    assert "benchmark" not in no_ctx.lower()
    assert "math500" not in no_ctx.lower()
    # Neither probe leaks the held-out continuation (only the prefix is shown).
    assert prefix in with_ctx and prefix in no_ctx


# --- Track B: TS-Guessing masked wrong-option -------------------------------


def test_mc_memorizing_model_flagged_contaminated():
    items = _mc_items()
    result = score.eval_leakage_check(
        benchmark="gpqa-diamond", lm=MemorizingMCLM(items), items=items, n=6, seed=0
    )
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert "contamination detected" in result["details"]
    assert "masked-option EM" in result["details"]


def test_mc_memorizing_async_interface():
    items = _mc_items()
    result = score.eval_leakage_check(
        benchmark="gpqa-diamond", lm=AsyncMemorizingMCLM(items), items=items, n=6, seed=0
    )
    assert result["passed"] is False
    assert "contamination detected" in result["details"]


def test_mc_clean_model_not_inverted_on_single_letter_golds():
    # REGRESSION for the live failure: a clean, non-reciting model on
    # single-letter golds (C/D/B) previously scored rate=1.000 via substring
    # containment and INVERTED the gate. It must now PASS.
    items = _mc_items()
    assert {it["answer"] for it in items} <= {"A", "B", "C", "D"}
    result = score.eval_leakage_check(
        benchmark="gpqa-diamond", lm=CleanMCLM(), items=items, n=6, seed=0
    )
    assert result["passed"] is True
    assert result["score"] == 1.0
    assert "no contamination" in result["details"]
    assert "EM=0.000" in result["details"]


def test_mc_masked_option_text_never_in_prompt():
    # The masked WRONG option's text is the memorization target and must not
    # appear anywhere in the probe prompt.
    items = _mc_items(1)
    stem, options = score._parse_mc_problem(items[0]["problem"])
    correct = items[0]["answer"]
    wrong = sorted(x for x in options if x != correct)[0]
    prompt = score._mc_probe_prompt("gpqa-diamond", stem, options, correct, wrong)
    assert "[MASK]" in prompt
    assert options[wrong] not in prompt
    # Other options and the stem remain visible.
    assert stem in prompt
    other = [x for x in options if x not in (wrong,)][0]
    assert options[other] in prompt


def test_parse_mc_problem_matches_loader_schema():
    items = _mc_items(1)
    stem, options = score._parse_mc_problem(items[0]["problem"])
    assert set(options) == {"A", "B", "C", "D"}
    assert "Question stem for item 0" in stem
    # Trailing boxed instruction must not bleed into any option text.
    assert all("boxed" not in v.lower() for v in options.values())


def test_mc_unparseable_items_degrade_gracefully():
    # Items with no option lines cannot be probed -> skipped -> could not run,
    # never a false contamination flag.
    items = [
        {"problem": f"free-form question {i} with no options", "answer": "A",
         "unique_id": f"bad-{i}"}
        for i in range(4)
    ]
    result = score.eval_leakage_check(
        benchmark="gpqa-diamond", lm=CleanMCLM(), items=items, n=4, seed=0
    )
    assert result["passed"] is False
    assert "could not run" in result["details"]
    assert "skipped" in result["details"]


# --- matching helpers (replace the broken substring _leakage_exact_match) ----


def test_rouge_l_identical_is_one():
    assert score._rouge_l_f1("the quick brown fox", "the quick brown fox") == 1.0


def test_rouge_l_disjoint_is_zero():
    assert score._rouge_l_f1("alpha beta gamma", "one two three") == 0.0


def test_rouge_l_partial_is_between():
    s = score._rouge_l_f1("the quick brown fox jumps high", "the quick brown fox")
    assert 0.0 < s < 1.0


def test_rouge_l_empty_side_is_zero():
    assert score._rouge_l_f1("", "anything at all") == 0.0
    assert score._rouge_l_f1("anything at all", "") == 0.0


def test_word_boundary_no_incidental_digit_match():
    # These are the exact false positives from the live QA report — the new
    # matcher must reject every one of them.
    assert score._word_boundary_match("4", "factors of 14 and 42") is False
    assert score._word_boundary_match("6", "prime factorization of 10!") is False
    assert score._word_boundary_match("2", "the recipe needs 42 eggs and 10 cups") is False


def test_word_boundary_matches_standalone_token():
    assert score._word_boundary_match("2", "the final answer is 2.") is True
    assert score._word_boundary_match("331", "we get 331 exactly") is True


def test_normalize_math_answer_forgives_rendering():
    assert score._normalize_math_answer("$1{,}000$") == score._normalize_math_answer("1000")
    assert score._normalize_math_answer("\\boxed{42}") == "42"


def test_normalize_option_text_exact_equality():
    assert score._normalize_option_text("The Answer.") == score._normalize_option_text("the answer")
    assert score._normalize_option_text("foo") != score._normalize_option_text("bar")


def test_extract_answer_prefers_boxed():
    assert score._leakage_extract_answer("work here\n\\boxed{7}") == "7"
    assert score._leakage_extract_answer("line one\nfinal line") == "final line"


# --- seeded sampling is deterministic ---------------------------------------


def _sampled_ids(items, seed, n=4):
    """Recover which items were probed for a given seed via a recording LM."""
    seen = []

    class RecordingLM:
        def generate(self, messages):
            prompt = messages[-1]["content"]
            for it in items:
                # match on the unique prefix of each problem
                if it["problem"][: len(it["problem"]) // 2] in prompt:
                    seen.append(it["unique_id"])
                    break
            return {"content": "no idea sorry"}

    score.eval_leakage_check(
        benchmark="math500", lm=RecordingLM(), items=items, n=n, seed=seed
    )
    # de-dup while preserving order (each item is probed twice: with/without)
    ordered = []
    for uid in seen:
        if uid not in ordered:
            ordered.append(uid)
    return ordered


def test_seeded_sampling_is_deterministic():
    items = _open_items(20)
    first = _sampled_ids(items, seed=123)
    second = _sampled_ids(items, seed=123)
    assert first == second
    assert len(first) == 4


def test_different_seeds_select_different_items():
    items = _open_items(50)
    a = _sampled_ids(items, seed=1)
    b = _sampled_ids(items, seed=2)
    assert a != b


def test_sampled_indices_are_sorted_and_in_range():
    items = _open_items(20)
    ids = _sampled_ids(items, seed=7)
    idxs = [int(x.split("-")[1]) for x in ids]
    assert idxs == sorted(idxs)
    assert all(0 <= i < 20 for i in idxs)


# --- threshold boundary behaviour -------------------------------------------


def test_mc_threshold_boundary_rate_equal_passes():
    # 6 items, memorizing model -> EM rate 1.0. threshold 1.0 -> rate == thr passes.
    items = _mc_items(6)
    result = score.eval_leakage_check(
        benchmark="gpqa-diamond", lm=MemorizingMCLM(items), items=items,
        n=6, seed=0, threshold=1.0,
    )
    assert result["passed"] is True
    assert "EM=1.000" in result["details"]


def test_mc_threshold_boundary_above_fails():
    items = _mc_items(6)
    result = score.eval_leakage_check(
        benchmark="gpqa-diamond", lm=MemorizingMCLM(items), items=items,
        n=6, seed=0, threshold=0.99,
    )
    assert result["passed"] is False
    assert "contamination detected" in result["details"]


def test_open_high_threshold_suppresses_flag():
    # Even a memorizing model is not flagged if the delta threshold is >= 1.0.
    items = _open_items()
    result = score.eval_leakage_check(
        benchmark="math500", lm=MemorizingOpenLM(items), items=items,
        n=6, seed=0, threshold=1.0,
    )
    assert result["passed"] is True


# --- return-shape / contract ------------------------------------------------


def test_result_matches_contract():
    items = _open_items()
    result = score.eval_leakage_check(
        benchmark="math500", lm=CleanOpenLM(), items=items, n=3, seed=0
    )
    assert set(result) == _CONTRACT_KEYS
    assert result["name"] == "leakage_check"
    assert isinstance(result["score"], float)
    assert isinstance(result["weight"], float)
    assert isinstance(result["passed"], bool)
    assert isinstance(result["details"], str) and result["details"]


def test_n_larger_than_pool_is_clamped():
    items = _open_items(3)
    result = score.eval_leakage_check(
        benchmark="math500", lm=CleanOpenLM(), items=items, n=99, seed=0
    )
    assert result["passed"] is True
    assert "3 items" in result["details"]


def test_default_threshold_is_track_specific():
    # No explicit threshold -> open track uses the 0.3 ROUGE-L delta default.
    items = _open_items()
    result = score.eval_leakage_check(
        benchmark="math500", lm=CleanOpenLM(), items=items, n=4, seed=0
    )
    assert "threshold=0.300" in result["details"]
    # ...and the MC track uses the 0.5 exact-match default.
    mc = _mc_items()
    mc_result = score.eval_leakage_check(
        benchmark="gpqa-diamond", lm=CleanMCLM(), items=mc, n=4, seed=0
    )
    assert "threshold=0.500" in mc_result["details"]


# --- graceful degradation ---------------------------------------------------


def test_no_items_degrades_gracefully():
    result = score.eval_leakage_check(lm=CleanOpenLM(), items=[], n=5, seed=0)
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert "could not run" in result["details"]


def test_no_lm_and_no_endpoint_degrades(monkeypatch):
    monkeypatch.delenv("ITS_ENDPOINT", raising=False)
    monkeypatch.delenv("OPENAI_ENDPOINT", raising=False)
    result = score.eval_leakage_check(items=_open_items(), n=5, seed=0)
    assert result["passed"] is False
    assert "could not run" in result["details"]


# --- purely-additive guarantees ---------------------------------------------


def test_leakage_check_excluded_from_bare_bundle():
    assert score.eval_leakage_check not in score.EVALS


def test_leakage_check_weight_is_zero():
    assert score._LEAKAGE_WEIGHT == 0.0
    result = score.eval_leakage_check(lm=CleanOpenLM(), items=_open_items(), n=2, seed=0)
    assert result["weight"] == 0.0


def test_leakage_check_registered_as_dimension():
    assert score._DIMENSIONS["leakage_check"] is score.eval_leakage_check


def test_existing_accuracy_dimension_unchanged():
    assert score._DIMENSIONS["accuracy"] is score.eval_accuracy
    assert score._ACCURACY_WEIGHT == 0.50
    assert score.eval_accuracy not in score.EVALS
