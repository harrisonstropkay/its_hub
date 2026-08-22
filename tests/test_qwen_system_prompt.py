"""Shape tests for the quantitative token-budget ``QWEN_SYSTEM_PROMPT`` (H1).

H1 replaces the open-ended ``QWEN_SYSTEM_PROMPT`` with a quantitative
token-budget + guaranteed-box rewrite grounded in TALE (arXiv:2412.18547),
attacking the dominant CONTEXT_TRUNCATION failure mode by making the model emit
a terminal ``\\boxed{}`` before the completion window is exhausted.

These assert the *structural* properties the scored self-consistency config and
the FIXED extractors (``_extract_boxed``, ``n``) depend on, not an exact string,
so the wording can be tuned without churning the test. New tests only -- no
existing tests are modified.
"""

import re

from its_hub.core.utils import QWEN_SYSTEM_PROMPT


def test_prompt_states_explicit_numeric_token_budget():
    """(a) An explicit *numeric* reasoning-token ceiling must be present.

    This is the quantitative structural constraint that distinguishes H1 from
    the reverted exp-1 qualitative "be concise" directive.
    """
    numbers = [int(n) for n in re.findall(r"\d+", QWEN_SYSTEM_PROMPT)]
    assert numbers, "prompt must contain an explicit numeric reasoning-token budget"
    # Guard against TALE token-elasticity: the budget must stay >= ~512 tokens.
    assert any(n >= 512 for n in numbers), (
        f"expected a token budget >= 512, found only {numbers}"
    )
    # And it must reference tokens, so the number is a token budget (not, e.g.,
    # option letters or a step count).
    assert "token" in QWEN_SYSTEM_PROMPT.lower()


def test_prompt_contains_boxed_directive():
    """(d) The ``\\boxed{}`` convention the FIXED extractors expect is preserved."""
    assert "\\boxed{}" in QWEN_SYSTEM_PROMPT


def test_prompt_preserves_step_by_step_scaffold():
    """(c) The 'step by step' scaffold Qwen expects must be preserved.

    Dropping it toward direct-answer / no-reasoning caused Qwen
    repetition-collapse in the reverted exp-1.
    """
    assert "step by step" in QWEN_SYSTEM_PROMPT.lower()


def test_prompt_covers_both_math_and_multiple_choice():
    """(d) The box directive must cover numeric/expression AND a single MCQ letter."""
    lowered = QWEN_SYSTEM_PROMPT.lower()
    assert "multiple-choice" in lowered or "multiple choice" in lowered
    # A single letter answer for MCQ items (GPQA A-D).
    assert re.search(r"\bA,?\s*B,?\s*C,?\s*(?:or\s*)?D\b", QWEN_SYSTEM_PROMPT)


def test_prompt_directs_reaching_answer_early():
    """(b) The model is instructed to reach the answer early / keep reasoning brief."""
    lowered = QWEN_SYSTEM_PROMPT.lower()
    assert "early" in lowered or "brief" in lowered
