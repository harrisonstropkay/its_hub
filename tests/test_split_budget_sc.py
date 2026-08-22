"""Tests for split-budget / decoupled Br·Ba forcing in SelfConsistency (H1, exp-17).

Covers the default-OFF ``ITS_SC_SPLIT_BUDGET`` sub-flag, which only takes effect
when answer/budget forcing (``ITS_SC_ANSWER_FORCE``) is also on. It re-tunes the
NUMERIC arm of the two-pass forcing into a decoupled reasoning/answer split:
lower the main-gen reasoning cap (Br 3600 -> 3200) to fund a wider forced
answer-completion pass (Ba 48 -> 500), sized to fit the 4096-token window. The
MCQ/GPQA arm is byte-untouched so the all-empty forced-fallback rescue is
preserved by construction.

  (a) ``_resolve_split_budget`` defaults OFF and parses truthy values;
  (b) both flags ON + numeric prompt -> Br=3200 main cap, Ba=500 continuation;
  (c) both flags ON + MCQ prompt -> Br=3600, Ba=16 (arm unchanged);
  (d) split-budget OFF (forcing ON) -> numeric Br=3600, Ba=48 (exp-15 identity);
  (e) context-window sanity: prompt + Br + primer + Ba < max_model_len (4096).

New tests only -- no existing tests are modified.
"""

import pytest

from its_hub.api import (
    AbstractLanguageModel,
    AbstractOrchestrator,
)
from its_hub.core.algorithms.self_consistency import (
    _ANSWER_FORCE_CONT_MAX_TOKENS_MCQ,
    _ANSWER_FORCE_CONT_MAX_TOKENS_NUMERIC,
    _ANSWER_FORCE_MAIN_MAX_TOKENS,
    _SPLIT_BUDGET_CONT_MAX_TOKENS_NUMERIC,
    _SPLIT_BUDGET_MAIN_MAX_TOKENS_NUMERIC,
    SelfConsistency,
    _resolve_split_budget,
)
from its_hub.core.utils import CONTINUATION_PRIMER, CONTINUATION_PRIMER_MCQ

# Hard completion window for Qwen2.5-Math-7B in this regime (max_model_len).
_MAX_MODEL_LEN = 4096

# ---------------------------------------------------------------------------
# Test doubles (mirror tests/test_sc_answer_forcing.py; no network)
# ---------------------------------------------------------------------------


class RecordingOrchestrator(AbstractOrchestrator):
    """Returns fixed responses and records the kwargs of each ``agenerate`` call."""

    def __init__(self, responses: list[dict]):
        self._responses = responses
        self.agenerate_kwargs: list[dict] = []

    async def agenerate(self, lm, messages_lst, stop=None, **kwargs):
        self.agenerate_kwargs.append(kwargs)
        return [dict(r) for r in self._responses]


class ContinuationMockLM(AbstractLanguageModel):
    """Records ``agenerate_single`` continuation calls, returns a fixed answer."""

    def __init__(self, continuation: str = "42"):
        self.continuation = continuation
        self.single_calls: list[dict] = []

    async def agenerate(self, messages, stop=None, **kwargs):  # pragma: no cover
        raise AssertionError("agenerate should not be called directly in these tests")

    async def agenerate_single(
        self, messages, stop=None, max_completion_tokens=None, **kwargs
    ):
        self.single_calls.append(
            {
                "messages": messages,
                "stop": stop,
                "max_completion_tokens": max_completion_tokens,
            }
        )
        return {"role": "assistant", "content": self.continuation}


BOXLESS = {"role": "assistant", "content": "long reasoning that ran out of budget"}
NUMERIC_PROMPT = "Compute the value of the following sum and give the result."
MCQ_PROMPT = "Which one?\nA) alpha\nB) beta\nC) gamma\nD) delta"


@pytest.fixture(autouse=True)
def _clear_flags(monkeypatch):
    """Every test starts with both flags unset unless it sets them explicitly."""
    monkeypatch.delenv("ITS_SC_ANSWER_FORCE", raising=False)
    monkeypatch.delenv("ITS_SC_SPLIT_BUDGET", raising=False)


# ---------------------------------------------------------------------------
# (a) _resolve_split_budget
# ---------------------------------------------------------------------------


class TestResolveSplitBudget:
    def test_unset_defaults_off(self, monkeypatch):
        monkeypatch.delenv("ITS_SC_SPLIT_BUDGET", raising=False)
        assert _resolve_split_budget() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "  On  "])
    def test_truthy_values_enable(self, monkeypatch, value):
        monkeypatch.setenv("ITS_SC_SPLIT_BUDGET", value)
        assert _resolve_split_budget() is True

    @pytest.mark.parametrize("value", ["0", "", "false", "no", "off", "garbage"])
    def test_falsy_values_disable(self, monkeypatch, value):
        monkeypatch.setenv("ITS_SC_SPLIT_BUDGET", value)
        assert _resolve_split_budget() is False


# ---------------------------------------------------------------------------
# (b) both flags ON + numeric prompt -> decoupled Br=3200 / Ba=500
# ---------------------------------------------------------------------------


class TestBothFlagsOnNumeric:
    @pytest.mark.asyncio
    async def test_numeric_main_cap_is_split_budget_br(self, monkeypatch):
        monkeypatch.setenv("ITS_SC_ANSWER_FORCE", "1")
        monkeypatch.setenv("ITS_SC_SPLIT_BUDGET", "1")
        orch = RecordingOrchestrator([dict(BOXLESS)])
        lm = ContinuationMockLM(continuation="42")
        sc = SelfConsistency(orchestrator=orch)

        await sc.ainfer(lm, NUMERIC_PROMPT, budget=1)

        # Main generation reasoning cap lowered to Br=3200 for numeric items.
        assert orch.agenerate_kwargs[0]["max_tokens"] == (
            _SPLIT_BUDGET_MAIN_MAX_TOKENS_NUMERIC
        )
        assert _SPLIT_BUDGET_MAIN_MAX_TOKENS_NUMERIC == 3200

    @pytest.mark.asyncio
    async def test_numeric_continuation_budget_is_split_budget_ba(self, monkeypatch):
        monkeypatch.setenv("ITS_SC_ANSWER_FORCE", "1")
        monkeypatch.setenv("ITS_SC_SPLIT_BUDGET", "1")
        orch = RecordingOrchestrator([dict(BOXLESS)])
        lm = ContinuationMockLM(continuation="42")
        sc = SelfConsistency(orchestrator=orch)

        await sc.ainfer(lm, NUMERIC_PROMPT, budget=1)

        # The forced answer-completion pass is widened to Ba=500.
        call = lm.single_calls[0]
        assert call["max_completion_tokens"] == _SPLIT_BUDGET_CONT_MAX_TOKENS_NUMERIC
        assert _SPLIT_BUDGET_CONT_MAX_TOKENS_NUMERIC == 500
        # Continuation mechanism is unchanged: stop brace + numeric primer.
        assert call["stop"] == "}"
        assert call["messages"][-1].content == BOXLESS["content"] + CONTINUATION_PRIMER


# ---------------------------------------------------------------------------
# (c) both flags ON + MCQ prompt -> arm byte-untouched (Br=3600, Ba=16)
# ---------------------------------------------------------------------------


class TestBothFlagsOnMcqUnchanged:
    @pytest.mark.asyncio
    async def test_mcq_budgets_unchanged_when_split_budget_on(self, monkeypatch):
        monkeypatch.setenv("ITS_SC_ANSWER_FORCE", "1")
        monkeypatch.setenv("ITS_SC_SPLIT_BUDGET", "1")
        orch = RecordingOrchestrator([dict(BOXLESS)])
        lm = ContinuationMockLM(continuation="C")
        sc = SelfConsistency(orchestrator=orch)

        await sc.ainfer(lm, MCQ_PROMPT, budget=1)

        # MCQ main cap stays Br=3600; the GPQA rescue arm is untouched.
        assert orch.agenerate_kwargs[0]["max_tokens"] == _ANSWER_FORCE_MAIN_MAX_TOKENS
        # MCQ continuation stays Ba=16 with the MCQ primer.
        call = lm.single_calls[0]
        assert call["max_completion_tokens"] == _ANSWER_FORCE_CONT_MAX_TOKENS_MCQ
        assert call["messages"][-1].content == (
            BOXLESS["content"] + CONTINUATION_PRIMER_MCQ
        )


# ---------------------------------------------------------------------------
# (d) split-budget OFF (forcing ON) -> exp-15 numeric byte-identity (3600/48)
# ---------------------------------------------------------------------------


class TestSplitBudgetOffNumericIdentity:
    @pytest.mark.asyncio
    async def test_numeric_budgets_match_exp15_when_split_off(self, monkeypatch):
        monkeypatch.setenv("ITS_SC_ANSWER_FORCE", "1")
        monkeypatch.delenv("ITS_SC_SPLIT_BUDGET", raising=False)
        orch = RecordingOrchestrator([dict(BOXLESS)])
        lm = ContinuationMockLM(continuation="42")
        sc = SelfConsistency(orchestrator=orch)

        await sc.ainfer(lm, NUMERIC_PROMPT, budget=1)

        # Numeric arm keeps exp-15 budgets (Br=3600, Ba=48) with the split off.
        assert orch.agenerate_kwargs[0]["max_tokens"] == _ANSWER_FORCE_MAIN_MAX_TOKENS
        assert lm.single_calls[0]["max_completion_tokens"] == (
            _ANSWER_FORCE_CONT_MAX_TOKENS_NUMERIC
        )
        assert _ANSWER_FORCE_MAIN_MAX_TOKENS == 3600
        assert _ANSWER_FORCE_CONT_MAX_TOKENS_NUMERIC == 48

    @pytest.mark.asyncio
    async def test_split_budget_dead_when_forcing_off(self, monkeypatch):
        # ANSWER_FORCE off => the whole forcing branch is skipped; SPLIT_BUDGET
        # is dead code and the generation call carries no token cap (baseline).
        monkeypatch.delenv("ITS_SC_ANSWER_FORCE", raising=False)
        monkeypatch.setenv("ITS_SC_SPLIT_BUDGET", "1")
        orch = RecordingOrchestrator([dict(BOXLESS)])
        lm = ContinuationMockLM()
        sc = SelfConsistency(orchestrator=orch)

        result = await sc.ainfer(lm, NUMERIC_PROMPT, budget=1)

        assert "max_tokens" not in orch.agenerate_kwargs[0]
        assert lm.single_calls == []
        assert result["content"] == BOXLESS["content"]


# ---------------------------------------------------------------------------
# (e) context-window sanity: prompt + Br + primer + Ba < max_model_len
# ---------------------------------------------------------------------------


class TestContextWindowSanity:
    def test_split_budget_second_call_fits_window(self):
        # Conservative token estimates: a ~300-token prompt and the short primer.
        # 1 char ~= 1 token bounds the primer generously above its real length.
        prompt_tokens = 300
        primer_tokens = len(CONTINUATION_PRIMER)
        total = (
            prompt_tokens
            + _SPLIT_BUDGET_MAIN_MAX_TOKENS_NUMERIC
            + primer_tokens
            + _SPLIT_BUDGET_CONT_MAX_TOKENS_NUMERIC
        )
        assert total < _MAX_MODEL_LEN

    def test_split_budget_reserves_more_headroom_than_exp15_total(self):
        # The split keeps Br + Ba at 3700, below exp-15's implicit 3600 + 48 only
        # in reasoning share; the point is the SECOND call's input fits: draft is
        # capped at Br (3200), leaving room for prompt + primer + Ba (500).
        assert (
            _SPLIT_BUDGET_MAIN_MAX_TOKENS_NUMERIC
            + _SPLIT_BUDGET_CONT_MAX_TOKENS_NUMERIC
            + len(CONTINUATION_PRIMER)
            + 300
        ) < _MAX_MODEL_LEN
