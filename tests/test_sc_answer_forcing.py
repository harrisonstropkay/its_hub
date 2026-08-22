"""Tests for answer/budget forcing in SelfConsistency (H1, ITS_SC_ANSWER_FORCE).

Covers the default-OFF ``ITS_SC_ANSWER_FORCE`` decode mode that attacks the
dominant CONTEXT_TRUNCATION failure by capping main generation below the context
window and issuing ONE short forced ``\\boxed{}`` continuation for any sample
that never emitted a terminal answer:

  (a) flag OFF (default): the orchestrator generation call carries NO cap
      (``max_tokens``/``max_completion_tokens`` absent) and NO forced
      continuation is issued -- the path is byte-identical to baseline;
  (b) flag ON: a box-absent sample is completed into a real ``\\boxed{...}``;
  (c) flag ON: a sample that already carries a box is left untouched;
  (d) MCQ-shaped prompts use the tighter continuation budget; numeric/ambiguous
      prompts use the wider one;
  (e) the flag resolver / detection helpers behave as specified.

New tests only -- no existing tests are modified.
"""

import logging

import pytest

from its_hub.api import (
    AbstractLanguageModel,
    AbstractOrchestrator,
    ChatMessages,
    GenerationUsage,
)
from its_hub.core.algorithms.self_consistency import (
    _ANSWER_FORCE_CONT_MAX_TOKENS_MCQ,
    _ANSWER_FORCE_CONT_MAX_TOKENS_NUMERIC,
    _ANSWER_FORCE_MAIN_MAX_TOKENS,
    _ANSWER_FORCE_STOP,
    SelfConsistency,
    _has_boxed_answer,
    _looks_like_mcq,
    _resolve_answer_force,
)
from its_hub.core.utils import CONTINUATION_PRIMER

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class RecordingOrchestrator(AbstractOrchestrator):
    """Returns fixed responses and records the kwargs of each ``agenerate`` call.

    Returning canned responses (rather than delegating to the LM) isolates the
    main-generation call from the forced-continuation calls, so the test can
    assert on each independently.
    """

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
BOXED = {"role": "assistant", "content": "the answer is \\boxed{7}"}


@pytest.fixture(autouse=True)
def _clear_flag(monkeypatch):
    """Every test starts with the flag unset unless it explicitly sets it."""
    monkeypatch.delenv("ITS_SC_ANSWER_FORCE", raising=False)


# ---------------------------------------------------------------------------
# _resolve_answer_force
# ---------------------------------------------------------------------------


class TestResolveAnswerForce:
    def test_unset_defaults_off(self, monkeypatch):
        monkeypatch.delenv("ITS_SC_ANSWER_FORCE", raising=False)
        assert _resolve_answer_force() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "  On  "])
    def test_truthy_values_enable(self, monkeypatch, value):
        monkeypatch.setenv("ITS_SC_ANSWER_FORCE", value)
        assert _resolve_answer_force() is True

    @pytest.mark.parametrize("value", ["0", "", "false", "no", "off", "garbage"])
    def test_falsy_values_disable(self, monkeypatch, value):
        monkeypatch.setenv("ITS_SC_ANSWER_FORCE", value)
        assert _resolve_answer_force() is False


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


class TestBoxDetection:
    def test_completed_box_present(self):
        assert _has_boxed_answer("so \\boxed{42} done")

    def test_nested_braces_present(self):
        assert _has_boxed_answer("\\boxed{\\frac{1}{2}}")

    def test_truncated_open_box_absent(self):
        assert not _has_boxed_answer("almost there \\boxed{4")

    def test_no_box_absent(self):
        assert not _has_boxed_answer("no answer here at all")

    def test_empty_box_absent(self):
        assert not _has_boxed_answer("\\boxed{}")

    def test_none_and_empty(self):
        assert not _has_boxed_answer(None)
        assert not _has_boxed_answer("")


class TestMCQDetection:
    def test_four_option_prompt_is_mcq(self):
        prompt = "Q?\nA) alpha\nB) beta\nC) gamma\nD) delta"
        assert _looks_like_mcq(prompt) is True

    def test_parenthesized_options_is_mcq(self):
        prompt = "Pick: (A) x (B) y (C) z (D) w"
        assert _looks_like_mcq(prompt) is True

    def test_numeric_prompt_not_mcq(self):
        assert _looks_like_mcq("What is 2 + 2?") is False

    def test_ambiguous_single_letter_not_mcq(self):
        # A single stray "A)" is not enough signal -> default to numeric.
        assert _looks_like_mcq("See figure A) for details.") is False

    def test_none_not_mcq(self):
        assert _looks_like_mcq(None) is False


# ---------------------------------------------------------------------------
# Flag OFF: byte-identical baseline path
# ---------------------------------------------------------------------------


class TestFlagOffByteIdentical:
    @pytest.mark.asyncio
    async def test_generation_call_has_no_cap_and_no_continuation(self, monkeypatch):
        monkeypatch.delenv("ITS_SC_ANSWER_FORCE", raising=False)
        orch = RecordingOrchestrator([dict(BOXLESS), dict(BOXLESS), dict(BOXLESS)])
        lm = ContinuationMockLM()
        sc = SelfConsistency(orchestrator=orch)

        result = await sc.ainfer(lm, "test prompt", budget=3, return_response_only=True)

        # Exactly one generation call, and it carries NO token cap.
        assert len(orch.agenerate_kwargs) == 1
        kwargs = orch.agenerate_kwargs[0]
        assert "max_tokens" not in kwargs
        assert "max_completion_tokens" not in kwargs
        # Baseline kwargs are exactly the historical set.
        assert set(kwargs) == {"tools", "tool_choice", "usage_accumulator", "logprobs"}
        assert kwargs["logprobs"] is True
        # No forced continuation was issued.
        assert lm.single_calls == []
        # Response content is returned unmodified (no forced box spliced in).
        assert result["content"] == BOXLESS["content"]


# ---------------------------------------------------------------------------
# Flag ON: forcing behavior
# ---------------------------------------------------------------------------


class TestFlagOnForcing:
    @pytest.mark.asyncio
    async def test_box_absent_sample_is_forced(self, monkeypatch):
        monkeypatch.setenv("ITS_SC_ANSWER_FORCE", "1")
        orch = RecordingOrchestrator([dict(BOXLESS), dict(BOXLESS), dict(BOXLESS)])
        lm = ContinuationMockLM(continuation="42")
        sc = SelfConsistency(orchestrator=orch)

        result = await sc.ainfer(
            lm, "test prompt", budget=3, return_response_only=False
        )

        # Main generation was capped below the window.
        assert orch.agenerate_kwargs[0]["max_tokens"] == _ANSWER_FORCE_MAIN_MAX_TOKENS
        # One forced continuation per box-absent sample.
        assert len(lm.single_calls) == 3
        # Every completed response now carries a real boxed answer.
        for resp in result.responses:
            assert "\\boxed{42}" in resp["content"]
            assert _has_boxed_answer(resp["content"])
        # the_one is the forced, boxed answer.
        assert "\\boxed{42}" in result.the_one["content"]

    @pytest.mark.asyncio
    async def test_continuation_call_shape(self, monkeypatch):
        monkeypatch.setenv("ITS_SC_ANSWER_FORCE", "1")
        orch = RecordingOrchestrator([dict(BOXLESS)])
        lm = ContinuationMockLM(continuation="42")
        sc = SelfConsistency(orchestrator=orch)

        await sc.ainfer(lm, "test prompt", budget=1)

        call = lm.single_calls[0]
        # numeric/ambiguous prompt -> wider continuation budget + stop brace.
        assert call["stop"] == _ANSWER_FORCE_STOP == "}"
        assert call["max_completion_tokens"] == _ANSWER_FORCE_CONT_MAX_TOKENS_NUMERIC
        # The assistant turn is the draft + primer (continuation appended to it).
        assistant_turn = call["messages"][-1]
        assert assistant_turn.role == "assistant"
        assert assistant_turn.content == BOXLESS["content"] + CONTINUATION_PRIMER

    @pytest.mark.asyncio
    async def test_boxed_sample_is_not_forced(self, monkeypatch):
        monkeypatch.setenv("ITS_SC_ANSWER_FORCE", "1")
        orch = RecordingOrchestrator([dict(BOXED), dict(BOXLESS)])
        lm = ContinuationMockLM(continuation="42")
        sc = SelfConsistency(orchestrator=orch)

        result = await sc.ainfer(
            lm, "test prompt", budget=2, return_response_only=False
        )

        # Only the box-absent sample triggered a continuation.
        assert len(lm.single_calls) == 1
        # The already-boxed sample is byte-unchanged.
        contents = {r["content"] for r in result.responses}
        assert BOXED["content"] in contents

    @pytest.mark.asyncio
    async def test_mcq_prompt_uses_tighter_budget(self, monkeypatch):
        monkeypatch.setenv("ITS_SC_ANSWER_FORCE", "1")
        orch = RecordingOrchestrator([dict(BOXLESS)])
        lm = ContinuationMockLM(continuation="C")
        sc = SelfConsistency(orchestrator=orch)

        mcq_prompt = "Which one?\nA) alpha\nB) beta\nC) gamma\nD) delta"
        await sc.ainfer(lm, mcq_prompt, budget=1)

        assert lm.single_calls[0]["max_completion_tokens"] == (
            _ANSWER_FORCE_CONT_MAX_TOKENS_MCQ
        )


# ---------------------------------------------------------------------------
# Flag ON: stale-logprob correctness on forced samples
# ---------------------------------------------------------------------------


def _logprobs(mean: float) -> dict:
    """OpenAI-format ``_logprobs`` payload with a single token of ``mean``."""
    return {"content": [{"logprob": mean}]}


class TestForcedSampleLogprobsNulled:
    """A forced sample's stale draft ``_logprobs`` must not resolve a vote tie.

    ``_force_box_absent_samples`` overrides only ``content`` on a forced sample;
    its ``_logprobs`` still describe the TRUNCATED draft, not the forced final
    answer. Left in place, ``_aggregate_logprob`` would compute a confidence over
    the wrong tokens and ``_process_responses`` could let a forced sample win a
    tie on that stale confidence. The fix nulls ``_logprobs`` on forced samples.
    """

    @pytest.mark.asyncio
    async def test_forced_sample_logprobs_are_nulled(self, monkeypatch):
        monkeypatch.setenv("ITS_SC_ANSWER_FORCE", "1")
        # A single box-absent sample carrying a (high) draft confidence.
        boxless = {**BOXLESS, "_logprobs": _logprobs(-0.01)}
        orch = RecordingOrchestrator([dict(boxless)])
        lm = ContinuationMockLM(continuation="9")
        sc = SelfConsistency(orchestrator=orch)

        result = await sc.ainfer(
            lm, "What is 2 + 2?", budget=1, return_response_only=False
        )

        forced = result.responses[0]
        # Content was completed into a real boxed answer...
        assert _has_boxed_answer(forced["content"])
        # ...but the stale draft logprobs are gone, so aggregate confidence is
        # None and the forced sample is excluded from the tie-break.
        assert forced["_logprobs"] is None
        assert SelfConsistency._aggregate_logprob(forced) is None

    @pytest.mark.asyncio
    async def test_forced_sample_does_not_win_tie_via_stale_confidence(
        self, monkeypatch
    ):
        monkeypatch.setenv("ITS_SC_ANSWER_FORCE", "1")
        # Response 0: a GENUINE boxed answer with a LOW draft confidence.
        # Response 1: box-absent, but with a HIGH stale draft confidence that --
        # if preserved -- would let it win the 1-1 tie on the tie-break.
        genuine = {**BOXED, "_logprobs": _logprobs(-2.0)}  # \boxed{7}
        stale_high = {**BOXLESS, "_logprobs": _logprobs(-0.01)}
        orch = RecordingOrchestrator([dict(genuine), dict(stale_high)])
        lm = ContinuationMockLM(continuation="9")  # forced -> \boxed{9}
        sc = SelfConsistency(orchestrator=orch)

        result = await sc.ainfer(
            lm, "What is 2 + 2?", budget=2, return_response_only=False
        )

        # Two distinct answer groups (7 vs forced 9), a genuine 1-1 tie.
        assert len(result.response_counts) == 2
        assert set(result.response_counts.values()) == {1}
        # The genuine boxed sample (index 0) wins the tie: the forced sample's
        # stale high confidence was nulled and cannot resolve the tie.
        assert result.selected_index == 0
        assert "\\boxed{7}" in result.the_one["content"]


# ---------------------------------------------------------------------------
# Flag ON: empty-continuation and tool-call edge cases (direct helper tests)
# ---------------------------------------------------------------------------


class TestForcingEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_continuation_stays_box_absent(self, monkeypatch, caplog):
        monkeypatch.setenv("ITS_SC_ANSWER_FORCE", "1")
        # Continuation yields no body -> completed text is "\boxed{}" (no answer).
        orch = RecordingOrchestrator([dict(BOXLESS)])
        lm = ContinuationMockLM(continuation="")
        sc = SelfConsistency(orchestrator=orch)

        with caplog.at_level(logging.INFO):
            result = await sc.ainfer(
                lm, "What is 2 + 2?", budget=1, return_response_only=False
            )

        # The forced sample still has no real boxed answer.
        assert not _has_boxed_answer(result.responses[0]["content"])
        # Telemetry records the still-truncated sample.
        assert "n_forced=1" in caplog.text
        assert "n_still_truncated=1" in caplog.text

    @pytest.mark.asyncio
    async def test_tool_call_sample_is_left_untouched(self):
        # Directly exercise the forcing helper so tool-call routing in voting
        # does not obscure the "skip tool calls" behavior under test.
        tool_response = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "function": {"name": "lookup", "arguments": "{}"}}
            ],
        }
        lm = ContinuationMockLM(continuation="9")
        sc = SelfConsistency()
        chat_messages = ChatMessages.from_prompt_or_messages("What is 2 + 2?")
        usage = GenerationUsage()

        forced = await sc._force_box_absent_samples(
            lm, chat_messages, [dict(tool_response)], usage
        )

        # No continuation was issued for the tool-call sample...
        assert lm.single_calls == []
        # ...and it is returned byte-unchanged.
        assert forced[0] == tool_response
