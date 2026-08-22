"""Tests for tiered (natural-first) voting in SelfConsistency (H1, exp-16).

Covers the ``ITS_SC_TIER_VOTING`` sub-flag layered on top of answer-forcing
(``ITS_SC_ANSWER_FORCE``). When tiering is active, plurality runs over the
NATURAL tier (samples that emitted a terminal ``\\boxed{}`` on their own) first;
the FORCED tier (samples completed by a synthetic continuation) is consulted
only as a fallback when no natural candidate survives:

  (a) the resolver is default-OFF opt-in: active ONLY on an explicit truthy
      ``ITS_SC_TIER_VOTING``; unset (even with forcing ON) and ``=0`` are OFF;
  (b) natural-first gate FIRES: a correct natural box beats forced wrong guesses
      (the AIME FORCED_ANSWER_DILUTION case);
  (c) forced-fallback FIRES: natural set empty -> forced plurality wins (the GPQA
      all-empty rescue case);
  (d) flag-OFF no-op: with no forced samples present, the eligible set and its
      ordering are unchanged whether or not tiering is enabled;
  (e) TIER unset / TIER=0 reproduces the non-tiered exp-15 behavior (forced
      dilution stands), preserving the exp-15 tie test byte-for-byte;
  (f) tier telemetry is emitted (observability only, no behavioral effect).

Because tiering is default-OFF, every case that exercises the tiering gate sets
``ITS_SC_TIER_VOTING=1`` explicitly via monkeypatch.

New tests only -- no existing tests are modified.
"""

import logging

import pytest

from its_hub.api import (
    AbstractLanguageModel,
    AbstractOrchestrator,
)
from its_hub.core.algorithms.self_consistency import (
    SelfConsistency,
    _resolve_tier_voting,
)

# ---------------------------------------------------------------------------
# Test doubles (mirror the answer-forcing suite so the two read alike)
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

    def __init__(self, continuation: str = "1"):
        self.continuation = continuation
        self.single_calls: list[dict] = []

    async def agenerate(self, messages, stop=None, **kwargs):  # pragma: no cover
        raise AssertionError("agenerate should not be called directly in these tests")

    async def agenerate_single(
        self, messages, stop=None, max_completion_tokens=None, **kwargs
    ):
        self.single_calls.append(
            {"messages": messages, "stop": stop, "max_completion_tokens": max_completion_tokens}
        )
        return {"role": "assistant", "content": self.continuation}


# A box-absent draft (triggers forcing) and a correct natural-stop boxed answer.
BOXLESS = {"role": "assistant", "content": "long reasoning that ran out of budget"}
NATURAL_CORRECT = {"role": "assistant", "content": "the work leads to \\boxed{42}"}


@pytest.fixture(autouse=True)
def _clear_flags(monkeypatch):
    """Every test starts with both flags unset unless it sets them explicitly."""
    monkeypatch.delenv("ITS_SC_ANSWER_FORCE", raising=False)
    monkeypatch.delenv("ITS_SC_TIER_VOTING", raising=False)


# ---------------------------------------------------------------------------
# _resolve_tier_voting
# ---------------------------------------------------------------------------


class TestResolveTierVoting:
    def test_unset_and_no_forcing_is_off(self, monkeypatch):
        # No forcing, no explicit tier flag -> tiering off (nothing to gate).
        monkeypatch.delenv("ITS_SC_ANSWER_FORCE", raising=False)
        monkeypatch.delenv("ITS_SC_TIER_VOTING", raising=False)
        assert _resolve_tier_voting() is False

    def test_unset_stays_off_even_when_forcing_active(self, monkeypatch):
        # Default-OFF opt-in (Option A): forcing ON but the tier flag unset must
        # leave tiering OFF, so forced samples vote alongside natural ones exactly
        # as in exp-15 -- this is what preserves the exp-15 tie test unchanged.
        monkeypatch.setenv("ITS_SC_ANSWER_FORCE", "1")
        monkeypatch.delenv("ITS_SC_TIER_VOTING", raising=False)
        assert _resolve_tier_voting() is False

    def test_explicit_zero_disables_even_when_forcing_active(self, monkeypatch):
        monkeypatch.setenv("ITS_SC_ANSWER_FORCE", "1")
        monkeypatch.setenv("ITS_SC_TIER_VOTING", "0")
        assert _resolve_tier_voting() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "  On  "])
    def test_explicit_truthy_enables(self, monkeypatch, value):
        monkeypatch.delenv("ITS_SC_ANSWER_FORCE", raising=False)
        monkeypatch.setenv("ITS_SC_TIER_VOTING", value)
        assert _resolve_tier_voting() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "garbage"])
    def test_explicit_falsy_disables(self, monkeypatch, value):
        monkeypatch.setenv("ITS_SC_ANSWER_FORCE", "1")
        monkeypatch.setenv("ITS_SC_TIER_VOTING", value)
        assert _resolve_tier_voting() is False


# ---------------------------------------------------------------------------
# Natural-first gate FIRES (AIME FORCED_ANSWER_DILUTION case)
# ---------------------------------------------------------------------------


class TestNaturalFirstGate:
    @pytest.mark.asyncio
    async def test_natural_box_beats_forced_wrong_guesses(self, monkeypatch):
        # 1 correct natural \boxed{42} vs 3 box-absent samples forced to a wrong
        # \boxed{1}. Non-tiered plurality would let the 3 forced guesses win
        # (dilution); tiering drops them so the lone natural box carries the vote.
        monkeypatch.setenv("ITS_SC_ANSWER_FORCE", "1")
        monkeypatch.setenv("ITS_SC_TIER_VOTING", "1")  # default-OFF: opt in explicitly
        orch = RecordingOrchestrator(
            [dict(NATURAL_CORRECT), dict(BOXLESS), dict(BOXLESS), dict(BOXLESS)]
        )
        lm = ContinuationMockLM(continuation="1")
        sc = SelfConsistency(orchestrator=orch)

        result = await sc.ainfer(
            lm, "What is 6 * 7?", budget=4, return_response_only=False
        )

        # All three box-absent samples were still forced (identical LM request
        # pattern as exp-15); tiering only changes vote-time selection.
        assert len(lm.single_calls) == 3
        # The natural correct answer wins, not the 3-vote forced guess.
        assert result.selected_index == 0
        assert "\\boxed{42}" in result.the_one["content"]
        assert result.the_one.get("_forced") is None

    @pytest.mark.asyncio
    async def test_forcing_still_fires_for_all_box_absent_samples(self, monkeypatch):
        # Replay guardrail: tiering must NOT short-circuit forcing. Every
        # box-absent sample still gets its continuation call regardless of the
        # eventual vote-time gating.
        monkeypatch.setenv("ITS_SC_ANSWER_FORCE", "1")
        monkeypatch.setenv("ITS_SC_TIER_VOTING", "1")  # default-OFF: opt in explicitly
        orch = RecordingOrchestrator(
            [dict(NATURAL_CORRECT), dict(BOXLESS), dict(BOXLESS)]
        )
        lm = ContinuationMockLM(continuation="1")
        sc = SelfConsistency(orchestrator=orch)

        result = await sc.ainfer(
            lm, "What is 6 * 7?", budget=3, return_response_only=False
        )

        # Two continuations issued (one per box-absent sample); the forced
        # responses are marked and remain in the result set (only gated at vote).
        assert len(lm.single_calls) == 2
        forced = [r for r in result.responses if r.get("_forced")]
        assert len(forced) == 2


# ---------------------------------------------------------------------------
# Forced-fallback FIRES (GPQA all-empty rescue case)
# ---------------------------------------------------------------------------


class TestForcedFallback:
    @pytest.mark.asyncio
    async def test_all_box_absent_uses_forced_plurality(self, monkeypatch):
        # No natural boxed candidate survives -> the natural tier is empty, so the
        # forced tier is consulted (the GPQA rescue). Without forcing these would
        # all project empty and abstain; with forced-fallback a real answer wins.
        monkeypatch.setenv("ITS_SC_ANSWER_FORCE", "1")
        monkeypatch.setenv("ITS_SC_TIER_VOTING", "1")  # default-OFF: opt in explicitly
        orch = RecordingOrchestrator([dict(BOXLESS) for _ in range(4)])
        lm = ContinuationMockLM(continuation="C")
        sc = SelfConsistency(orchestrator=orch)

        result = await sc.ainfer(
            lm, "Pick one:\nA) w\nB) x\nC) y\nD) z", budget=4, return_response_only=False
        )

        # The forced answer is selected -- the fallback fired rather than raising.
        assert "\\boxed{C}" in result.the_one["content"]
        assert result.the_one.get("_forced") is True


# ---------------------------------------------------------------------------
# flag-OFF no-op: no forced samples -> eligible set + ordering unchanged
# ---------------------------------------------------------------------------


class TestNoForcedSamplesNoOp:
    def test_projection_unchanged_when_no_forced_present(self, monkeypatch):
        # With no ``_forced`` markers, enabling tiering must not change which
        # indices are eligible NOR their order (byte-identity guardrail).
        responses = [
            {"role": "assistant", "content": "a \\boxed{1}"},
            {"role": "assistant", "content": "b \\boxed{2}"},
            {"role": "assistant", "content": "c \\boxed{1}"},
        ]
        sc = SelfConsistency()

        monkeypatch.delenv("ITS_SC_ANSWER_FORCE", raising=False)
        monkeypatch.delenv("ITS_SC_TIER_VOTING", raising=False)
        base_idx, base_proj = sc._project_responses([dict(r) for r in responses])

        # Tiering explicitly ON, but there are no forced samples to gate.
        monkeypatch.setenv("ITS_SC_TIER_VOTING", "1")
        tier_idx, tier_proj = sc._project_responses([dict(r) for r in responses])

        assert tier_idx == base_idx == [0, 1, 2]
        assert tier_proj == base_proj


# ---------------------------------------------------------------------------
# TIER=0 reproduces non-tiered exp-15 behavior
# ---------------------------------------------------------------------------


class TestTierDisabledReproducesExp15:
    @pytest.mark.asyncio
    async def test_forced_dilution_stands_when_tiering_disabled(self, monkeypatch):
        # Same AIME scenario as the natural-first test, but with tiering disabled.
        # The 3 forced \boxed{1} guesses out-vote the lone natural \boxed{42} --
        # the exp-15 FORCED_ANSWER_DILUTION regression, reproduced exactly.
        monkeypatch.setenv("ITS_SC_ANSWER_FORCE", "1")
        monkeypatch.setenv("ITS_SC_TIER_VOTING", "0")
        orch = RecordingOrchestrator(
            [dict(NATURAL_CORRECT), dict(BOXLESS), dict(BOXLESS), dict(BOXLESS)]
        )
        lm = ContinuationMockLM(continuation="1")
        sc = SelfConsistency(orchestrator=orch)

        result = await sc.ainfer(
            lm, "What is 6 * 7?", budget=4, return_response_only=False
        )

        # Forced majority wins under the non-tiered path.
        assert "\\boxed{1}" in result.the_one["content"]
        assert result.the_one.get("_forced") is True


# ---------------------------------------------------------------------------
# Tier telemetry (observability only)
# ---------------------------------------------------------------------------


class TestTierTelemetry:
    @pytest.mark.asyncio
    async def test_tier_stats_logged_when_gate_fires(self, monkeypatch, caplog):
        monkeypatch.setenv("ITS_SC_ANSWER_FORCE", "1")
        monkeypatch.setenv("ITS_SC_TIER_VOTING", "1")  # default-OFF: opt in explicitly
        orch = RecordingOrchestrator(
            [dict(NATURAL_CORRECT), dict(BOXLESS), dict(BOXLESS), dict(BOXLESS)]
        )
        lm = ContinuationMockLM(continuation="1")
        sc = SelfConsistency(orchestrator=orch)

        with caplog.at_level(logging.INFO):
            await sc.ainfer(lm, "What is 6 * 7?", budget=4)

        assert "n_natural=1" in caplog.text
        assert "n_forced=3" in caplog.text
        assert "tier_fired=True" in caplog.text

    @pytest.mark.asyncio
    async def test_no_tier_log_when_tiering_disabled(self, monkeypatch, caplog):
        monkeypatch.setenv("ITS_SC_ANSWER_FORCE", "1")
        monkeypatch.setenv("ITS_SC_TIER_VOTING", "0")
        orch = RecordingOrchestrator([dict(NATURAL_CORRECT), dict(BOXLESS)])
        lm = ContinuationMockLM(continuation="1")
        sc = SelfConsistency(orchestrator=orch)

        with caplog.at_level(logging.INFO):
            await sc.ainfer(lm, "What is 6 * 7?", budget=2)

        assert "tiered voting" not in caplog.text
