# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the H1 per-sample temperature ladder (ITS_SC_TEMP_LADDER).

Mock-based, no GPU. Verifies the additive, default-OFF contract:
- Flag unset  => orchestrator.agenerate called with NO temperature (byte-identical
  control payload).
- Flag set    => a list[float] of length ``budget`` passed as ``temperature``.

Also covers the resolver's parsing/cycling rules directly.
"""

from unittest.mock import AsyncMock

import pytest

from its_hub.core.algorithms.self_consistency import (
    _DEFAULT_TEMP_LADDER,
    SelfConsistency,
    _resolve_temp_ladder,
)


def _make_responses(n: int) -> list[dict]:
    """A batch of ``n`` plain content responses with a clear majority answer."""
    return [{"role": "assistant", "content": "The answer is \\boxed{42}"} for _ in range(n)]


@pytest.mark.asyncio
async def test_flag_unset_passes_no_temperature(monkeypatch):
    """Default/control arm: agenerate is called WITHOUT any temperature kwarg."""
    monkeypatch.delenv("ITS_SC_TEMP_LADDER", raising=False)

    budget = 4
    orch = AsyncMock()
    orch.agenerate = AsyncMock(return_value=_make_responses(budget))
    sc = SelfConsistency(orchestrator=orch)

    await sc.ainfer(AsyncMock(), "test prompt", budget=budget)

    assert orch.agenerate.await_count == 1
    _, kwargs = orch.agenerate.await_args
    # OFF path must not pass a temperature argument at all (defaults to None).
    assert "temperature" not in kwargs


@pytest.mark.asyncio
async def test_flag_set_passes_temperature_list_of_len_budget(monkeypatch):
    """Treatment arm: a list[float] of length ``budget`` is passed as temperature."""
    monkeypatch.setenv("ITS_SC_TEMP_LADDER", "0.3,0.6,0.9,1.2")

    budget = 4
    orch = AsyncMock()
    orch.agenerate = AsyncMock(return_value=_make_responses(budget))
    sc = SelfConsistency(orchestrator=orch)

    await sc.ainfer(AsyncMock(), "test prompt", budget=budget)

    assert orch.agenerate.await_count == 1
    _, kwargs = orch.agenerate.await_args
    assert "temperature" in kwargs
    temperature = kwargs["temperature"]
    assert isinstance(temperature, list)
    assert len(temperature) == budget
    assert all(isinstance(t, float) for t in temperature)
    assert temperature == [0.3, 0.6, 0.9, 1.2]


def test_resolver_unset_returns_none(monkeypatch):
    monkeypatch.delenv("ITS_SC_TEMP_LADDER", raising=False)
    assert _resolve_temp_ladder(4) is None


@pytest.mark.parametrize("falsey", ["", "0", "false", "off", "no", "  "])
def test_resolver_falsey_returns_none(monkeypatch, falsey):
    monkeypatch.setenv("ITS_SC_TEMP_LADDER", falsey)
    assert _resolve_temp_ladder(4) is None


@pytest.mark.parametrize("sentinel", ["1", "true", "on", "yes", "TRUE"])
def test_resolver_truthy_sentinel_uses_default_ladder(monkeypatch, sentinel):
    monkeypatch.setenv("ITS_SC_TEMP_LADDER", sentinel)
    assert _resolve_temp_ladder(4) == list(_DEFAULT_TEMP_LADDER)


def test_resolver_explicit_list_parsed(monkeypatch):
    monkeypatch.setenv("ITS_SC_TEMP_LADDER", "0.2, 0.5, 0.8")
    assert _resolve_temp_ladder(3) == [0.2, 0.5, 0.8]


def test_resolver_cycles_when_budget_exceeds_ladder(monkeypatch):
    monkeypatch.setenv("ITS_SC_TEMP_LADDER", "0.3,0.6")
    # budget=5 cycles: [0.3, 0.6, 0.3, 0.6, 0.3]
    assert _resolve_temp_ladder(5) == [0.3, 0.6, 0.3, 0.6, 0.3]


def test_resolver_truncates_when_budget_below_ladder(monkeypatch):
    monkeypatch.setenv("ITS_SC_TEMP_LADDER", "0.3,0.6,0.9,1.2")
    assert _resolve_temp_ladder(2) == [0.3, 0.6]


def test_resolver_ignores_non_numeric_tokens(monkeypatch):
    monkeypatch.setenv("ITS_SC_TEMP_LADDER", "0.3,abc,0.9")
    assert _resolve_temp_ladder(2) == [0.3, 0.9]


def test_resolver_non_positive_budget_returns_empty(monkeypatch):
    monkeypatch.setenv("ITS_SC_TEMP_LADDER", "0.3,0.6")
    assert _resolve_temp_ladder(0) == []
