# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the H1 (exp-19) low-temp-heavy default temperature ladder.

Additive, mock-free, no GPU. Covers the retuned low-temp-heavy default ladder
``(0.3, 0.3, 0.6, 0.9)`` resolved through ``_resolve_temp_ladder``. Allocation is
the deterministic round-robin cycling inherited from exp-18 (the front-loaded
fractional allocator was deferred to an additive follow-up so the locked exp-18
round-robin contract stays intact).

Contract (default ladder):
- The resolved ladder is cycled round-robin to length ``budget``.
- budget=4 -> [0.3, 0.3, 0.6, 0.9]
- budget=8 -> [0.3, 0.3, 0.6, 0.9, 0.3, 0.3, 0.6, 0.9]
- budget=1 -> [0.3]
- budget<=0 -> []
- Flag unset -> None (byte-identical OFF path preserved).
"""

import pytest

from its_hub.core.algorithms.self_consistency import (
    _DEFAULT_TEMP_LADDER,
    _resolve_temp_ladder,
)


def test_default_ladder_is_low_temp_heavy():
    """The exp-19 default duplicates the 0.3 rescue rung and drops the 1.2 rung."""
    assert _DEFAULT_TEMP_LADDER == (0.3, 0.3, 0.6, 0.9)


@pytest.mark.parametrize(
    "budget,expected",
    [
        (4, [0.3, 0.3, 0.6, 0.9]),
        (8, [0.3, 0.3, 0.6, 0.9, 0.3, 0.3, 0.6, 0.9]),
        (1, [0.3]),
    ],
)
def test_round_robin_allocation_default_ladder(monkeypatch, budget, expected):
    """Round-robin cycling over the default ladder (sentinel enables)."""
    monkeypatch.setenv("ITS_SC_TEMP_LADDER", "1")
    assert _resolve_temp_ladder(budget) == expected


@pytest.mark.parametrize("budget", [0, -1, -5])
def test_non_positive_budget_returns_empty(monkeypatch, budget):
    monkeypatch.setenv("ITS_SC_TEMP_LADDER", "1")
    assert _resolve_temp_ladder(budget) == []


def test_flag_unset_returns_none(monkeypatch):
    """OFF path preserved: unset flag -> None -> ainfer passes no temperature."""
    monkeypatch.delenv("ITS_SC_TEMP_LADDER", raising=False)
    assert _resolve_temp_ladder(4) is None


def test_resolved_ladder_has_length_budget(monkeypatch):
    """The resolved ladder is always exactly ``budget`` long when enabled."""
    monkeypatch.setenv("ITS_SC_TEMP_LADDER", "1")
    for budget in (2, 4, 6, 8, 10):
        ladder = _resolve_temp_ladder(budget)
        assert len(ladder) == budget


def test_explicit_override_cycles_round_robin(monkeypatch):
    """An explicit comma-separated ladder is round-robined, not front-loaded."""
    monkeypatch.setenv("ITS_SC_TEMP_LADDER", "0.2,0.5,0.8,1.1")
    assert _resolve_temp_ladder(4) == [0.2, 0.5, 0.8, 1.1]


def test_single_rung_ladder_all_low(monkeypatch):
    """A single-rung ladder puts every sample on that rung."""
    monkeypatch.setenv("ITS_SC_TEMP_LADDER", "0.3")
    assert _resolve_temp_ladder(4) == [0.3, 0.3, 0.3, 0.3]
