# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the H1 (exp-19) front-loaded fractional temperature allocation.

Additive, mock-free, no GPU. Covers the NEW budget-invariant allocator inside
``_resolve_temp_ladder`` (replacing the previous round-robin), plus the retuned
low-temp-heavy default ladder ``(0.3, 0.3, 0.6, 0.9)``.

Contract (default ladder):
- ~50% of samples pinned to the LOWEST rung at ANY budget, remainder round-robined
  over the higher (non-low) rungs.
- budget=4 -> [0.3, 0.3, 0.6, 0.9]
- budget=8 -> [0.3, 0.3, 0.3, 0.3, 0.6, 0.9, 0.6, 0.9]
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
        (8, [0.3, 0.3, 0.3, 0.3, 0.6, 0.9, 0.6, 0.9]),
        (1, [0.3]),
    ],
)
def test_fractional_allocation_default_ladder(monkeypatch, budget, expected):
    """Front-loaded fractional allocation over the default ladder (sentinel enables)."""
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


def test_low_temp_density_is_budget_invariant(monkeypatch):
    """~50% of samples land on the lowest rung regardless of budget."""
    monkeypatch.setenv("ITS_SC_TEMP_LADDER", "1")
    for budget in (2, 4, 6, 8, 10):
        ladder = _resolve_temp_ladder(budget)
        assert len(ladder) == budget
        n_low = sum(1 for t in ladder if t == _DEFAULT_TEMP_LADDER[0])
        assert n_low == max(1, budget // 2)


def test_explicit_override_flows_through_allocator(monkeypatch):
    """An explicit comma-separated ladder is front-loaded, not round-robined."""
    monkeypatch.setenv("ITS_SC_TEMP_LADDER", "0.2,0.5,0.8,1.1")
    # low=0.2 (n_low=2), higher=[0.5,0.8,1.1] round-robined over 2 residual slots.
    assert _resolve_temp_ladder(4) == [0.2, 0.2, 0.5, 0.8]


def test_single_rung_ladder_all_low(monkeypatch):
    """A single-rung ladder puts every sample on the low rung (no higher rungs)."""
    monkeypatch.setenv("ITS_SC_TEMP_LADDER", "0.3")
    assert _resolve_temp_ladder(4) == [0.3, 0.3, 0.3, 0.3]
