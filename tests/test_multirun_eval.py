"""Unit tests for the H3 multi-run eval extensions in eval/score.py.

Covers, without a live model endpoint (the benchmark sweep is mocked):

  (a) ITS_EVAL_RUNS=1 (default) is behavior/score-identical to the pre-H3 path;
  (b) ITS_EVAL_RUNS=N runs the slice N times with distinct DERIVED seeds and
      reports mean ± stddev;
  (c) stddev > noise floor sets the "signal vs sampling" flag in details but does
      NOT change the score / weight / passed gate;
  (d) hardware/version stamping is present and degrades to 'unknown'/None when a
      package is missing or no GPU is available;
  (e) determinism: the same master seed always yields the same derived per-run
      seed sequence.

The subprocess benchmark sweep is mocked (via `_run_benchmarks_once`) so these
tests never touch the GPU endpoint.
"""

import importlib.util
import math
import os
import sys

import pytest

# Load eval/score.py by path ("eval" is a builtin name and not a package).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCORE_PATH = os.path.join(_REPO_ROOT, "eval", "score.py")
_spec = importlib.util.spec_from_file_location("its_eval_score_multirun", _SCORE_PATH)
score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score)

_CONTRACT_KEYS = {"name", "score", "weight", "passed", "details"}


@pytest.fixture
def endpoint_env(monkeypatch):
    """Endpoint reachable, budget valid, no seed / run-count set by default."""
    monkeypatch.setenv("ITS_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.setenv("ITS_BUDGET", "4")
    monkeypatch.delenv("ITS_EVAL_RUNS", raising=False)
    monkeypatch.delenv("ITS_SEED", raising=False)
    monkeypatch.setattr(score, "_endpoint_reachable", lambda ep: True)


def _install_sweep(monkeypatch, per_seed_fn, seen_seeds):
    """Replace the real benchmark sweep with a canned, seed-aware stub."""

    def fake(**kwargs):
        seen_seeds.append(kwargs["seed"])
        return dict(per_seed_fn(kwargs["seed"])), []

    monkeypatch.setattr(score, "_run_benchmarks_once", fake)


def _constant(accs):
    return lambda _seed: accs


# --- (e) deterministic seed derivation --------------------------------------


def test_derive_run_seeds_is_master_plus_index():
    assert score._derive_run_seeds(7, 4) == [7, 8, 9, 10]


def test_derive_run_seeds_zero_master():
    assert score._derive_run_seeds(0, 3) == [0, 1, 2]


def test_derive_run_seeds_deterministic_across_calls():
    assert score._derive_run_seeds(100, 5) == score._derive_run_seeds(100, 5)


def test_derive_run_seeds_single():
    assert score._derive_run_seeds(42, 1) == [42]


# --- sample stddev helper ----------------------------------------------------


def test_sample_stddev_single_value_is_zero():
    assert score._sample_stddev([0.5]) == 0.0


def test_sample_stddev_empty_is_zero():
    assert score._sample_stddev([]) == 0.0


def test_sample_stddev_two_values():
    # ddof=1: sqrt(((2-3)^2 + (4-3)^2) / (2-1)) = sqrt(2)
    assert math.isclose(score._sample_stddev([2.0, 4.0]), math.sqrt(2.0))


# --- (a) N=1 default is behavior/score-identical -----------------------------


def test_runs_env_defaults_to_one(monkeypatch):
    monkeypatch.delenv("ITS_EVAL_RUNS", raising=False)
    assert score._get_eval_runs() == 1


def test_runs_env_invalid_degrades_to_one(monkeypatch):
    monkeypatch.setenv("ITS_EVAL_RUNS", "not-a-number")
    assert score._get_eval_runs() == 1
    monkeypatch.setenv("ITS_EVAL_RUNS", "0")
    assert score._get_eval_runs() == 1


def test_default_single_run_score_identical(endpoint_env, monkeypatch):
    accs = {"math500": 0.5, "aime-2024": 0.4, "gpqa-diamond": 0.6}
    seen: list = []
    _install_sweep(monkeypatch, _constant(accs), seen)

    result = score.eval_accuracy()

    # Pre-H3 composite: 0.5 * mean(0.5, 0.4) + 0.5 * 0.6 = 0.525
    assert result["score"] == pytest.approx(0.525)
    assert result["weight"] == score._ACCURACY_WEIGHT
    assert result["passed"] is True
    assert result["name"] == "accuracy"
    assert set(result) == _CONTRACT_KEYS
    # Original details core preserved.
    assert "budget=4" in result["details"]
    assert "-> score=0.5250" in result["details"]
    # At N=1 no stddev / multi-run flag is emitted.
    assert "runs=" not in result["details"]
    assert "stddev" not in result["details"]
    assert "signal vs sampling" not in result["details"]


def test_default_single_run_uses_no_injected_seed(endpoint_env, monkeypatch):
    # N=1 default must launch the sweep exactly as before: seed=None (no env
    # injection), so behavior is byte-identical to pre-H3.
    seen: list = []
    _install_sweep(
        monkeypatch,
        _constant({"math500": 0.5, "aime-2024": 0.4, "gpqa-diamond": 0.6}),
        seen,
    )
    score.eval_accuracy()
    assert seen == [None]


# --- (b) N runs, distinct derived seeds, mean ± stddev -----------------------


def test_multi_run_invokes_sweep_n_times_with_derived_seeds(endpoint_env, monkeypatch):
    monkeypatch.setenv("ITS_EVAL_RUNS", "3")
    seen: list = []
    _install_sweep(
        monkeypatch,
        _constant({"math500": 0.5, "aime-2024": 0.4, "gpqa-diamond": 0.6}),
        seen,
    )
    result = score.eval_accuracy()
    # Default master seed 0 -> [0, 1, 2].
    assert seen == [0, 1, 2]
    assert "runs=3" in result["details"]
    assert "stddev[" in result["details"]


def test_multi_run_master_seed_derives_sequence(endpoint_env, monkeypatch):
    monkeypatch.setenv("ITS_EVAL_RUNS", "3")
    monkeypatch.setenv("ITS_SEED", "100")
    seen: list = []
    _install_sweep(
        monkeypatch,
        _constant({"math500": 0.5, "aime-2024": 0.4, "gpqa-diamond": 0.6}),
        seen,
    )
    score.eval_accuracy()
    assert seen == [100, 101, 102]


def test_multi_run_reports_mean(endpoint_env, monkeypatch):
    monkeypatch.setenv("ITS_EVAL_RUNS", "3")
    # math500 varies across runs; mean is 0.5.
    per_seed = {
        0: {"math500": 0.4, "aime-2024": 0.4, "gpqa-diamond": 0.6},
        1: {"math500": 0.5, "aime-2024": 0.4, "gpqa-diamond": 0.6},
        2: {"math500": 0.6, "aime-2024": 0.4, "gpqa-diamond": 0.6},
    }
    seen: list = []
    _install_sweep(monkeypatch, lambda s: per_seed[s], seen)
    result = score.eval_accuracy()
    # math_acc = mean(mean_math500=0.5, aime=0.4) = 0.45; score = 0.5*0.45 + 0.5*0.6
    assert result["score"] == pytest.approx(0.525)


# --- (c) stddev flag is report-only ------------------------------------------


def test_stddev_flag_fires_above_noise_floor(endpoint_env, monkeypatch):
    monkeypatch.setenv("ITS_EVAL_RUNS", "3")
    # math500 spread 0.4/0.5/0.6 -> stddev 0.1 > 0.04 floor.
    per_seed = {
        0: {"math500": 0.4, "aime-2024": 0.4, "gpqa-diamond": 0.6},
        1: {"math500": 0.5, "aime-2024": 0.4, "gpqa-diamond": 0.6},
        2: {"math500": 0.6, "aime-2024": 0.4, "gpqa-diamond": 0.6},
    }
    seen: list = []
    _install_sweep(monkeypatch, lambda s: per_seed[s], seen)
    result = score.eval_accuracy()
    assert "signal vs sampling: math500 stddev=" in result["details"]
    assert "noise floor" in result["details"]


def test_stddev_flag_does_not_change_score_weight_pass(endpoint_env, monkeypatch):
    monkeypatch.setenv("ITS_EVAL_RUNS", "3")
    # Same MEANS as a zero-variance run, but with spread that trips the flag.
    varying = {
        0: {"math500": 0.4, "aime-2024": 0.4, "gpqa-diamond": 0.6},
        1: {"math500": 0.5, "aime-2024": 0.4, "gpqa-diamond": 0.6},
        2: {"math500": 0.6, "aime-2024": 0.4, "gpqa-diamond": 0.6},
    }
    seen_v: list = []
    _install_sweep(monkeypatch, lambda s: varying[s], seen_v)
    flagged = score.eval_accuracy()

    # A zero-variance run with identical means (0.5, 0.4, 0.6).
    seen_c: list = []
    _install_sweep(
        monkeypatch,
        _constant({"math500": 0.5, "aime-2024": 0.4, "gpqa-diamond": 0.6}),
        seen_c,
    )
    clean = score.eval_accuracy()

    assert "signal vs sampling" in flagged["details"]
    assert "signal vs sampling" not in clean["details"]
    # The flag is report-only: score / weight / passed are unaffected.
    assert flagged["score"] == clean["score"]
    assert flagged["weight"] == clean["weight"]
    assert flagged["passed"] == clean["passed"]


def test_no_flag_when_stddev_below_floor(endpoint_env, monkeypatch):
    monkeypatch.setenv("ITS_EVAL_RUNS", "3")
    # Tiny spread 0.50/0.51/0.52 -> stddev 0.01 < 0.04.
    per_seed = {
        0: {"math500": 0.50, "aime-2024": 0.4, "gpqa-diamond": 0.6},
        1: {"math500": 0.51, "aime-2024": 0.4, "gpqa-diamond": 0.6},
        2: {"math500": 0.52, "aime-2024": 0.4, "gpqa-diamond": 0.6},
    }
    seen: list = []
    _install_sweep(monkeypatch, lambda s: per_seed[s], seen)
    result = score.eval_accuracy()
    assert "signal vs sampling" not in result["details"]
    assert "runs=3" in result["details"]


# --- (d) hardware/version stamping -------------------------------------------


def test_hw_metadata_has_all_keys():
    meta = score._hw_version_metadata()
    for key in (
        "vllm_version",
        "torch_version",
        "cuda_version",
        "gpu_model",
        "vllm_batch_invariant",
    ):
        assert key in meta


def test_hw_metadata_reads_batch_invariant_flag(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    assert score._hw_version_metadata()["vllm_batch_invariant"] == "1"
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    assert score._hw_version_metadata()["vllm_batch_invariant"] is None


def test_hw_metadata_degrades_without_packages(monkeypatch):
    # Simulate torch/vllm being absent: `import torch` raises ImportError when
    # sys.modules[name] is None.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "vllm", None)
    meta = score._hw_version_metadata()
    assert meta["torch_version"] == "unknown"
    assert meta["vllm_version"] == "unknown"
    assert meta["gpu_model"] is None
    assert meta["cuda_version"] is None


def test_hw_metadata_never_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "vllm", None)
    # Must not raise regardless of environment.
    score._hw_version_metadata()


def test_details_include_hw_stamp(endpoint_env, monkeypatch):
    seen: list = []
    _install_sweep(
        monkeypatch,
        _constant({"math500": 0.5, "aime-2024": 0.4, "gpqa-diamond": 0.6}),
        seen,
    )
    result = score.eval_accuracy()
    assert "hw:" in result["details"]
    assert "torch=" in result["details"]
    assert "batch_invariant=" in result["details"]


# --- seed threading at the subprocess boundary -------------------------------


def test_run_benchmarks_once_injects_seed_env(monkeypatch):
    captured = []

    def fake_run(cmd, **kwargs):
        captured.append(kwargs)

        class _R:
            stdout = ""
            stderr = ""
            returncode = 0

        return _R()

    monkeypatch.setattr(score.subprocess, "run", fake_run)

    score._run_benchmarks_once(
        endpoint="http://x",
        model="m",
        api_key="k",
        alg="self-consistency",
        budget=4,
        timeout_s=10,
        tokens_per_step=None,
        repo_root=_REPO_ROOT,
        bench_script="benchmark.py",
        seed=42,
    )
    # Every subprocess call carries ITS_SEED=42 in an explicit env override.
    assert captured
    for kw in captured:
        assert "env" in kw
        assert kw["env"]["ITS_SEED"] == "42"


def test_run_benchmarks_once_no_seed_no_env_override(monkeypatch):
    captured = []

    def fake_run(cmd, **kwargs):
        captured.append(kwargs)

        class _R:
            stdout = ""
            stderr = ""
            returncode = 0

        return _R()

    monkeypatch.setattr(score.subprocess, "run", fake_run)

    score._run_benchmarks_once(
        endpoint="http://x",
        model="m",
        api_key="k",
        alg="self-consistency",
        budget=4,
        timeout_s=10,
        tokens_per_step=None,
        repo_root=_REPO_ROOT,
        bench_script="benchmark.py",
        seed=None,
    )
    # seed=None -> launched exactly as pre-H3: no env override.
    assert captured
    for kw in captured:
        assert "env" not in kw
