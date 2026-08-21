"""Tests for the H4.9 QA-iteration-1 reproducibility fixes (exp_id=12).

Two blocking defects from adversarial QA are covered here, both hermetic
(no network, no GPU):

  DEFECT 1 — ``ITS_SEED`` did not seed LM sampling. ``_seed_global_random_from_env``
    only seeded Python's global ``random`` (the plurality tie-break); the LM was
    built with no ``seed=``, so vLLM drew different tokens every run and GPQA
    swung run-to-run. The fix threads the resolved seed into
    ``OpenAICompatibleLanguageModel(seed=...)`` → vLLM ``SamplingParams(seed=...)``
    for the LIVE (``off`` cache mode) path, while leaving the seed OFF when
    ``ITS_SEED`` is unset (regression guard) or the cache is recording/replaying
    (so the H1 record/replay path stays seed-invariant / byte-identical).

  DEFECT 2 — the new GPQA MCQ prompt was inert under default ``datasets`` caching
    because a stale ``datasets.map`` fingerprint cache served the OLD prompt. The
    fix passes ``load_from_cache_file=False`` to the GPQA normalize map so the
    current "Answer: X AND \\boxed{X}" prompt reaches the model regardless of
    on-disk cache state.

New tests only — no existing tests are modified (Sacred Rule 1).
"""

import importlib.util
import os
import random

import datasets
import pytest

# Load benchmarking/benchmark.py by path (it is not an importable package),
# mirroring tests/test_gpqa_mcq_voting.py.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BENCH_PATH = os.path.join(_REPO_ROOT, "benchmarking", "benchmark.py")
_spec = importlib.util.spec_from_file_location("its_benchmark_qa1", _BENCH_PATH)
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)

GPQA = benchmark.BenchmarkDataset.GPQA_DIAMOND
_seed_global_random_from_env = benchmark._seed_global_random_from_env
_lm_sampling_seed = benchmark._lm_sampling_seed
load_benchmark_dataset = benchmark.load_benchmark_dataset

# The new MCQ instruction that DEFECT 2 must make reach the model. Kept in one
# place so a future prompt edit fails loudly here rather than silently.
_NEW_MCQ_MARKERS = ("Answer: X", "\\boxed{X}")


@pytest.fixture(autouse=True)
def _clean_seed_env(monkeypatch):
    """Each test starts with ITS_SEED / ITS_CACHE_MODE unset."""
    monkeypatch.delenv("ITS_SEED", raising=False)
    monkeypatch.delenv("ITS_CACHE_MODE", raising=False)


# --- DEFECT 1: ITS_SEED threads into the LM sampling seed --------------------


class TestSamplingSeedResolution:
    def test_seed_set_is_forwarded(self, monkeypatch):
        # ITS_SEED set + default (off) cache mode -> the resolved seed is
        # forwarded to the LM sampling seed verbatim.
        monkeypatch.setenv("ITS_SEED", "12345")
        resolved = _seed_global_random_from_env()
        assert resolved == 12345
        assert _lm_sampling_seed(resolved) == 12345

    def test_seed_zero_is_forwarded(self, monkeypatch):
        # Regression guard for the falsy-seed bug: seed 0 is meaningful and must
        # be forwarded, not dropped.
        monkeypatch.setenv("ITS_SEED", "0")
        resolved = _seed_global_random_from_env()
        assert resolved == 0
        assert _lm_sampling_seed(resolved) == 0

    def test_seed_unset_forwards_nothing(self):
        # ITS_SEED unset -> resolver returns None -> no sampling seed forwarded.
        assert _seed_global_random_from_env() is None
        assert _lm_sampling_seed(None) is None

    def test_seed_invalid_forwards_nothing(self, monkeypatch):
        monkeypatch.setenv("ITS_SEED", "not-an-int")
        assert _seed_global_random_from_env() is None
        assert _lm_sampling_seed(None) is None

    def test_seed_unset_leaves_rng_untouched(self, monkeypatch):
        # REGRESSION GUARD: with ITS_SEED unset the global RNG must NOT be
        # re-seeded, so the MATH500/AIME/default path stays byte-identical to
        # baseline. Snapshot the RNG state, call the resolver, assert unchanged.
        random.seed()  # move off any deterministic state a prior test set
        before = random.getstate()
        assert _seed_global_random_from_env() is None
        assert random.getstate() == before

    def test_replay_mode_suppresses_sampling_seed(self, monkeypatch):
        # In record/replay the cached draw is the determinism source; a sampling
        # seed would perturb the (seed-sensitive) H1 cache key and miss caches
        # recorded without it. So even with ITS_SEED set, no seed is forwarded.
        monkeypatch.setenv("ITS_SEED", "12345")
        resolved = _seed_global_random_from_env()
        assert resolved == 12345
        for mode in ("replay", "record"):
            monkeypatch.setenv("ITS_CACHE_MODE", mode)
            assert _lm_sampling_seed(resolved) is None

    def test_off_mode_forwards_sampling_seed(self, monkeypatch):
        monkeypatch.setenv("ITS_SEED", "777")
        monkeypatch.setenv("ITS_CACHE_MODE", "off")
        resolved = _seed_global_random_from_env()
        assert _lm_sampling_seed(resolved) == 777

    def test_forwarded_seed_reaches_lm_sampling_params(self):
        # End-to-end: the value we forward lands in the request payload as the
        # ``seed`` field (vLLM maps this to SamplingParams(seed=...)).
        from its_hub import OpenAICompatibleLanguageModel
        from its_hub.api.types import ChatMessage

        lm = OpenAICompatibleLanguageModel(
            endpoint="http://127.0.0.1:9/v1",
            api_key="k",
            model_name="m",
            temperature=0.7,
            seed=_lm_sampling_seed(12345),
        )
        req = lm._prepare_request_data([ChatMessage(role="user", content="hi")])
        assert req["seed"] == 12345

    def test_unforwarded_seed_leaves_payload_unseeded(self):
        # ITS_SEED unset -> seed=None -> no ``seed`` key in the request payload
        # -> byte-identical to the pre-seed request.
        from its_hub import OpenAICompatibleLanguageModel
        from its_hub.api.types import ChatMessage

        lm = OpenAICompatibleLanguageModel(
            endpoint="http://127.0.0.1:9/v1",
            api_key="k",
            model_name="m",
            temperature=0.7,
            seed=_lm_sampling_seed(None),
        )
        req = lm._prepare_request_data([ChatMessage(role="user", content="hi")])
        assert "seed" not in req


# --- DEFECT 2: the new MCQ prompt reaches the model under default caching -----


def _fake_gpqa_raw():
    """A minimal in-memory GPQA-Diamond raw split (real ``datasets.Dataset``)."""
    return datasets.Dataset.from_dict(
        {
            "Question": ["What is the answer?"],
            "Correct Answer": ["the correct one"],
            "Incorrect Answer 1": ["wrong 1"],
            "Incorrect Answer 2": ["wrong 2"],
            "Incorrect Answer 3": ["wrong 3"],
            "Record ID": ["rec-xyz"],
        }
    )


class TestGpqaPromptEffectiveUnderDefaultCaching:
    def test_loaded_prompt_contains_new_mcq_instruction(self, monkeypatch):
        # Run the REAL load_benchmark_dataset(GPQA) path under DEFAULT datasets
        # caching (caching NOT disabled here) against an in-memory raw split.
        monkeypatch.setattr(
            datasets, "load_dataset", lambda *a, **k: {"train": _fake_gpqa_raw()}
        )
        ds = load_benchmark_dataset(GPQA)
        problem = ds[0]["problem"]
        for marker in _NEW_MCQ_MARKERS:
            assert marker in problem, f"missing {marker!r} in loaded GPQA prompt"

    def test_prompt_is_stable_across_repeated_loads(self, monkeypatch):
        # DEFECT 2 was a *warm-cache* hazard: the first load populated a stale
        # map cache and later loads served the OLD prompt. Load twice and assert
        # BOTH carry the new prompt (load_from_cache_file=False re-runs the map).
        monkeypatch.setattr(
            datasets, "load_dataset", lambda *a, **k: {"train": _fake_gpqa_raw()}
        )
        first = load_benchmark_dataset(GPQA)[0]["problem"]
        second = load_benchmark_dataset(GPQA)[0]["problem"]
        assert first == second
        for marker in _NEW_MCQ_MARKERS:
            assert marker in second
