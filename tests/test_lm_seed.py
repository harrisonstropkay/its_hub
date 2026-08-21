"""Tests for seed-control forwarding to vLLM SamplingParams(seed) (H2, exp_id=8).

Covers the backward-compatibility contract (seed=None -> byte-identical request
payload AND byte-identical H1 cache key), cache-key discrimination across seeds,
and a full record/replay round-trip with a seed set.
"""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from its_hub.api.types import ChatMessage
from its_hub.core.lms.openai_lm import (
    OpenAICompatibleLanguageModel,
    _compute_cache_key,
)

# Golden cache key for the canonical request below, captured from the pre-H2
# code (commit 60c23ca). This value is the regression guard: if adding seed
# support ever perturbs the unseeded key, this test fails and the existing
# recorded caches would be silently invalidated.
_GOLDEN_UNSEEDED_KEY = (
    "a1a0a887ceb54c76985e474e94089fb3e7a6ae2015e1c4bc7b0c1747bbbde3e9"
)


def _lm(seed=None):
    return OpenAICompatibleLanguageModel(
        endpoint="http://127.0.0.1:9/v1",
        api_key="k",
        model_name="m",
        temperature=0.7,
        seed=seed,
    )


def _msgs():
    return [ChatMessage(role="user", content="hi")]


class TestSeedRequestPayload:
    def test_seed_none_adds_no_seed_key(self):
        # seed=None (default) must not introduce a "seed" field -> the payload is
        # byte-identical to a client with no seed support.
        req = _lm(seed=None)._prepare_request_data(_msgs())
        assert "seed" not in req

    def test_seed_none_is_default(self):
        # Constructing without passing seed at all == seed=None.
        req = _lm()._prepare_request_data(_msgs())
        assert "seed" not in req

    def test_seed_set_adds_seed_key(self):
        req = _lm(seed=42)._prepare_request_data(_msgs())
        assert req["seed"] == 42

    def test_seed_set_payload_equals_unseeded_plus_seed(self):
        # The ONLY difference a set seed makes to the payload is the added key.
        unseeded = _lm(seed=None)._prepare_request_data(_msgs())
        seeded = _lm(seed=42)._prepare_request_data(_msgs())
        assert seeded.pop("seed") == 42
        assert seeded == unseeded

    def test_seed_stored_on_instance(self):
        assert _lm(seed=None).seed is None
        assert _lm(seed=42).seed == 42


class TestSeedCacheKey:
    def test_seed_none_matches_golden_unseeded_key(self):
        # Regression guard: the unseeded key must equal the pre-H2 golden value.
        req = _lm(seed=None)._prepare_request_data(_msgs())
        assert _compute_cache_key(req, "m") == _GOLDEN_UNSEEDED_KEY

    def test_seed_none_key_equals_absent_seed_key(self):
        # An explicit seed=None and a request with no seed field collapse to the
        # same key (both mean "no seed").
        with_none = _lm(seed=None)._prepare_request_data(_msgs())
        assert _compute_cache_key(with_none, "m") == _GOLDEN_UNSEEDED_KEY

    def test_seed_set_key_differs_from_unseeded(self):
        unseeded = _lm(seed=None)._prepare_request_data(_msgs())
        seeded = _lm(seed=42)._prepare_request_data(_msgs())
        assert _compute_cache_key(seeded, "m") != _compute_cache_key(unseeded, "m")

    def test_distinct_seeds_produce_distinct_keys(self):
        req42 = _lm(seed=42)._prepare_request_data(_msgs())
        req43 = _lm(seed=43)._prepare_request_data(_msgs())
        assert _compute_cache_key(req42, "m") != _compute_cache_key(req43, "m")

    def test_seed_key_is_stable(self):
        r1 = _lm(seed=42)._prepare_request_data(_msgs())
        r2 = _lm(seed=42)._prepare_request_data(_msgs())
        assert _compute_cache_key(r1, "m") == _compute_cache_key(r2, "m")


async def _make_counting_server():
    """Chat-completion stand-in that advances its content on every hit."""
    state = {"hits": 0}

    async def handler(request):
        state["hits"] += 1
        return web.json_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"resp-{state['hits']}",
                        },
                        "logprobs": {"content": [{"token": "x", "logprob": -0.1}]},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    server = TestServer(app)
    await server.start_server()
    return server, state


@pytest.mark.asyncio
async def test_seeded_record_then_replay_roundtrip(monkeypatch, tmp_path):
    """A seeded client records and replays through the H1 cache byte-identically."""
    import its_hub.core.lms.openai_lm as mod

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(mod, "_cache", None)
    monkeypatch.setattr(mod, "_DEFAULT_CACHE_DIR", cache_dir)

    server, state = await _make_counting_server()
    try:
        lm = OpenAICompatibleLanguageModel(
            endpoint=f"http://127.0.0.1:{server.port}/v1",
            api_key="k",
            model_name="m",
            temperature=0.7,
            seed=42,
            max_tries=1,
        )

        monkeypatch.setenv("ITS_CACHE_MODE", "record")
        rec = await lm.agenerate_single(_msgs())
        assert state["hits"] == 1
        assert rec["content"] == "resp-1"

        monkeypatch.setenv("ITS_CACHE_MODE", "replay")
        rep = await lm.agenerate_single(_msgs())
        assert state["hits"] == 1  # replayed, not regenerated
        assert rep == rec
        await lm.close()
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_different_seeds_are_separate_cache_entries(monkeypatch, tmp_path):
    """A record made under seed=42 is a REPLAY MISS for seed=43 (distinct keys)."""
    import its_hub.core.lms.openai_lm as mod
    from its_hub.core.lms.openai_lm import CacheMissError

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(mod, "_cache", None)
    monkeypatch.setattr(mod, "_DEFAULT_CACHE_DIR", cache_dir)

    server, state = await _make_counting_server()
    try:
        # RECORD under seed=42.
        lm42 = OpenAICompatibleLanguageModel(
            endpoint=f"http://127.0.0.1:{server.port}/v1",
            api_key="k",
            model_name="m",
            temperature=0.7,
            seed=42,
            max_tries=1,
        )
        monkeypatch.setenv("ITS_CACHE_MODE", "record")
        await lm42.agenerate_single(_msgs())
        assert state["hits"] == 1
        await lm42.close()

        # REPLAY under seed=43 -> distinct key -> hard miss, API never called.
        monkeypatch.setattr(mod, "_cache", None)  # reset cursors
        lm43 = OpenAICompatibleLanguageModel(
            endpoint=f"http://127.0.0.1:{server.port}/v1",
            api_key="k",
            model_name="m",
            temperature=0.7,
            seed=43,
            max_tries=1,
        )
        monkeypatch.setenv("ITS_CACHE_MODE", "replay")
        with pytest.raises(CacheMissError):
            await lm43.agenerate_single(_msgs())
        assert state["hits"] == 1  # unchanged -> no regeneration
        await lm43.close()
    finally:
        await server.close()
