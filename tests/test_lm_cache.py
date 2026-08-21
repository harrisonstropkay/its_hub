"""Tests for the deterministic verification cache (H1).

Covers the cache key (stability + discrimination), the LMResponseCache
record/replay round-trip, CacheMissError on replay miss, and the OFF-mode
byte-identical / zero-overhead contract at the fetch_response seam.
"""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from its_hub.api.types import ChatMessage
from its_hub.core.lms.openai_lm import (
    CacheMissError,
    LMResponseCache,
    OpenAICompatibleLanguageModel,
    _compute_cache_key,
    _get_cache,
)


def _req(**overrides) -> dict:
    """A minimal prepared-request dict with sensible defaults."""
    base = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
    }
    base.update(overrides)
    return base


class TestCacheKey:
    def test_stability_across_calls(self):
        r1 = _req()
        r2 = _req()
        assert _compute_cache_key(r1, "m") == _compute_cache_key(r2, "m")
        # and repeated calls on the same object are stable too
        assert _compute_cache_key(r1, "m") == _compute_cache_key(r1, "m")

    def test_stability_independent_of_dict_order(self):
        r1 = {"messages": [{"role": "user", "content": "hi"}], "temperature": 0.7}
        r2 = {"temperature": 0.7, "messages": [{"role": "user", "content": "hi"}]}
        assert _compute_cache_key(r1, "m") == _compute_cache_key(r2, "m")

    def test_discriminates_temperature(self):
        assert _compute_cache_key(_req(temperature=0.0), "m") != _compute_cache_key(
            _req(temperature=1.0), "m"
        )

    def test_discriminates_logprobs_true_vs_false(self):
        assert _compute_cache_key(_req(logprobs=True), "m") != _compute_cache_key(
            _req(logprobs=False), "m"
        )

    def test_discriminates_logprobs_true_vs_none(self):
        assert _compute_cache_key(_req(logprobs=True), "m") != _compute_cache_key(
            _req(logprobs=None), "m"
        )

    def test_discriminates_tools_empty_vs_none(self):
        # tools=[] means "tool-calling on, none declared"; None means "off".
        assert _compute_cache_key(_req(tools=[]), "m") != _compute_cache_key(
            _req(tools=None), "m"
        )

    def test_discriminates_model_name(self):
        assert _compute_cache_key(_req(), "m1") != _compute_cache_key(_req(), "m2")

    def test_missing_field_equals_explicit_none(self):
        # A missing param and an explicit None collapse to the same key.
        assert _compute_cache_key(_req(), "m") == _compute_cache_key(
            _req(stop=None), "m"
        )

    def test_key_is_hex_sha256(self):
        key = _compute_cache_key(_req(), "m")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


class TestLMResponseCache:
    def test_put_writes_bucket_entry(self, tmp_path):
        cache = LMResponseCache(cache_dir=tmp_path, mode="record")
        key = "abc123ef" + "0" * 56
        cache.put(key, {"content": "hello", "_logprobs": {"content": []}})

        bucket = tmp_path / "abc123ef.jsonl"
        assert bucket.exists()
        lines = [ln for ln in bucket.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1

    def test_get_returns_cached_value_on_hit(self, tmp_path):
        cache = LMResponseCache(cache_dir=tmp_path, mode="record")
        key = "deadbeef" + "1" * 56
        payload = {"content": "42", "_logprobs": {"content": [{"logprob": -0.5}]}}
        cache.put(key, payload)

        # fresh instance -> must read from disk
        replay = LMResponseCache(cache_dir=tmp_path, mode="replay")
        assert replay.get(key) == payload

    def test_get_returns_none_on_miss(self, tmp_path):
        cache = LMResponseCache(cache_dir=tmp_path, mode="replay")
        assert cache.get("f" * 64) is None

    def test_bucket_sharded_by_prefix(self, tmp_path):
        cache = LMResponseCache(cache_dir=tmp_path, mode="record")
        k1 = "aaaaaaaa" + "0" * 56
        k2 = "bbbbbbbb" + "0" * 56
        cache.put(k1, {"content": "a"})
        cache.put(k2, {"content": "b"})
        assert (tmp_path / "aaaaaaaa.jsonl").exists()
        assert (tmp_path / "bbbbbbbb.jsonl").exists()

    def test_multiple_entries_same_bucket(self, tmp_path):
        # Distinct keys that collide on the 8-char prefix share a bucket file.
        cache = LMResponseCache(cache_dir=tmp_path, mode="record")
        k1 = "cafef00d" + "1" * 56
        k2 = "cafef00d" + "2" * 56
        cache.put(k1, {"content": "one"})
        cache.put(k2, {"content": "two"})
        assert cache.get(k1) == {"content": "one"}
        assert cache.get(k2) == {"content": "two"}

    def test_invalid_mode_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid ITS_CACHE_MODE"):
            LMResponseCache(cache_dir=tmp_path, mode="bogus")

    def test_off_mode_does_not_create_dir(self, tmp_path):
        cache_dir = tmp_path / "should_not_exist"
        LMResponseCache(cache_dir=cache_dir, mode="off")
        assert not cache_dir.exists()


class TestGetCacheSingleton:
    def test_rebuilds_on_mode_change(self, monkeypatch):
        import its_hub.core.lms.openai_lm as mod

        monkeypatch.setattr(mod, "_cache", None)
        monkeypatch.setenv("ITS_CACHE_MODE", "record")
        c_record = _get_cache()
        assert c_record.mode == "record"

        monkeypatch.setenv("ITS_CACHE_MODE", "replay")
        c_replay = _get_cache()
        assert c_replay.mode == "replay"
        assert c_replay is not c_record

    def test_defaults_to_off(self, monkeypatch):
        import its_hub.core.lms.openai_lm as mod

        monkeypatch.setattr(mod, "_cache", None)
        monkeypatch.delenv("ITS_CACHE_MODE", raising=False)
        assert _get_cache().mode == "off"


async def _make_counting_server():
    """A stand-in chat-completion server that counts hits and echoes a counter.

    Content changes per call so we can detect whether the API was actually hit
    (record miss) vs served from cache (replay/record hit).
    """
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
async def test_off_mode_is_byte_identical_and_touches_no_cache(monkeypatch, tmp_path):
    """OFF mode: no cache dir created, response unchanged from no-cache path."""
    import its_hub.core.lms.openai_lm as mod

    monkeypatch.setattr(mod, "_cache", None)
    monkeypatch.setattr(mod, "_DEFAULT_CACHE_DIR", tmp_path / "cache")
    monkeypatch.delenv("ITS_CACHE_MODE", raising=False)  # default off

    server, state = await _make_counting_server()
    try:
        lm = OpenAICompatibleLanguageModel(
            endpoint=f"http://127.0.0.1:{server.port}/v1",
            api_key="k",
            model_name="m",
            max_tries=1,
        )
        r1 = await lm.agenerate_single([ChatMessage(role="user", content="hi")])
        r2 = await lm.agenerate_single([ChatMessage(role="user", content="hi")])
        await lm.close()
    finally:
        await server.close()

    # Each call hit the API (no caching) and content advanced.
    assert state["hits"] == 2
    assert r1["content"] == "resp-1"
    assert r2["content"] == "resp-2"
    # No cache directory was created.
    assert not (tmp_path / "cache").exists()


@pytest.mark.asyncio
async def test_record_then_replay_roundtrip(monkeypatch, tmp_path):
    """record writes; replay serves the recorded draw without hitting the API."""
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
            max_tries=1,
        )

        # RECORD: one real API call, written to cache.
        monkeypatch.setenv("ITS_CACHE_MODE", "record")
        rec = await lm.agenerate_single(
            [ChatMessage(role="user", content="what is 2+2?")]
        )
        assert state["hits"] == 1
        assert rec["content"] == "resp-1"
        assert cache_dir.exists()

        # REPLAY: same request -> served from cache, API NOT hit again.
        monkeypatch.setenv("ITS_CACHE_MODE", "replay")
        rep = await lm.agenerate_single(
            [ChatMessage(role="user", content="what is 2+2?")]
        )
        assert state["hits"] == 1  # unchanged -> no regeneration
        assert rep == rec  # byte-identical incl. _logprobs

        await lm.close()
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_replay_raises_on_miss(monkeypatch, tmp_path):
    """replay mode never calls the API; a miss is a hard CacheMissError."""
    import its_hub.core.lms.openai_lm as mod

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(mod, "_cache", None)
    monkeypatch.setattr(mod, "_DEFAULT_CACHE_DIR", cache_dir)
    monkeypatch.setenv("ITS_CACHE_MODE", "replay")

    server, state = await _make_counting_server()
    try:
        lm = OpenAICompatibleLanguageModel(
            endpoint=f"http://127.0.0.1:{server.port}/v1",
            api_key="k",
            model_name="m",
            max_tries=1,
        )
        with pytest.raises(CacheMissError, match="Cache miss in replay mode"):
            await lm.agenerate_single(
                [ChatMessage(role="user", content="never recorded")]
            )
        # crucially, the API was never contacted
        assert state["hits"] == 0
        await lm.close()
    finally:
        await server.close()


async def _make_pool_server(pool):
    """Stand-in server returning ``pool[hit-1]`` per call (with logprobs)."""
    state = {"hits": 0}

    async def handler(request):
        content = pool[state["hits"] % len(pool)]
        state["hits"] += 1
        return web.json_response(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": content},
                        "logprobs": {
                            "content": [{"token": "t", "logprob": -0.1}]
                        },
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
async def test_self_consistency_record_replay_and_modify(monkeypatch, tmp_path):
    """Decisive property (CI stand-in): record a SelfConsistency run, replay it
    -> identical the_one + vote counts with NO regeneration; then change the
    aggregation under replay -> the result changes, proving replay feeds the
    cached samples into live aggregation rather than secretly regenerating.
    """
    import its_hub.core.lms.openai_lm as mod
    from its_hub.core.algorithms.self_consistency import SelfConsistency

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(mod, "_cache", None)
    monkeypatch.setattr(mod, "_DEFAULT_CACHE_DIR", cache_dir)

    budget = 5
    pool = ["ans-X", "ans-X", "ans-X", "ans-Y", "ans-Y"]  # majority ans-X
    server, state = await _make_pool_server(pool)
    try:
        lm = OpenAICompatibleLanguageModel(
            endpoint=f"http://127.0.0.1:{server.port}/v1",
            api_key="k",
            model_name="m",
            temperature=0.8,
            max_tries=1,
        )

        # RECORD
        monkeypatch.setenv("ITS_CACHE_MODE", "record")
        sc = SelfConsistency()
        rec = await sc.ainfer(lm, "q", budget=budget, return_response_only=False)
        assert state["hits"] == budget  # N distinct fresh draws captured
        assert rec.the_one["content"] == "ans-X"
        assert dict(rec.response_counts) == {"ans-X": 3, "ans-Y": 2}

        # REPLAY (same aggregation) -> byte-identical winner + counts, no new
        # API hits. NOTE: selected_index is NOT asserted: SelfConsistency breaks
        # the intra-winner tie randomly, so the integer index of the chosen
        # winner varies run-to-run even though the winning ANSWER (the_one) and
        # the vote counts are exactly reproduced from the cached samples.
        monkeypatch.setenv("ITS_CACHE_MODE", "replay")
        sc2 = SelfConsistency()
        rep = await sc2.ainfer(lm, "q", budget=budget, return_response_only=False)
        assert state["hits"] == budget  # unchanged -> no regeneration
        assert rep.the_one["content"] == rec.the_one["content"]
        assert dict(rep.response_counts) == dict(rec.response_counts)

        # MODIFY aggregation under replay: collapse every sample to one bucket.
        monkeypatch.setattr(mod, "_cache", None)  # reset replay cursors
        # Same cached samples, different voting -> different counts/result.
        monkeypatch.setenv("ITS_CACHE_MODE", "replay")
        sc3 = SelfConsistency(consistency_space_projection_func=lambda _c: "ALL")
        mod_res = await sc3.ainfer(lm, "q", budget=budget, return_response_only=False)
        assert state["hits"] == budget  # still no regeneration
        assert dict(mod_res.response_counts) == {"ALL": budget}
        assert dict(mod_res.response_counts) != dict(rep.response_counts)

        await lm.close()
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_record_then_replay_via_agenerate_batch(monkeypatch, tmp_path):
    """The twin seam in _agenerate() (deprecated batch path) also caches."""
    import warnings

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
            temperature=0.5,
            max_tries=1,
        )
        msgs = [ChatMessage(role="user", content="batch q")]

        monkeypatch.setenv("ITS_CACHE_MODE", "record")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            rec = await lm.agenerate(msgs)
        assert state["hits"] == 1

        monkeypatch.setenv("ITS_CACHE_MODE", "replay")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            rep = await lm.agenerate(msgs)
        assert state["hits"] == 1  # replayed, not regenerated
        assert rep == rec
        await lm.close()
    finally:
        await server.close()
