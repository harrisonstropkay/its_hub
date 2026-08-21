import asyncio
import hashlib
import json
import logging
import os
import ssl
import threading
import warnings
import weakref
from pathlib import Path

import aiohttp
import backoff
import certifi

from its_hub.api import (
    RETRYABLE_ERRORS,
    AbstractLanguageModel,
    APIError,
    ChatMessage,
    GenerationUsage,
    enhanced_on_backoff,
    format_non_retryable_error,
    parse_api_error,
    should_retry,
)
from its_hub.core.utils import resolve_max_completion_tokens

# ---------------------------------------------------------------------------
# Deterministic verification cache (H1)
#
# Records the ONE stochastic event in the ITS pipeline -- the LM HTTP response
# -- at the client seam so that aggregation-only experiments (voting,
# confidence weighting, medoid selection, ...) can be replayed off-GPU,
# byte-identical, without regenerating samples.
#
# Mode is selected via the ITS_CACHE_MODE environment variable:
#   off    (default) -- no caching; behaviour is byte-identical to no-cache and
#                       the cache key is never computed on the hot path.
#   record           -- on hit, return the cached response; on miss, call the
#                       API as normal and write (key -> response) afterwards.
#   replay           -- on hit, return the cached response; on miss, raise
#                       CacheMissError (NEVER call the API, NEVER regenerate).
# ---------------------------------------------------------------------------

# Environment variable that selects the cache mode.
ITS_CACHE_MODE_ENV = "ITS_CACHE_MODE"
_VALID_CACHE_MODES = frozenset({"off", "record", "replay"})
_DEFAULT_CACHE_DIR = Path(".factory/cache/lm_responses")

# Parameters that affect generation and therefore participate in the cache key.
# Order is irrelevant (JSON is emitted with sorted keys) but the set is fixed.
_CACHE_KEY_PARAMS = (
    "temperature",
    "stop",
    "max_completion_tokens",
    "tools",
    "tool_choice",
    "response_format",
    "logprobs",
    "top_logprobs",
)


class CacheMissError(RuntimeError):
    """Raised when replay mode encounters a cache miss.

    In replay mode the API is never called, so a missing entry is a hard
    failure -- silently regenerating would break the reproducibility contract
    (replay == the recorded draw, byte-identical).
    """


def _compute_cache_key(request_data: dict, model_name: str) -> str:
    """Content-addressed SHA256 key for an LM request.

    The key covers every parameter that affects generation. It is built with a
    ``dict.get`` per field so that:

    * a missing field and an explicit ``None`` collapse to the same ``null``
      (they mean the same thing to the API), while
    * ``tools=[]`` and ``tools=None`` stay distinct ([] is serialized, None is
      the JSON null), and
    * ``logprobs=True`` and ``logprobs=False``/``None`` are distinct entries
      (the API may return different metadata; exact replay demands separation).

    Returns a 64-char hex digest.
    """
    key_obj = {
        "model": model_name,
        # messages are always present in a prepared request; keep them explicit
        # so a malformed request fails loudly rather than hashing to a shared key
        "messages": request_data["messages"],
    }
    for param in _CACHE_KEY_PARAMS:
        key_obj[param] = request_data.get(param)

    # Seed (H2) participates in the key ONLY when set. Adding a "seed": null
    # entry for the unseeded case would perturb the canonical JSON and thus the
    # digest, invalidating every cache recorded before H2. By omitting the field
    # entirely when seed is None, an unseeded request hashes byte-identically to
    # the pre-H2 key (backward-compatible), while a set seed yields a distinct
    # entry -- seed=42, seed=43, and unseeded are three different cache lines.
    seed = request_data.get("seed")
    if seed is not None:
        key_obj["seed"] = seed

    canonical_json = json.dumps(key_obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class LMResponseCache:
    """JSONL-backed store of recorded LM responses.

    Entries are sharded into buckets named by the first 8 hex chars of the
    cache key (256^4 buckets keeps any single file small). Each line is a
    self-contained JSON object holding the full response payload -- including
    ``_logprobs`` -- so replay is byte-preservable through a JSON round-trip.

    **Multiset / sequential replay.** A single self-consistency (or best-of-N)
    generation fires N requests with an *identical* key (same prompt, params,
    model) and, at temperature>0, draws N *distinct* samples. The cache must
    therefore behave as an ordered log per key, not a single value:

    * ``put`` appends -- multiple draws under one key are all retained.
    * ``get`` serves the recorded draws in file order, one per call, advancing a
      per-key cursor. The first ``get`` for a key snapshots the on-disk entries
      into memory; later ``put`` calls in the same run do NOT feed back into that
      snapshot. This makes ``record`` capture N distinct fresh draws regardless
      of concurrency (every request in a fresh run misses the frozen snapshot),
      while still returning prior-run entries on a re-record. When the recorded
      draws for a key are exhausted, ``get`` returns ``None`` (a replay miss).
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        mode: str | None = None,
    ):
        # Resolve the dir at call time (not def time) so the module-level
        # default remains overridable, e.g. in tests.
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
        self.mode = mode if mode is not None else _resolve_cache_mode()
        if self.mode not in _VALID_CACHE_MODES:
            raise ValueError(
                f"Invalid {ITS_CACHE_MODE_ENV}={self.mode!r}; "
                f"expected one of {sorted(_VALID_CACHE_MODES)}"
            )
        # Per-key snapshot of recorded responses (in file order) and a consume
        # cursor, both established lazily on first access to a key.
        self._entries: dict[str, list[dict]] = {}
        self._cursors: dict[str, int] = {}

    def _bucket_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key[:8]}.jsonl"

    def _load_entries(self, key: str) -> list[dict]:
        """Read all recorded responses for ``key`` from disk, in file order."""
        bucket = self._bucket_path(key)
        if not bucket.exists():
            return []
        entries: list[dict] = []
        with bucket.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("cache_key") == key:
                    entries.append(entry["response"])
        return entries

    def get(self, key: str) -> dict | None:
        """Return the next recorded response for ``key`` or ``None`` if exhausted.

        Snapshots on-disk entries on first access to the key, then serves them
        sequentially so that N identical-key requests replay N distinct draws.
        """
        if key not in self._entries:
            self._entries[key] = self._load_entries(key)
        entries = self._entries[key]
        idx = self._cursors.get(key, 0)
        if idx >= len(entries):
            return None
        self._cursors[key] = idx + 1
        return entries[idx]

    def put(self, key: str, response_dict: dict) -> None:
        """Append ``key -> response_dict`` to the appropriate bucket file."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        bucket = self._bucket_path(key)
        entry = {"cache_key": key, "response": response_dict}
        with bucket.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, separators=(",", ":"), sort_keys=True) + "\n")


def _resolve_cache_mode() -> str:
    """Read the current cache mode from the environment (default 'off')."""
    return os.getenv(ITS_CACHE_MODE_ENV, "off")


# Lazy module-level singleton, rebuilt when the env-selected mode changes so a
# process can switch record -> replay within a single run (e.g. tests, the A/B
# record-then-replay workflow).
_cache: LMResponseCache | None = None


def _get_cache() -> LMResponseCache:
    """Return the process-wide cache, honoring the current ITS_CACHE_MODE."""
    global _cache
    mode = _resolve_cache_mode()
    if _cache is None or _cache.mode != mode:
        _cache = LMResponseCache(mode=mode)
    return _cache


class OpenAICompatibleLanguageModel(AbstractLanguageModel):
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model_name: str,
        system_prompt: str | None = None,
        is_async: bool = False,  # Deprecated: parameter is ignored (always async internally)
        # default runtime parameters
        stop: str | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        max_tries: int = 3,
        max_concurrency: int = -1,
        replace_error_with_message: str | None = None,
        # SSL configuration
        verify_ssl: bool = True,
        ssl_context: ssl.SSLContext | None = None,
        # Raw response preservation
        include_raw_choices: bool = False,
        # Reproducible sampling (H2)
        seed: int | None = None,
    ):
        """OpenAI-compatible chat-completion client.

        Args:
            seed: Optional sampling seed forwarded to the server on every
                request. On a vLLM backend this maps to
                ``SamplingParams(seed=...)``. When ``None`` (the default) no
                ``seed`` key is added to the request payload -- behaviour is
                byte-identical to a client with no seed support, and the H1
                verification cache key is unchanged, so caches recorded before
                seed support remain valid. A set seed also produces a distinct
                cache entry from unseeded / other-seed draws.

                For seed-level reproducibility of the *generated tokens* (not
                just record/replay of a captured draw) set the environment
                variable ``VLLM_BATCH_INVARIANT=1`` on the vLLM server so its
                kernels are batch-invariant. Even then, seeded reproducibility
                is best-effort: at ``temperature > 0`` outputs remain sensitive
                to hardware and library versions. Pair the seed with the H1
                cache (``ITS_CACHE_MODE``) when byte-exact replay is required.
        """
        assert max_concurrency == -1 or max_concurrency > 0, (
            "max_concurrency must be -1 (unlimited concurrency) or a positive integer"
        )

        # Warn about deprecated is_async parameter
        if is_async is not False:
            warnings.warn(
                "The 'is_async' parameter is deprecated and will be removed in a future version. "
                "The implementation now always uses async internally. "
                "Sync methods (generate, evaluate) automatically wrap async calls with asyncio.run().",
                DeprecationWarning,
                stacklevel=2,
            )

        self.endpoint = endpoint
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        # Keep is_async for backward compatibility but it's no longer used
        self.is_async = is_async
        self.max_tries = max_tries
        self.max_concurrency = max_concurrency
        self.replace_error_with_message = replace_error_with_message

        # runtime parameters
        self.stop = stop
        self.max_completion_tokens = resolve_max_completion_tokens(
            max_completion_tokens, max_tokens
        )
        self.temperature = temperature
        self.seed = seed

        # SSL configuration
        self.verify_ssl = verify_ssl
        if ssl_context is not None:
            self.ssl_context = ssl_context
        elif not verify_ssl:
            # Create an SSL context that doesn't verify certificates
            self.ssl_context = ssl.create_default_context()
            self.ssl_context.check_hostname = False
            self.ssl_context.verify_mode = ssl.CERT_NONE
        else:
            # For async requests, create SSL context using the same CA bundle as requests
            # This ensures aiohttp uses the same certificates as requests library
            self.ssl_context = ssl.create_default_context(cafile=certifi.where())

        # set up headers for API requests
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # raw response preservation
        self.include_raw_choices = include_raw_choices

        # endpoint type
        self.endpoint_type = "openai" if "openai" in self.endpoint else "vllm"

        # Session cache: one session per event loop, entry is auto-cleaned via weak references.
        # Session(s) need to be closed via close or close_session function calls.
        self._sessions: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, aiohttp.ClientSession
        ] = weakref.WeakKeyDictionary()
        self._session_lock = threading.Lock()

    @property
    def _chat_completion_endpoint(self) -> str:
        return self.endpoint.rstrip("/") + "/chat/completions"

    def _get_session(self, loop: asyncio.AbstractEventLoop) -> aiohttp.ClientSession:
        """Get or create an HTTP session for the given event loop.

        Sessions are cached per event loop and automatically cleaned up
        when the loop is garbage collected.
        """
        with self._session_lock:
            session = self._sessions.get(loop)
            if session is not None and not session.closed:
                return session

            # Create new session for the event loop.
            # trust_env=True so aiohttp honors HTTP(S)_PROXY/NO_PROXY/.netrc.
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            session = aiohttp.ClientSession(connector=connector, trust_env=True)
            self._sessions[loop] = session

            return session

    async def close(self) -> None:
        """Close all cached HTTP sessions."""
        with self._session_lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        for session in sessions:
            if not session.closed:
                await session.close()

    async def close_session(
        self, loop: asyncio.AbstractEventLoop | None = None
    ) -> None:
        """Close the cached session for the given event loop.

        Args:
            loop: Event loop whose session to close. Defaults to the current running loop.
        """
        if loop is None:
            loop = asyncio.get_running_loop()
        with self._session_lock:
            session = self._sessions.get(loop)
        if session and not session.closed:
            await session.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - ensures sessions are closed."""
        await self.close()
        return False

    def _prepare_request_data(
        self,
        messages: list[ChatMessage],
        stop: str | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        include_stop_str_in_output: bool | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> dict:
        # helper method to prepare request data for both sync and async methods
        # Convert dict messages to Message objects if needed
        messages = [
            msg if isinstance(msg, ChatMessage) else ChatMessage(**msg)
            for msg in messages
        ]

        if self.system_prompt:
            messages = [
                ChatMessage(role="system", content=self.system_prompt),
                *messages,
            ]

        request_data = {
            "model": self.model_name,
            "messages": [msg.to_dict() for msg in messages],
        }

        if self.endpoint_type == "vllm":
            request_data["extra_body"] = {}
            if messages[-1].role == "assistant":
                request_data["extra_body"]["add_generation_prompt"] = False
                request_data["extra_body"]["continue_final_message"] = True
                request_data["add_generation_prompt"] = False
                request_data["continue_final_message"] = True
            if include_stop_str_in_output is not None:
                request_data["extra_body"]["include_stop_str_in_output"] = (
                    include_stop_str_in_output
                )
                request_data["include_stop_str_in_output"] = include_stop_str_in_output
        else:
            logging.info(
                "openai endpoint does not support add_generation_prompt, continue_final_message, or include_stop_str_in_output"
            )
            if include_stop_str_in_output is not None:
                logging.warning(
                    "include_stop_str_in_output parameter is not supported with OpenAI endpoints and will be ignored"
                )

        # set default runtime parameters
        if self.stop is not None:
            request_data["stop"] = self.stop
        if self.max_completion_tokens is not None:
            request_data["max_completion_tokens"] = self.max_completion_tokens
        if self.temperature is not None:
            request_data["temperature"] = self.temperature

        # override runtime parameters
        if stop is not None:
            request_data["stop"] = stop
        if max_completion_tokens is not None:
            request_data["max_completion_tokens"] = max_completion_tokens
        if temperature is not None:
            request_data["temperature"] = temperature

        # seed for reproducible sampling (H2). Only emitted when set, so an
        # unseeded client sends a byte-identical payload to the pre-seed
        # behaviour. On vLLM this maps to SamplingParams(seed=...).
        if self.seed is not None:
            request_data["seed"] = self.seed

        # add tools and tool_choice if provided
        if tools is not None:
            request_data["tools"] = tools
        if tool_choice is not None:
            request_data["tool_choice"] = tool_choice

        # add response_format for structured outputs
        if response_format is not None:
            request_data["response_format"] = response_format

        # add logprobs for token-level log probability data
        if logprobs is not None:
            request_data["logprobs"] = logprobs
        if top_logprobs is not None:
            request_data["top_logprobs"] = top_logprobs

        return request_data

    async def _agenerate(
        self,
        messages_lst: list[list[ChatMessage]],
        stop: str | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        include_stop_str_in_output: bool | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict | None = None,
        usage_accumulator: GenerationUsage | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> list[dict]:
        # limit concurrency to max_concurrency using a semaphore
        semaphore = asyncio.Semaphore(
            len(messages_lst) if self.max_concurrency == -1 else self.max_concurrency
        )

        # create a single session for all requests in this call
        # Use the same SSL behavior as requests library
        connector = aiohttp.TCPConnector(ssl=self.ssl_context)
        async with aiohttp.ClientSession(
            connector=connector, trust_env=True
        ) as session:

            @backoff.on_exception(
                backoff.expo,
                RETRYABLE_ERRORS,
                max_tries=self.max_tries,
                on_backoff=enhanced_on_backoff,
                giveup=lambda e: not should_retry(e),
            )
            async def fetch_response(
                messages: list[ChatMessage], _temperature: float | None
            ) -> dict:
                async with semaphore:
                    request_data = self._prepare_request_data(
                        messages,
                        stop,
                        max_completion_tokens,
                        _temperature,
                        include_stop_str_in_output,
                        tools,
                        tool_choice,
                        response_format,
                        logprobs,
                        top_logprobs,
                    )

                    # --- verification cache (H1) ---
                    # OFF is the default: the key is never computed and no cache
                    # dir is touched, so behaviour is byte-identical to no-cache.
                    cache_mode = _resolve_cache_mode()
                    if cache_mode != "off":
                        cache = _get_cache()
                        cache_key = _compute_cache_key(request_data, self.model_name)
                        cached = cache.get(cache_key)
                        if cached is not None:
                            return cached
                        if cache_mode == "replay":
                            raise CacheMissError(
                                f"Cache miss in replay mode for key {cache_key[:16]}... "
                                f"(model={self.model_name}, temperature={_temperature})"
                            )

                    async with session.post(
                        self._chat_completion_endpoint,
                        headers=self.headers,
                        json=request_data,
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            api_error = parse_api_error(response.status, error_text)
                            if not should_retry(api_error):
                                logging.error(format_non_retryable_error(api_error))
                            raise api_error
                        response_json = await response.json()
                        choice = response_json["choices"][0]
                        message = dict(choice["message"])
                        if self.include_raw_choices:
                            message["_raw_choice"] = {
                                **choice,
                                "message": dict(choice["message"]),
                            }
                        if (lp := choice.get("logprobs")) is not None:
                            message["_logprobs"] = lp
                        if usage_accumulator is not None:
                            api_usage = response_json.get("usage", {})
                            usage_accumulator.add(
                                api_usage.get("prompt_tokens", 0),
                                api_usage.get("completion_tokens", 0),
                            )
                        # record the freshly-drawn sample after a successful call
                        if cache_mode == "record":
                            cache.put(cache_key, message)
                        return message

            async def safe_fetch_response(
                messages: list[ChatMessage], _temperature: float | None
            ) -> dict:
                if self.replace_error_with_message is not None:
                    try:
                        return await fetch_response(messages, _temperature)
                    except (aiohttp.ClientError, TimeoutError) as e:
                        logging.error(f"Network error during async generation: {e}")
                        return {
                            "role": "assistant",
                            "content": self.replace_error_with_message,
                        }
                    except APIError as e:
                        logging.error(f"API error during async generation: {e}")
                        return {
                            "role": "assistant",
                            "content": self.replace_error_with_message,
                        }
                else:
                    return await fetch_response(messages, _temperature)

            # gather all responses asynchronously, with concurrency limited to max_concurrency
            temperature_lst = (
                temperature
                if isinstance(temperature, list)
                else [temperature] * len(messages_lst)
            )
            return await asyncio.gather(
                *(
                    safe_fetch_response(messages, _temperature)
                    for messages, _temperature in zip(messages_lst, temperature_lst)
                )
            )

    async def agenerate(
        self,
        messages_or_messages_lst: list[ChatMessage] | list[list[ChatMessage]],
        stop: str | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | list[float] | None = None,
        include_stop_str_in_output: bool | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict | None = None,
        usage_accumulator: GenerationUsage | None = None,
    ) -> dict | list[dict]:
        """
        generate response(s) asynchronously

        FIXME: Batch processing has been moved to orchestrator. This function will be fully
        replaced by agenerate_single once all algorithms have been moved to using orchestrator.
        """

        warnings.warn(
            "agenerate() is deprecated and will be removed in a future version. "
            "Use agenerate_single() with the orchestrator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        max_completion_tokens = resolve_max_completion_tokens(
            max_completion_tokens, max_tokens
        )

        is_single = not isinstance(messages_or_messages_lst[0], list)
        messages_lst = (
            [messages_or_messages_lst] if is_single else messages_or_messages_lst
        )
        response_or_responses = await self._agenerate(
            messages_lst,
            stop,
            max_completion_tokens,
            temperature,
            include_stop_str_in_output,
            tools,
            tool_choice,
            response_format,
            usage_accumulator,
        )
        return response_or_responses[0] if is_single else response_or_responses

    async def agenerate_single(
        self,
        messages: list[ChatMessage],
        stop: str | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        include_stop_str_in_output: bool | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        usage_accumulator: GenerationUsage | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> dict:
        max_completion_tokens = resolve_max_completion_tokens(
            max_completion_tokens, max_tokens
        )

        # Fallback to the current event loop
        if loop is None:
            loop = asyncio.get_running_loop()
        # Get or create session for the event loop
        session = self._get_session(loop)

        @backoff.on_exception(
            backoff.expo,
            RETRYABLE_ERRORS,
            max_tries=self.max_tries,
            on_backoff=enhanced_on_backoff,
            giveup=lambda e: not should_retry(e),
        )
        async def fetch_response(
            messages: list[ChatMessage], _temperature: float | None
        ) -> dict:
            request_data = self._prepare_request_data(
                messages,
                stop,
                max_completion_tokens,
                _temperature,
                include_stop_str_in_output,
                tools,
                tool_choice,
                response_format,
                logprobs,
                top_logprobs,
            )

            # --- verification cache (H1) ---
            # OFF is the default: the key is never computed and no cache dir is
            # touched, so behaviour is byte-identical to no-cache.
            cache_mode = _resolve_cache_mode()
            if cache_mode != "off":
                cache = _get_cache()
                cache_key = _compute_cache_key(request_data, self.model_name)
                cached = cache.get(cache_key)
                if cached is not None:
                    return cached
                if cache_mode == "replay":
                    raise CacheMissError(
                        f"Cache miss in replay mode for key {cache_key[:16]}... "
                        f"(model={self.model_name}, temperature={_temperature})"
                    )

            async with session.post(
                self._chat_completion_endpoint,
                headers=self.headers,
                json=request_data,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    api_error = parse_api_error(response.status, error_text)
                    if not should_retry(api_error):
                        logging.error(format_non_retryable_error(api_error))
                    raise api_error
                response_json = await response.json()
                choice = response_json["choices"][0]
                message = dict(choice["message"])
                if self.include_raw_choices:
                    message["_raw_choice"] = {
                        **choice,
                        "message": dict(choice["message"]),
                    }
                if (lp := choice.get("logprobs")) is not None:
                    message["_logprobs"] = lp
                if usage_accumulator is not None:
                    api_usage = response_json.get("usage", {})
                    usage_accumulator.add(
                        api_usage.get("prompt_tokens", 0),
                        api_usage.get("completion_tokens", 0),
                    )
                # record the freshly-drawn sample after a successful call
                if cache_mode == "record":
                    cache.put(cache_key, message)
                return message

        async def safe_fetch_response(
            messages: list[ChatMessage], _temperature: float | None
        ) -> dict:
            if self.replace_error_with_message is not None:
                try:
                    return await fetch_response(messages, _temperature)
                except (aiohttp.ClientError, TimeoutError) as e:
                    logging.error(f"Network error during async generation: {e}")
                    return {
                        "role": "assistant",
                        "content": self.replace_error_with_message,
                    }
                except APIError as e:
                    logging.error(f"API error during async generation: {e}")
                    return {
                        "role": "assistant",
                        "content": self.replace_error_with_message,
                    }
            else:
                return await fetch_response(messages, _temperature)

        return await safe_fetch_response(messages, temperature)
