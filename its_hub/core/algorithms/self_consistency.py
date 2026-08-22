import json
import logging
import math
import os
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from its_hub.api import (
    AbstractLanguageModel,
    AbstractOrchestrator,
    AbstractScalingAlgorithm,
    AbstractScalingResult,
    ChatMessage,
    ChatMessages,
    GenerationUsage,
)
from its_hub.core.algorithms._sc_voting import (
    _default_projection_func,
    _resolve_vote_mode,
    _select_hierarchical_most_common_or_random,
    _select_most_common_or_random,
    _tiebreak_by_confidence,
)
from its_hub.core.orchestrator import LMOrchestrator
from its_hub.core.utils import (
    CONTINUATION_PRIMER,
    CONTINUATION_PRIMER_MCQ,
    _canonicalize_vote_key,
    extract_content_from_lm_response,
)

# Re-exported for backward compatibility: these voting/selection helpers now
# live in ``_sc_voting`` but are still imported from this module by
# ``weighted_self_consistency`` and the test suite.
__all__ = [
    "SelfConsistency",
    "SelfConsistencyResult",
    "_default_projection_func",
    "_select_hierarchical_most_common_or_random",
    "_select_most_common_or_random",
    "_tiebreak_by_confidence",
    "create_regex_projection_function",
    "validate_regex_patterns",
]

# Default tool-call voting strategy for the self-consistency family. Chosen so
# that responses containing tool calls vote sensibly out of the box (the primary
# use case for the IaaS gateway). Pass tool_vote=None to force content-only
# voting, which raises if every response is a tool call.
DEFAULT_TOOL_VOTE = "tool_hierarchical"

# --- Answer/budget forcing (H1, exp-15) -------------------------------------
# Default-OFF decode mode gated on ``ITS_SC_ANSWER_FORCE``. When ON it caps the
# main generation below the context window and issues ONE short forced
# continuation for any sample that never emitted a terminal ``\boxed{}`` answer,
# attacking the dominant CONTEXT_TRUNCATION failure. Resolved at the ``ainfer``
# call site (mirroring ``ITS_SC_VOTE``) so a single env var flips the branch with
# NO code edits between A/B runs. With the flag unset the generation call and ALL
# downstream processing are byte-identical to the baseline path.
_ANSWER_FORCE_ENV_VAR = "ITS_SC_ANSWER_FORCE"

# Main-generation cap when forcing is ON: reserves headroom below the ~3.9k
# completion window (4096 context - prompt) so the forced continuation fits
# before the hard cap (s1, arXiv:2501.19393).
_ANSWER_FORCE_MAIN_MAX_TOKENS = 3600
# Short continuation budgets. Numeric (MATH/AIME) answers need room for a short
# expression; MCQ/GPQA answers are a single letter, so a tighter cap physically
# prevents verbose completions. Ambiguous item shapes default to the numeric
# (wider) budget so a numeric answer is never under-budgeted.
_ANSWER_FORCE_CONT_MAX_TOKENS_NUMERIC = 48
_ANSWER_FORCE_CONT_MAX_TOKENS_MCQ = 16
# Stop string terminating the forced ``\boxed{...}``. Relies on vLLM's default
# ``include_stop_str_in_output=False`` so the closing brace is not echoed back
# (we re-append it when reconstructing the completed text).
_ANSWER_FORCE_STOP = "}"

# A completed ``\boxed{...}`` answer: opening brace, a non-empty body, and a
# closing brace. Presence means the sample carries a real terminal answer;
# absence (a truncated ``\boxed{`` with no close, or no box at all) triggers
# forcing. This is the same ``\boxed`` notion the projection relies on for
# vote-key extraction.
_BOXED_RE = re.compile(r"\\boxed\{.+?\}", re.DOTALL)
# Decorated option-letter markers (A-D) as they appear in MCQ/GPQA prompts:
# ``A)`` ``(A)`` ``A.`` ``[A]``. Used to infer item shape from the prompt only,
# without reading any dataset or benchmark code.
_MCQ_OPTION_RE = re.compile(r"(?:^|\n|\s)[(\[]?\s*([A-D])\s*[).\]]", re.MULTILINE)
# Minimum distinct decorated A-D letters required to treat an item as MCQ.
_MCQ_MIN_DISTINCT_OPTIONS = 3


def _resolve_answer_force() -> bool:
    """Whether answer/budget forcing is enabled via ``ITS_SC_ANSWER_FORCE``.

    Default OFF: only an explicit truthy value (``1``/``true``/``yes``/``on``,
    case-insensitive) enables forcing. Unset, ``0``, empty, or any other value
    returns ``False`` so the baseline decode path is byte-identical.
    """
    val = os.environ.get(_ANSWER_FORCE_ENV_VAR, "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _has_boxed_answer(content: str | None) -> bool:
    """Whether text carries a completed ``\\boxed{...}`` answer."""
    return bool(content) and _BOXED_RE.search(content) is not None


def _looks_like_mcq(prompt_text: str | None) -> bool:
    """Infer MCQ/GPQA item shape from the prompt's option-letter markers.

    True when at least ``_MCQ_MIN_DISTINCT_OPTIONS`` distinct decorated letters
    among A-D appear (e.g. ``A) ...`` ``(B) ...`` ``C. ...``). Deliberately
    conservative: when the signal is ambiguous this returns ``False`` so the
    caller uses the wider numeric continuation budget.
    """
    letters = {m.group(1).upper() for m in _MCQ_OPTION_RE.finditer(prompt_text or "")}
    return len(letters) >= _MCQ_MIN_DISTINCT_OPTIONS


@dataclass
class SelfConsistencyResult(AbstractScalingResult):
    responses: list[dict]  # Keep original message format with tool calls
    response_counts: Counter[str] | Counter[tuple] | Counter
    selected_index: int
    usage: GenerationUsage | None = None

    @property
    def the_one(self) -> dict:
        return self.responses[self.selected_index]


class SelfConsistency(AbstractScalingAlgorithm):
    def __init__(
        self,
        consistency_space_projection_func: Callable | None = None,
        tool_vote: str | None = DEFAULT_TOOL_VOTE,
        exclude_args: list[str] | None = None,
        orchestrator: AbstractOrchestrator | None = None,
    ):
        """Initialize SelfConsistency algorithm with optional tool-vote capability.

        Args:
            consistency_space_projection_func: Function that maps response content (str)
                to a comparable value for voting. Used when tool_vote is None or when
                responses don't contain tool calls. Can return str, tuple, or any hashable type.

            tool_vote: Tool voting strategy when responses contain tool calls. Options:
                - "tool_hierarchical" (default): Vote on tool name first, then arguments
                - None: Vote on message content using consistency_space_projection_func;
                  used when no tool calls are present, or to force content-only voting.
                  Raises at inference time if every response is a tool call.
                - "tool_name": Vote on tool function names only
                - "tool_args": Vote on tool function arguments only (as dicts)
                - "tool_flat_all": Vote on ALL tool calls combined as a flat sorted signature.
                  Unlike other modes which only consider the first tool call, this creates a
                  single signature from all tool calls in a response. Two responses calling
                  the same tools with the same args (in any order) produce the same signature.
                When tool calls exist and tool_vote is set, this takes priority over content voting.

            exclude_args: List of argument names to exclude from tool voting when
                tool_vote is "tool_args" or "tool_hierarchical". Useful for filtering out
                non-semantic arguments like timestamps, request IDs, etc.

            orchestrator: Orchestrator that manages parallel calls to LM

        Raises:
            ValueError: If tool_vote is not one of the supported options.
        """
        # Validate tool_vote parameter - only validation needed since typing handles the rest
        valid_tool_vote_options = {
            None,
            "tool_name",
            "tool_args",
            "tool_hierarchical",
            "tool_flat_all",
        }
        if tool_vote not in valid_tool_vote_options:
            raise ValueError(
                f"tool_vote must be one of {valid_tool_vote_options}, got: {tool_vote}"
            )
        # Set default projection function if provided None
        self.consistency_space_projection_func = (
            consistency_space_projection_func or _default_projection_func
        )
        self.tool_vote = tool_vote
        self.exclude_args = exclude_args or []

        if orchestrator is None:
            # Fallback to default implementation
            orchestrator = LMOrchestrator()
        self.orchestrator = orchestrator

    async def ainfer(
        self,
        lm: AbstractLanguageModel,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        budget: int,
        return_response_only: bool = True,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict | SelfConsistencyResult:
        """run inference asynchronously with self-consistency"""
        chat_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)

        usage = GenerationUsage()

        # generate responses. logprobs=True is requested so the confidence
        # tie-break (see _process_responses) has per-token log probabilities to
        # break ties among equally-voted answer groups. Requesting logprobs is a
        # returned-metadata flag and does not alter sampling, so the vote outcome
        # for a clear majority is unaffected.
        #
        # Answer/budget forcing (H1) is gated on ITS_SC_ANSWER_FORCE. When OFF
        # (default) the generation call below and all downstream processing are
        # byte-identical to baseline. When ON, the main generation is capped
        # below the context window and box-absent samples get one short forced
        # continuation before voting.
        if _resolve_answer_force():
            responses = await self.orchestrator.agenerate(
                lm,
                chat_messages.to_batch(budget),
                tools=tools,
                tool_choice=tool_choice,
                usage_accumulator=usage,
                logprobs=True,
                max_tokens=_ANSWER_FORCE_MAIN_MAX_TOKENS,
            )
            responses = await self._force_box_absent_samples(
                lm, chat_messages, responses, usage
            )
        else:
            responses = await self.orchestrator.agenerate(
                lm,
                chat_messages.to_batch(budget),
                tools=tools,
                tool_choice=tool_choice,
                usage_accumulator=usage,
                logprobs=True,
            )

        # process responses and return result
        return self._process_responses(responses, return_response_only, usage)

    async def _force_box_absent_samples(
        self,
        lm: AbstractLanguageModel,
        chat_messages: ChatMessages,
        responses: list[dict],
        usage: GenerationUsage | None,
    ) -> list[dict]:
        """Force a terminal ``\\boxed{}`` on every box-absent sample (H1).

        For each sample lacking a completed ``\\boxed{...}``, issue ONE short
        continuation appended to the assistant turn
        (``draft + CONTINUATION_PRIMER``, ``stop="}"``) and splice
        ``draft + primer + continuation + "}"`` back into the response so the
        existing vote-key extraction picks up a real answer. Samples that already
        carry a box (or are tool calls) are left untouched. Emits
        ``n_forced``/``n_still_truncated`` telemetry via ``logging``.
        """
        original_messages = chat_messages.to_chat_messages()
        is_mcq = _looks_like_mcq(chat_messages.to_prompt())
        primer = CONTINUATION_PRIMER_MCQ if is_mcq else CONTINUATION_PRIMER
        cont_max_tokens = (
            _ANSWER_FORCE_CONT_MAX_TOKENS_MCQ
            if is_mcq
            else _ANSWER_FORCE_CONT_MAX_TOKENS_NUMERIC
        )

        n_forced = 0
        n_still_truncated = 0
        for i, response in enumerate(responses):
            # Tool-call responses vote via signatures, not \boxed{}; skip them.
            if response.get("tool_calls"):
                continue
            draft = extract_content_from_lm_response(response)
            if _has_boxed_answer(draft):
                continue

            n_forced += 1
            cont_messages = [
                *original_messages,
                ChatMessage(role="assistant", content=draft + primer),
            ]
            cont_response = await lm.agenerate_single(
                cont_messages,
                stop=_ANSWER_FORCE_STOP,
                max_completion_tokens=cont_max_tokens,
                usage_accumulator=usage,
            )
            continuation = extract_content_from_lm_response(cont_response)
            completed = draft + primer + continuation + _ANSWER_FORCE_STOP

            # If the forced continuation produced an empty body, the sample still
            # has no real answer (proxy for a continuation that itself truncated).
            if not _has_boxed_answer(completed):
                n_still_truncated += 1

            # Override content so downstream projection re-extracts the
            # now-completed answer. The draft's ``_logprobs`` describe the
            # TRUNCATED draft tokens, not the forced final answer, so they carry
            # no valid confidence for the voted content -- null them out. This
            # makes ``_aggregate_logprob`` return None for the forced sample so
            # it is excluded from the confidence tie-break (ties among forced
            # samples fall back to the existing seeded-random selection). This is
            # pure metadata handling: no LM request parameter changes.
            forced = dict(response)
            forced["content"] = completed
            forced["_logprobs"] = None
            responses[i] = forced

        if n_forced:
            logging.info(
                "SelfConsistency answer-forcing: n_forced=%d n_still_truncated=%d "
                "mcq=%s out of %d samples",
                n_forced,
                n_still_truncated,
                is_mcq,
                len(responses),
            )
        return responses

    def _is_tool_vote_path(self, responses: list[dict]) -> bool:
        """Whether voting routes through tool-call signatures (vs content).

        Mirrors the branch decision in ``_project_responses`` so callers can tell
        which projection space a response list will use without re-projecting.
        Tool-call signatures are already normalized and must NOT be
        text-canonicalized for vote grouping.
        """
        tool_call_count = sum(1 for r in responses if r.get("tool_calls"))
        required_majority = math.ceil(len(responses) / 2)
        has_majority_tool_calls = tool_call_count >= required_majority
        return bool(has_majority_tool_calls and self.tool_vote)

    def _project_responses(self, responses: list[dict]) -> tuple[list[int], list]:
        """Project responses to comparable values for voting.

        Handles tool-call vs content routing based on tool_vote setting.

        Returns:
            (eligible_indices, projected_values) where eligible_indices maps
            back to positions in the original responses list.
        """
        if self._is_tool_vote_path(responses):
            eligible_indices = [
                i for i, r in enumerate(responses) if r.get("tool_calls")
            ]
            projected = [
                self._extract_tool_call_features(responses[i]) for i in eligible_indices
            ]
        else:
            content_indices = [
                i for i, r in enumerate(responses) if not r.get("tool_calls")
            ]
            content_projected = [
                self.consistency_space_projection_func(
                    extract_content_from_lm_response(responses[i])
                )
                for i in content_indices
            ]
            # Answer-bearing eligibility filter: responses whose projection is
            # None/empty/whitespace-only carry no answer and must not win the
            # majority vote (mirrors the tool-call eligibility filter above).
            # If EVERY projection is empty, fall back to the full content set so
            # _process_responses still has at least one candidate (never zero).
            non_empty = [
                (idx, proj)
                for idx, proj in zip(content_indices, content_projected)
                if not self._is_empty_projection(proj)
            ]
            if non_empty:
                eligible_indices = [idx for idx, _ in non_empty]
                projected = [proj for _, proj in non_empty]
            else:
                eligible_indices = content_indices
                projected = content_projected

        return eligible_indices, projected

    @staticmethod
    def _aggregate_logprob(response: dict) -> float | None:
        """Mean per-token log probability of a response, or None if unavailable.

        Reads the OpenAI-format ``_logprobs`` metadata attached by the LM client
        (``choice["logprobs"]`` threaded through to ``response["_logprobs"]``).
        The MEAN (not sum) is used so responses of different token lengths are
        compared on an equal footing -- a summed logprob would systematically
        penalise longer answers regardless of their per-token confidence.
        Returns None when no logprob data is present so the tie-break can skip
        this candidate rather than treat missing data as low confidence.
        """
        lp = response.get("_logprobs")
        if not lp:
            return None
        content = lp.get("content")
        if not content:
            return None
        values = [
            tok["logprob"]
            for tok in content
            if isinstance(tok, dict) and tok.get("logprob") is not None
        ]
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _is_empty_projection(projected) -> bool:
        """Whether a content projection carries no answer.

        A projection is empty (ineligible to win a vote) when it is None, an
        empty/whitespace-only string, or a hierarchical tuple whose every level
        is itself None or empty/whitespace-only. Any other value is answer-bearing.
        """
        if projected is None:
            return True
        if isinstance(projected, str):
            return projected.strip() == ""
        if isinstance(projected, tuple):
            return all(
                level is None or (isinstance(level, str) and level.strip() == "")
                for level in projected
            )
        return False

    def _process_responses(
        self,
        responses: list[dict],
        return_response_only: bool = True,
        usage: GenerationUsage | None = None,
    ) -> dict | SelfConsistencyResult:
        """Process responses and return result."""
        # Warn if tool calls detected but tool_vote not set
        tool_call_count = sum(1 for r in responses if r.get("tool_calls"))
        if tool_call_count > 0 and not self.tool_vote:
            logging.warning(
                f"Detected {tool_call_count}/{len(responses)} responses with tool calls, "
                "but tool_vote is not set. Consider setting tool_vote parameter "
                "(e.g., 'tool_name', 'tool_args', 'tool_hierarchical') for tool call voting."
            )

        eligible_indices, responses_projected = self._project_responses(responses)

        # Formatting-invariant vote keys for COUNTING only: on the content path,
        # group answers that differ merely in presentation (e.g. \boxed{\text{C}}
        # vs \boxed{C} vs (C)) into one vote group before counting. On the
        # tool-call path the raw projections are already canonical signatures and
        # must not be text-canonicalized, so the keys are the projections as-is.
        # Selection still returns a real response index and the_one still returns
        # the full, unmodified selected response for the grader to re-extract.
        if self._is_tool_vote_path(responses):
            vote_keys = responses_projected
        else:
            vote_keys = [_canonicalize_vote_key(proj) for proj in responses_projected]

        # Error if no eligible responses after filtering
        if not eligible_indices:
            raise ValueError(
                f"No eligible responses found after filtering. "
                f"Total responses: {len(responses)}, responses with tool calls: {tool_call_count}. "
                "This typically happens when tool_vote is not set but all responses contain tool calls."
            )

        # Confidence tie-break scores, aligned to the projected/eligible list.
        # Only consulted when the vote is a tie among >=2 answer groups; a clear
        # majority ignores these entirely (byte-unchanged clear-winner path).
        # If no response carries logprobs, pass None so selection stays random.
        tiebreak_scores = [self._aggregate_logprob(responses[i]) for i in eligible_indices]
        if all(score is None for score in tiebreak_scores):
            tiebreak_scores = None

        # Voting rule A/B switch (H5): ITS_SC_VOTE=confidence enables full
        # confidence-weighted voting; default ``plurality`` keeps the current
        # path byte-identical. Resolved here at the call site so a single env var
        # flips both selectors below with no code edits between A/B runs.
        vote_mode = _resolve_vote_mode()

        # Determine if we're dealing with hierarchical (tuple) or flat projections.
        # vote_keys carries the canonical grouping key (aligned to the projected
        # list); selection returns a position into the eligible list either way.
        if responses_projected and isinstance(responses_projected[0], tuple):
            response_counts, filtered_selected_index = (
                _select_hierarchical_most_common_or_random(
                    responses_projected,
                    tiebreak_scores,
                    vote_keys=vote_keys,
                    vote_mode=vote_mode,
                )
            )
        else:
            response_counts, filtered_selected_index = _select_most_common_or_random(
                responses_projected,
                tiebreak_scores,
                vote_keys=vote_keys,
                vote_mode=vote_mode,
            )

        # Map back to original index
        selected_index = eligible_indices[filtered_selected_index]

        # Return result with original responses preserved
        result = SelfConsistencyResult(
            responses=responses,  # ALL original responses
            response_counts=response_counts,
            selected_index=selected_index,  # Index into original responses
            usage=usage,
        )
        return result.the_one if return_response_only else result

    @staticmethod
    def _make_hashable(obj):
        """Recursively convert nested structures to hashable types."""
        if isinstance(obj, dict):
            return tuple(
                sorted((k, SelfConsistency._make_hashable(v)) for k, v in obj.items())
            )
        elif isinstance(obj, list):
            return tuple(SelfConsistency._make_hashable(item) for item in obj)
        elif isinstance(obj, set):
            return tuple(sorted(SelfConsistency._make_hashable(item) for item in obj))
        else:
            return obj

    def _parse_tool_args(self, raw_args) -> tuple:
        """Parse tool call arguments into a hashable tuple.

        Handles JSON strings, applies exclude_args filtering, and converts
        nested structures to hashable types.
        """
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                raw_args = {}
        if not isinstance(raw_args, dict):
            raw_args = {}
        if self.exclude_args:
            raw_args = {k: v for k, v in raw_args.items() if k not in self.exclude_args}
        return self._make_hashable(raw_args) if raw_args else ()

    def _extract_tool_call_features(self, message_obj: dict):
        """Extract tool call features for voting based on tool_vote type."""
        tool_calls = message_obj.get("tool_calls", [])
        if not tool_calls:
            return None if self.tool_vote == "tool_name" else (None, None)

        if self.tool_vote == "tool_flat_all":
            all_features = []
            for tc in tool_calls:
                tc_name = tc.get("function", {}).get("name")
                tc_args = tc.get("function", {}).get("arguments", {})
                args_tuple = self._parse_tool_args(tc_args)
                all_features.append((tc_name, args_tuple))
            return frozenset(all_features)

        first_tc = tool_calls[0]
        function_name = first_tc.get("function", {}).get("name")
        function_args = first_tc.get("function", {}).get("arguments", {})
        args_tuple = self._parse_tool_args(function_args)

        if self.tool_vote == "tool_name":
            return function_name
        elif self.tool_vote == "tool_args":
            return args_tuple
        elif self.tool_vote == "tool_hierarchical":
            return (function_name, args_tuple)
        else:
            raise ValueError(f"Unknown tool_vote type: {self.tool_vote}")


def validate_regex_patterns(patterns: list[str]) -> None:
    """Validate regex patterns before passing to create_regex_projection_function."""
    for p in patterns:
        try:
            re.compile(p)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern {p!r}: {e}") from e


def create_regex_projection_function(
    patterns: str | list[str],
) -> Callable[[str], tuple]:
    """Create a hierarchical projection function from regex pattern(s).

    Args:
        patterns: Single regex pattern string or list of regex patterns.
                 Each pattern should contain capturing groups to extract features.
                 For hierarchical consistency, earlier patterns in the list represent
                 higher hierarchy levels.

    Returns:
        A projection function that takes a response string and returns a tuple
        where each element corresponds to the first match from each pattern.
        If no match is found for a pattern, None is used for that position.

    Example:
        # Single pattern for extracting final answer
        pattern = r'\\\\boxed\\{([^}]+)\\}'
        proj_func = create_regex_projection_function(pattern)
        proj_func("The answer is \\\\boxed{42}") -> ("42",)

        # Multiple patterns for hierarchical consistency
        patterns = [r'Method:\\s*(\\w+)', r'\\\\boxed\\{([^}]+)\\}']
        proj_func = create_regex_projection_function(patterns)
        proj_func("Method: algebra\\n...\\nAnswer: \\\\boxed{42}") -> ("algebra", "42")
    """
    # Ensure patterns is a list
    if isinstance(patterns, str):
        patterns = [patterns]

    # Compile regex patterns for efficiency
    compiled_patterns = [
        re.compile(pattern, re.DOTALL | re.IGNORECASE) for pattern in patterns
    ]

    def projection_function(response: str) -> tuple:
        """Extract features from response using compiled regex patterns."""
        results = []

        # Handle None or empty response
        if response is None:
            response = ""

        for pattern in compiled_patterns:
            match = pattern.search(response)
            if match:
                # If pattern has capturing groups, use the first group
                if match.groups():
                    results.append(match.group(1).strip())
                else:
                    # If no capturing groups, use the entire match
                    results.append(match.group(0).strip())
            else:
                # No match found, use None
                results.append(None)

        return tuple(results)

    return projection_function
