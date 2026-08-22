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

# --- Completion-status-aware selection (H1) + telemetry (H2), cycle-3 ----------
# Both are OPT-IN and OFF by default so the scored path stays byte-identical to
# baseline. Mirrors the ``ITS_SC_VOTE`` A/B pattern in ``_sc_voting.py``.
#
# Motivation: the dominant scored failure is CONTEXT_TRUNCATION -- the model
# overruns the completion window and emits an unfinished, un-boxed essay. On the
# many GPQA/AIME items where a *subset* of the samples actually finishes-and-boxes
# while the rest truncate, the fixed answer extractor still lets the truncated
# samples carry the vote (with a salvaged/partial answer). H1 restricts the vote
# to the finished-and-boxed subset when such a proper subset exists, so the real,
# completed answer carries the item. This is a purely STRUCTURAL signal (a real
# ``\boxed{...}`` token present AND content length below a truncation cap); it
# reads NO logprobs and is explicitly NOT the exp-11 confidence-weighting NULL.
_COMPLETION_SELECT_ENV_VAR = "ITS_SC_COMPLETION_SELECT"
_LOG_COMPLETION_ENV_VAR = "ITS_SC_LOG_COMPLETION"
_TRUNC_CAP_ENV_VAR = "ITS_SC_TRUNC_CAP_CHARS"
# Default truncation cap in characters. Chosen well ABOVE the completed-response
# length distribution (finished boxed answers are far shorter) and well BELOW the
# observed ~17.9k-20.7k-char truncation regime, so completed and truncated
# samples separate cleanly and content-agnostically. DEV-tunable via the env var.
_DEFAULT_TRUNC_CAP_CHARS = 8000

_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _resolve_completion_select() -> bool:
    """Whether completion-status-aware selection (H1) is enabled.

    Reads ``ITS_SC_COMPLETION_SELECT`` fresh on every call. Any recognized truthy
    value enables it; unset / anything else keeps the baseline behavior, so the
    default scored path is byte-identical to today.
    """
    return (
        os.environ.get(_COMPLETION_SELECT_ENV_VAR, "").strip().lower()
        in _TRUTHY_ENV_VALUES
    )


def _resolve_log_completion() -> bool:
    """Whether per-sample completion-status logging (H2) is enabled.

    Enabled when either ``ITS_SC_COMPLETION_SELECT`` (so a selection A/B run is
    always auditable) or the dedicated ``ITS_SC_LOG_COMPLETION`` gate is truthy.
    Default OFF -> no log volume.
    """
    if _resolve_completion_select():
        return True
    return (
        os.environ.get(_LOG_COMPLETION_ENV_VAR, "").strip().lower()
        in _TRUTHY_ENV_VALUES
    )


def _resolve_trunc_cap() -> int:
    """Read the truncation cap (chars) from ``ITS_SC_TRUNC_CAP_CHARS``.

    Falls back to ``_DEFAULT_TRUNC_CAP_CHARS`` when unset, non-integer, or
    non-positive so a malformed override can never disable the length guard.
    """
    raw = os.environ.get(_TRUNC_CAP_ENV_VAR)
    if raw is None:
        return _DEFAULT_TRUNC_CAP_CHARS
    try:
        val = int(raw.strip())
    except (TypeError, ValueError):
        return _DEFAULT_TRUNC_CAP_CHARS
    return val if val > 0 else _DEFAULT_TRUNC_CAP_CHARS


def _has_real_boxed(content: str) -> bool:
    """Whether ``content`` contains an actual, CLOSED ``\\boxed{...}`` token.

    Structural completion marker: a salvaged mid-stream option letter (what the
    fixed extractor falls back to on a truncated essay) has no ``\\boxed{`` token,
    so this cleanly distinguishes a finished answer from a truncated one. We
    require a brace-balanced close after ``\\boxed{`` so a box that was cut off
    mid-emission by truncation (``...\\boxed{`` at end of stream) does NOT count
    as completed. The last occurrence is scanned so trailing final answers win.
    """
    if not content:
        return False
    token = "\\boxed{"
    start = content.rfind(token)
    if start == -1:
        return False
    # Scan from the opening brace of the boxed token, tracking brace depth so
    # nested LaTeX (e.g. \boxed{\frac{1}{2}}) is handled correctly.
    depth = 0
    for ch in content[start + len(token) - 1 :]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return True
    return False


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

            # Completion-status refinement (H1), layered on the same
            # "if proper-subset else full set" guard shape as above so the
            # never-zero-candidate contract is preserved. Only active when
            # ITS_SC_COMPLETION_SELECT is on; otherwise this is a no-op and the
            # returned lists are byte-identical to the answer-bearing filter.
            if _resolve_completion_select():
                eligible_indices, projected = self._filter_completed(
                    eligible_indices, projected, responses
                )

        return eligible_indices, projected

    @staticmethod
    def _is_completed_response(response: dict, cap: int) -> bool:
        """Structural per-response completion signal for H1.

        ``completed = has_real_boxed(content) AND char_len(content) < cap``.
        Reads the RAW response content (no logprobs, no ground truth): a real,
        closed ``\\boxed{...}`` token present AND the content shorter than the
        truncation cap. Both conditions are required so a box emitted early
        followed by a truncated over-run is not mistaken for a finished answer.
        """
        content = extract_content_from_lm_response(response)
        return _has_real_boxed(content) and len(content) < cap

    def _filter_completed(
        self,
        eligible_indices: list[int],
        projected: list,
        responses: list[dict],
    ) -> tuple[list[int], list]:
        """Restrict the vote set to the finished-and-boxed subset when one exists.

        Operates on the answer-bearing (eligible) content responses, keeping the
        ``eligible_indices``/``projected`` lists positionally aligned so index
        mapping stays correct through BOTH the flat and tuple/hierarchical
        projection paths downstream. The restriction fires ONLY when a PROPER
        SUBSET is completed (>=1 completed AND >=1 not completed):

        - all-samples-complete -> subset == full set -> no filtering
          (guarantee (a): byte-identical to baseline);
        - all-samples-truncate -> completed subset empty -> keep the full set
          (guarantee (b): degenerate to the current uniform fallback,
          never zero candidates).
        """
        cap = _resolve_trunc_cap()
        completed_mask = [
            self._is_completed_response(responses[i], cap) for i in eligible_indices
        ]
        n_completed = sum(completed_mask)
        if 0 < n_completed < len(eligible_indices):
            eligible_indices = [
                idx for idx, done in zip(eligible_indices, completed_mask) if done
            ]
            projected = [
                proj for proj, done in zip(projected, completed_mask) if done
            ]
        return eligible_indices, projected

    def _log_completion_status(
        self, responses: list[dict], selected_index: int
    ) -> None:
        """Emit ONE compact structured record of the completion partition (H2).

        Telemetry only -- makes the completed/truncated split H1 acts on auditable
        from replay logs without re-running the GPU. Recomputes its own view of the
        partition (it does not mutate any selection state) and logs at INFO under a
        stable ``sc_completion_status`` event key for grep/JSON parsing.
        """
        cap = _resolve_trunc_cap()
        per_sample = []
        for r in responses:
            content = extract_content_from_lm_response(r)
            per_sample.append(
                {"has_box": _has_real_boxed(content), "char_len": len(content)}
            )

        def _completed(entry: dict) -> bool:
            return entry["has_box"] and entry["char_len"] < cap

        n_samples = len(responses)
        n_completed = sum(1 for entry in per_sample if _completed(entry))
        n_truncated = n_samples - n_completed
        # Empty-projection count on the content path (mirrors the answer-bearing
        # eligibility filter) -- how many samples carry no votable answer at all.
        n_empty_projection = sum(
            1
            for r in responses
            if not r.get("tool_calls")
            and self._is_empty_projection(
                self.consistency_space_projection_func(
                    extract_content_from_lm_response(r)
                )
            )
        )
        selected_completed = (
            0 <= selected_index < n_samples and _completed(per_sample[selected_index])
        )
        record = {
            "event": "sc_completion_status",
            "n_samples": n_samples,
            "n_completed": n_completed,
            "n_truncated": n_truncated,
            "n_empty_projection": n_empty_projection,
            "selected_index": selected_index,
            "selected_completed": selected_completed,
            "cap_chars": cap,
            "per_sample": per_sample,
        }
        logging.info("sc_completion_status %s", json.dumps(record))

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

        # Per-sample completion-status telemetry (H2), gated OFF by default.
        # Pure logging: selection is UNCHANGED by this block.
        if _resolve_log_completion():
            self._log_completion_status(responses, selected_index)

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
