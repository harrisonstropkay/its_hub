import json
import logging
import math
import random
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
from its_hub.core.orchestrator import LMOrchestrator
from its_hub.core.utils import extract_content_from_lm_response

# Default tool-call voting strategy for the self-consistency family. Chosen so
# that responses containing tool calls vote sensibly out of the box (the primary
# use case for the IaaS gateway). Pass tool_vote=None to force content-only
# voting, which raises if every response is a tool call.
DEFAULT_TOOL_VOTE = "tool_hierarchical"


def _default_projection_func(response: str) -> str:
    """Default projection function that uses exact content matching.
    This function strips whitespace and returns the content as-is for voting.
    Responses with identical content (after stripping) will be considered equivalent.
    Args:
        response: The response content string to project.
    Returns:
        The stripped response content.
    """
    return response.strip()


@dataclass
class SelfConsistencyResult(AbstractScalingResult):
    responses: list[dict]  # Keep original message format with tool calls
    response_counts: Counter[str] | Counter[tuple] | Counter
    selected_index: int
    usage: GenerationUsage | None = None

    @property
    def the_one(self) -> dict:
        return self.responses[self.selected_index]


def _tiebreak_by_confidence(
    candidate_indices: list[int],
    tiebreak_scores: list[float | None],
) -> int:
    """Pick the candidate with the highest aggregate token log probability.

    Used only when the majority vote is a genuine tie among >=2 distinct answer
    groups. ``candidate_indices`` are positions into the projected/eligible list
    and ``tiebreak_scores`` is aligned to that same list (see
    ``SelfConsistency._process_responses``). Higher aggregate logprob = more
    confident. Ties in score are broken deterministically toward the lowest
    index so the result never depends on ``random``. Candidates lacking a score
    (None) are ignored; if none of the tied candidates carry a score, we fall
    back to a random pick to preserve the never-zero-candidate guarantee.
    """
    scored = [
        (tiebreak_scores[i], i)
        for i in candidate_indices
        if i < len(tiebreak_scores) and tiebreak_scores[i] is not None
    ]
    if not scored:
        return random.choice(candidate_indices)
    best_score = max(score for score, _ in scored)
    return min(i for score, i in scored if score == best_score)


def _select_most_common_or_random(
    list_to_select_from: list[str],
    tiebreak_scores: list[float | None] | None = None,
) -> tuple[Counter, int]:
    # count occurrences of each element
    counts = Counter(list_to_select_from)

    # find the element with maximum occurrences
    max_count = max(counts.values())

    # find indices of the most common elements
    most_common_indices = [
        i for i, r in enumerate(list_to_select_from) if counts[r] == max_count
    ]

    # A "tie" is >=2 DISTINCT answer groups sharing the top count. When a single
    # group holds the top count (clear majority) behavior is byte-unchanged: a
    # random member of that group is returned exactly as before. Only on a true
    # tie among groups do we defer to the confidence tie-break, when available.
    top_group_count = sum(1 for c in counts.values() if c == max_count)
    if top_group_count >= 2 and tiebreak_scores is not None:
        selected_index = _tiebreak_by_confidence(most_common_indices, tiebreak_scores)
    else:
        # select a random index from the most common ones
        # note above implementation ensures that if there are multiple
        #      elements with the same count, a random one is selected
        selected_index = random.choice(most_common_indices)

    return counts, selected_index


def _select_hierarchical_most_common_or_random(
    list_to_select_from: list[tuple],
    tiebreak_scores: list[float | None] | None = None,
) -> tuple[Counter, int]:
    if not list_to_select_from:
        raise ValueError("Cannot select from empty list")

    # If all elements are single-element tuples, fall back to flat behavior
    if all(len(item) == 1 for item in list_to_select_from):
        flat_list = [item[0] for item in list_to_select_from]
        _, selected_index = _select_most_common_or_random(flat_list, tiebreak_scores)
        # Convert back to tuple format for consistency
        tuple_counts = Counter(list_to_select_from)
        return tuple_counts, selected_index

    # Find the maximum hierarchy depth
    max_depth = max(len(item) for item in list_to_select_from)

    # Start with all indices as candidates
    candidate_indices = list(range(len(list_to_select_from)))

    # Process each level of the hierarchy
    for level in range(max_depth):
        # Get the values at this level for current candidates
        level_values = []
        valid_indices = []

        for idx in candidate_indices:
            item = list_to_select_from[idx]
            if level < len(item):
                level_values.append(item[level])
                valid_indices.append(idx)

        if not level_values:
            break

        # Count occurrences at this level
        level_counts = Counter(level_values)
        max_count = max(level_counts.values())

        # Filter candidates to only those with the most common value at this level
        new_candidates = []
        for i, idx in enumerate(valid_indices):
            if level_counts[level_values[i]] == max_count:
                new_candidates.append(idx)

        candidate_indices = new_candidates

        # If we have a unique winner, we can stop
        if len(candidate_indices) == 1:
            break

    # Select from the remaining candidates. If they map to >=2 distinct answer
    # tuples this is a genuine tie among groups -> defer to the confidence
    # tie-break when scores are available. If they are all the same tuple (a
    # clear winner with multiple identical members) behavior is byte-unchanged:
    # a random member is returned exactly as before.
    distinct_survivors = {list_to_select_from[idx] for idx in candidate_indices}
    if len(distinct_survivors) >= 2 and tiebreak_scores is not None:
        selected_index = _tiebreak_by_confidence(candidate_indices, tiebreak_scores)
    else:
        selected_index = random.choice(candidate_indices)

    # Count all original tuples for the result
    tuple_counts = Counter(list_to_select_from)

    return tuple_counts, selected_index


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

    def _project_responses(self, responses: list[dict]) -> tuple[list[int], list]:
        """Project responses to comparable values for voting.

        Handles tool-call vs content routing based on tool_vote setting.

        Returns:
            (eligible_indices, projected_values) where eligible_indices maps
            back to positions in the original responses list.
        """
        tool_call_count = sum(1 for r in responses if r.get("tool_calls"))
        required_majority = math.ceil(len(responses) / 2)
        has_majority_tool_calls = tool_call_count >= required_majority

        if has_majority_tool_calls and self.tool_vote:
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

        # Determine if we're dealing with hierarchical (tuple) or flat projections
        if responses_projected and isinstance(responses_projected[0], tuple):
            response_counts, filtered_selected_index = (
                _select_hierarchical_most_common_or_random(
                    responses_projected, tiebreak_scores
                )
            )
        else:
            response_counts, filtered_selected_index = _select_most_common_or_random(
                responses_projected, tiebreak_scores
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
