"""Voting/selection helpers for the self-consistency algorithm family.

Pure, self-contained functions extracted from ``self_consistency.py`` to keep
that module under the project's file-size limit. Behavior is byte-identical to
the in-line versions; ``self_consistency.py`` re-imports these names so existing
imports (including ``weighted_self_consistency`` and the test suite) continue to
resolve unchanged.
"""

import math
import os
import random
from collections import Counter

# A/B switch for the self-consistency voting rule. ``plurality`` (the default)
# is the historical max-count vote with the H2 confidence tie-break; the env var
# ``ITS_SC_VOTE`` may switch it to ``confidence`` for full confidence-weighted
# voting (H5). Resolved at the selection call site so a single env var flips both
# the flat and hierarchical selectors with NO code edits between A/B runs. Any
# unrecognized value resolves to ``plurality`` so the default path is unchanged.
_VOTE_MODE_ENV_VAR = "ITS_SC_VOTE"


def _resolve_vote_mode() -> str:
    """Read the voting mode from ``ITS_SC_VOTE`` (default ``plurality``).

    Returns ``"confidence"`` only for an explicit, case-insensitive
    ``confidence`` value; every other value (including unset) returns
    ``"plurality"`` so the default behavior is byte-identical to today.
    """
    mode = os.environ.get(_VOTE_MODE_ENV_VAR, "plurality").strip().lower()
    return "confidence" if mode == "confidence" else "plurality"


def _has_confidence_scores(tiebreak_scores: list[float | None] | None) -> bool:
    """Whether confidence weighting can run: at least one real logprob score.

    Confidence mode requires positive-weight-bearing candidates; when scores are
    entirely absent (``None`` or every entry ``None``) the caller must fall back
    to the plurality path rather than degenerate.
    """
    return tiebreak_scores is not None and any(
        s is not None for s in tiebreak_scores
    )


def _confidence_weight(mean_logprob: float) -> float:
    """Convert a mean-per-token logprob into a POSITIVE confidence weight.

    ``tiebreak_scores`` hold mean-per-token log-probabilities, which are
    NEGATIVE. Summing those raw values per answer group would reward SMALLER
    groups (fewer negative terms) -- the exact opposite of what a vote needs. We
    instead map each candidate to ``exp(mean_logprob)``: the geometric-mean
    per-token probability of its answer, a value in ``(0, 1]``. Because the
    weight is bounded above by 1, a single hyper-confident outlier (weight -> 1)
    cannot outvote a larger group of moderately-confident members whose weights
    sum above 1. This is the confidence-weighted-majority-vote formulation of
    CISC (Taubenfeld et al., "Confidence Improves Self-Consistency in LLMs",
    arXiv:2502.06233): the vote weight of an answer is the sum of its members'
    confidence scores. CISC also describes a softmax over candidates; that shares
    one positive denominator across every group, so it leaves the arg-max
    unchanged -- hence we use the un-normalized ``exp`` directly.
    """
    return math.exp(mean_logprob)


def _select_by_confidence_weight(
    keys: list,
    tiebreak_scores: list[float | None],
) -> int:
    """Return the index of the winning candidate under confidence weighting.

    Each candidate contributes ``_confidence_weight(mean_logprob)`` to its
    canonical answer group (``keys[i]``). The group with the greatest summed
    weight wins. Determinism: group-weight ties break toward the group whose
    first-seen member has the lowest index, and within the winning group the
    member is chosen by the existing confidence tie-break (highest individual
    logprob, lowest index) -- so the result never depends on ``random``.
    Candidates lacking a logprob score contribute zero weight (they cannot swing
    the vote) while remaining eligible members of their group.
    """
    group_weight: dict = {}
    group_first_index: dict = {}
    for i, key in enumerate(keys):
        if key not in group_weight:
            group_weight[key] = 0.0
            group_first_index[key] = i
        score = tiebreak_scores[i] if i < len(tiebreak_scores) else None
        if score is not None:
            group_weight[key] += _confidence_weight(score)

    max_weight = max(group_weight.values())
    winners = [k for k, w in group_weight.items() if w == max_weight]
    winning_key = min(winners, key=lambda k: group_first_index[k])

    member_indices = [i for i, key in enumerate(keys) if key == winning_key]
    return _tiebreak_by_confidence(member_indices, tiebreak_scores)


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
    vote_keys: list | None = None,
    vote_mode: str = "plurality",
) -> tuple[Counter, int]:
    # Group votes on ``vote_keys`` when provided (formatting-invariant canonical
    # keys aligned positionally to ``list_to_select_from``) so that variants that
    # denote the same answer merge into one group before counting. When None,
    # counting is byte-identical to the raw-projection behavior. The returned
    # index is always a position into ``list_to_select_from``.
    keys = vote_keys if vote_keys is not None else list_to_select_from

    # count occurrences of each element
    counts = Counter(keys)

    # Full confidence-weighted voting (H5, CISC arXiv:2502.06233): when enabled
    # via ITS_SC_VOTE=confidence, select the answer group with the maximum
    # aggregate POSITIVE confidence weight instead of the maximum raw count.
    # Reported ``counts`` stay frequency-based (reporting contract unchanged);
    # only the selection changes. Falls back to the plurality path below when no
    # logprob scores are available, so behavior degrades gracefully.
    if vote_mode == "confidence" and _has_confidence_scores(tiebreak_scores):
        return counts, _select_by_confidence_weight(keys, tiebreak_scores)

    # find the element with maximum occurrences
    max_count = max(counts.values())

    # find indices of the most common elements
    most_common_indices = [i for i, r in enumerate(keys) if counts[r] == max_count]

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
    vote_keys: list | None = None,
    vote_mode: str = "plurality",
) -> tuple[Counter, int]:
    if not list_to_select_from:
        raise ValueError("Cannot select from empty list")

    # Group/level votes on ``vote_keys`` (formatting-invariant canonical tuples
    # aligned positionally) when provided; otherwise on the raw projections,
    # byte-identical to the prior behavior. The returned index is always a
    # position into ``list_to_select_from``.
    keys = vote_keys if vote_keys is not None else list_to_select_from

    # Full confidence-weighted voting (H5, CISC arXiv:2502.06233): when enabled
    # via ITS_SC_VOTE=confidence, aggregate a positive confidence weight per full
    # canonical answer tuple and select the max-weight group. Grouping on the
    # whole tuple keeps distinct (name, args) tool signatures as separate answer
    # groups. Falls back to the plurality hierarchy below when scores are
    # unavailable, so behavior degrades gracefully.
    if vote_mode == "confidence" and _has_confidence_scores(tiebreak_scores):
        return Counter(keys), _select_by_confidence_weight(keys, tiebreak_scores)

    # If all elements are single-element tuples, fall back to flat behavior
    if all(len(item) == 1 for item in keys):
        flat_list = [item[0] for item in keys]
        _, selected_index = _select_most_common_or_random(flat_list, tiebreak_scores)
        # Convert back to tuple format for consistency
        tuple_counts = Counter(keys)
        return tuple_counts, selected_index

    # Find the maximum hierarchy depth
    max_depth = max(len(item) for item in keys)

    # Start with all indices as candidates
    candidate_indices = list(range(len(keys)))

    # Process each level of the hierarchy
    for level in range(max_depth):
        # Get the values at this level for current candidates
        level_values = []
        valid_indices = []

        for idx in candidate_indices:
            item = keys[idx]
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
    distinct_survivors = {keys[idx] for idx in candidate_indices}
    if len(distinct_survivors) >= 2 and tiebreak_scores is not None:
        selected_index = _tiebreak_by_confidence(candidate_indices, tiebreak_scores)
    else:
        selected_index = random.choice(candidate_indices)

    # Count all vote groups for the result
    tuple_counts = Counter(keys)

    return tuple_counts, selected_index
