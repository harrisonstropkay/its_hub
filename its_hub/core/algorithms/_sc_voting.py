"""Voting/selection helpers for the self-consistency algorithm family.

Pure, self-contained functions extracted from ``self_consistency.py`` to keep
that module under the project's file-size limit. Behavior is byte-identical to
the in-line versions; ``self_consistency.py`` re-imports these names so existing
imports (including ``weighted_self_consistency`` and the test suite) continue to
resolve unchanged.
"""

import random
from collections import Counter


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
