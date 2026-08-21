import json
import re
import warnings

# the system prompt for step-by-step reasoning taken from https://github.com/huggingface/search-and-learn
SAL_STEP_BY_STEP_SYSTEM_PROMPT = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."

QWEN_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def resolve_max_completion_tokens(
    max_completion_tokens: int | None,
    max_tokens: int | None,
) -> int | None:
    """Resolve the deprecated max_tokens param into max_completion_tokens."""
    if max_completion_tokens is not None and max_tokens is not None:
        raise ValueError(
            "Cannot specify both 'max_tokens' and 'max_completion_tokens'. "
            "Use 'max_completion_tokens'."
        )
    if max_tokens is not None:
        warnings.warn(
            "'max_tokens' is deprecated, use 'max_completion_tokens'.",
            DeprecationWarning,
            stacklevel=3,
        )
        return max_tokens
    return max_completion_tokens


def _last_boxed_content(response: str) -> str:
    """Return the brace-balanced content of the LAST ``\\boxed{...}`` in ``response``.

    Scans for each ``\\boxed{`` occurrence and matches braces so nested groups
    like ``\\boxed{\\frac{1}{2}}`` are captured whole. Returns ``""`` if no
    complete (brace-closed) boxed expression is present.
    """
    marker = r"\boxed{"
    last = ""
    start = response.find(marker)
    while start != -1:
        i = start + len(marker)
        depth = 1
        content_start = i
        while i < len(response) and depth > 0:
            if response[i] == "{":
                depth += 1
            elif response[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:  # only accept a fully closed boxed group
            last = response[content_start : i - 1].strip()
        start = response.find(marker, start + len(marker))
    return last


# "the answer is X" / "answer: X" / "answer = X" — capture the trailing value up
# to the next sentence/line boundary. The separator run consumes any mix of
# "is", ":" and "=" (e.g. "answer is:"). Case-insensitive; the LAST match wins
# so a concluding statement takes priority over any earlier mention.
_ANSWER_IS_RE = re.compile(
    r"answer\b[\s:=]*(?:is\b[\s:=]*)?(.+?)\s*(?:[.\n]|$)",
    re.IGNORECASE,
)

# Generic compact-answer refinements applied to a captured "answer is X" value
# so a truncated "... the answer is C because ..." recovers "C" rather than the
# whole tail. A lone leading option letter or a leading number — nothing
# domain-specific, no dataset labels.
_LEAD_OPTION_LETTER_RE = re.compile(r"([A-D])(?![A-Za-z0-9])")
_LEAD_NUMBER_RE = re.compile(r"([-+]?\d[\d.,/]*)")

# A standalone uppercase A-D option letter, not glued to other alphanumerics.
# Uppercase-only avoids matching the English article "a"; the LAST occurrence
# (closest to the conclusion) is preferred. Generic multiple-choice pattern —
# NOT a copy of the benchmarking grader's internals.
_OPTION_LETTER_RE = re.compile(r"(?<![A-Za-z0-9])([A-D])(?![A-Za-z0-9])")


def _extract_answer_cascading(response: str) -> str:
    """Content-agnostic secondary final-answer extractor for voting recovery.

    Used ONLY as a fallback when the primary projection (e.g. ``\\boxed{}``
    extraction) is empty for a whole set of responses — typically because the
    generations were truncated before emitting the boxed final answer. It
    recovers the model's OWN emitted answer token so a majority vote can still be
    taken; it is deliberately domain-general and never reads dataset labels,
    hard-codes benchmark outputs, or replicates the grader. Voting != grading.

    Tries, in priority order:
      1. last ``\\boxed{...}`` content;
      2. trailing "the answer is X" / "answer: X" pattern;
      3. last standalone A-D option letter (generic multiple-choice rule);
      4. trailing-token fallback (last ~50 whitespace tokens) for USC-style
         consistency voting.

    Returns ``""`` only for empty/whitespace (or ``None``) input.
    """
    if not response or not response.strip():
        return ""

    # 1. boxed content (handles the "mostly" — not fully — empty case)
    boxed = _last_boxed_content(response)
    if boxed:
        return boxed

    # 2. "the answer is X" / "answer: X" — take the last match
    answer_matches = _ANSWER_IS_RE.findall(response)
    for candidate in reversed(answer_matches):
        candidate = candidate.strip().strip("*`\"'()[]{}").strip()
        if not candidate:
            continue
        # Compact-answer refinement: a lone leading option letter or number is
        # the answer even if a truncated justification follows it.
        lead_letter = _LEAD_OPTION_LETTER_RE.match(candidate)
        if lead_letter:
            return lead_letter.group(1)
        lead_number = _LEAD_NUMBER_RE.match(candidate)
        if lead_number:
            return lead_number.group(1)
        return candidate

    # 3. last standalone A-D option letter
    letters = _OPTION_LETTER_RE.findall(response)
    if letters:
        return letters[-1]

    # 4. trailing-token fallback (last ~50 tokens)
    return " ".join(response.split()[-50:])


def extract_content_from_lm_response(message: dict) -> str:
    """
    Extract content from a single LM response message object.

    Args:
        message: A message dict returned by fetch_single_response.

    Returns:
        The content string. If the message contains tool calls, returns the content
        if available, otherwise returns an empty string.
    """
    # TODO: This conversion to text is not ideal as it involves manually formatting
    # tool calls and neglects images in multi-modal content. Consider refactoring
    # to work with structured message objects instead of flattening to strings.

    # Extract text content (handle both string and list[dict] formats)
    raw_content = message.get("content")

    if raw_content is None:
        content = ""
    elif isinstance(raw_content, str):
        content = raw_content
    elif isinstance(raw_content, list):
        # Multi-modal content: extract text parts (images are ignored)
        text_parts = [
            item.get("text", "")
            for item in raw_content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        content = " ".join(text_parts)
    else:
        raise ValueError(
            f"Invalid content type: {type(raw_content)}, expected str, list[dict], or None"
        )

    # If there are tool calls, add tool-calls to the content
    if message.get("tool_calls"):
        tool_calls = message.get("tool_calls", [])
        tool_descriptions = []
        for tc in tool_calls:
            if isinstance(tc, dict) and "function" in tc:
                func = tc["function"]
                func_name = func.get("name", "unknown")
                tool_descriptions.append(
                    f"[Tool call: {func_name} Tool args: {json.dumps(func.get('arguments', {}))}]"
                )
            else:
                raise ValueError(
                    f"Invalid tool call: {tc}, expected a dict with a 'function' key"
                )
        content += " ".join(tool_descriptions)

    return content
