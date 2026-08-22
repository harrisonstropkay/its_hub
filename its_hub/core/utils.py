import json
import re
import warnings

# the system prompt for step-by-step reasoning taken from https://github.com/huggingface/search-and-learn
SAL_STEP_BY_STEP_SYSTEM_PROMPT = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."

QWEN_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# --- Answer/budget forcing continuation strings (H1, ITS_SC_ANSWER_FORCE) -----
# Appended to the ASSISTANT turn to force a terminal ``\boxed{}`` on samples that
# exhausted the context window before emitting one. This is a CONTINUATION STRING
# on the assistant turn -- it is explicitly NOT a ``QWEN_SYSTEM_PROMPT`` rewrite
# and touches no system prompt (distinguishing it from the refuted exp-1/exp-13
# system-prompt brevity lever). Paired with ``stop="}"`` so the closing brace
# terminates the forced answer cleanly.
CONTINUATION_PRIMER = "\n\nFinal answer: \\boxed{"
# MCQ/GPQA-shaped items force a single option letter. The primer string is the
# same as the numeric one -- only the continuation token budget differs (tighter
# for MCQ, set at the call site) -- but a distinct named constant keeps the MCQ
# path explicit and independently tunable.
CONTINUATION_PRIMER_MCQ = "\n\nFinal answer: \\boxed{"


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


# --- Vote-key canonicalization (VOTING ONLY) -------------------------------
# LaTeX display/formatting wrappers that carry no answer content and should be
# unwrapped before comparing votes: ``\text{C}`` / ``\mathrm{C}`` / ``\mathbf{C}``
# / ``\textbf{C}`` / ``\mathit{C}`` all denote the same answer ``C``.
_LATEX_WRAPPER_RE = re.compile(r"\\(?:text|textbf|mathrm|mathbf|mathit)\s*\{([^{}]*)\}")
# An isolated single option-letter answer in any common decoration:
#   ``C``  ``(C)``  ``C.``  ``C)``  ``[C]``  (optionally surrounded by whitespace).
_ISOLATED_OPTION_LETTER_RE = re.compile(r"^[(\[]?\s*([A-Za-z])\s*[).\]]?$")


def _canonicalize_vote_key(projected):
    """Return a formatting-invariant vote key for an already-projected answer.

    Used **only** for grouping votes in the self-consistency family so that
    formatting-variant projections that denote the *same* answer merge into one
    vote group before counting. It is deliberately CONSERVATIVE and
    domain-general: it strips presentation wrappers/whitespace and normalizes an
    isolated option-letter's decoration/case, but it NEVER rewrites free-form
    numeric or expression content (e.g. ``1/2`` and ``0.5``, or ``x=2`` and
    ``2``, stay distinct). When in doubt it returns the raw projection.

    This does not touch the responses themselves: the grader still independently
    re-extracts the final answer from the full, unmodified selected response.

    Non-string inputs are returned unchanged; tuples are canonicalized
    element-wise so hierarchical (regex) projections are handled uniformly.
    """
    if isinstance(projected, tuple):
        return tuple(_canonicalize_vote_key(part) for part in projected)
    if not isinstance(projected, str):
        return projected

    original = projected
    s = projected.strip()
    if s == "":
        return s

    # 1. Strip LaTeX formatting wrappers (repeatedly, to unwrap nesting such as
    #    ``\text{\mathbf{C}}``) and surrounding math-mode ``$...$`` delimiters and
    #    outer braces/whitespace. Brace stripping is guarded so we never collapse
    #    a structured answer (e.g. a set/tuple ``{1,2}``) into its interior.
    prev = None
    while prev != s:
        prev = s
        s = _LATEX_WRAPPER_RE.sub(r"\1", s).strip()
        if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
            s = s[1:-1].strip()
        if (
            len(s) >= 2
            and s[0] == "{"
            and s[-1] == "}"
            and "{" not in s[1:-1]
            and "}" not in s[1:-1]
            and "," not in s[1:-1]
        ):
            s = s[1:-1].strip()

    # 2. Normalize an isolated single option-letter answer to a bare uppercase
    #    letter (``\text{C}`` -> ``C``, ``(C)`` -> ``C``, ``c`` -> ``C``). Case
    #    is collapsed ONLY here, where it is provably formatting.
    match = _ISOLATED_OPTION_LETTER_RE.match(s)
    if match:
        return match.group(1).upper()

    # 3. Never canonicalize a non-empty answer down to nothing; if wrapper/brace
    #    stripping emptied it, keep the raw (stripped) projection instead.
    if s == "":
        return original.strip()
    return s


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
