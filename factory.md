# factory.md — ITS Autoresearch Loop

This file is the governance contract for the Software Factory's design-mode loop on
`its_hub`. It declares what the loop optimizes, how it is scored, what it may edit,
and what would count as cheating. Read it before running any research cycle.

## Project Eval

- **Metric:** `accuracy` (composite math + hard-science benchmark accuracy)
- **Command:** `python eval/score.py`
- **Parser:** `json` — the command prints `{"results": [{name, score, weight, passed, details}, ...]}`;
  the project metric is the entry with `"name": "accuracy"`.
- **Definition:** `score = clamp(0.5 * mean(MATH500_acc, AIME_2024_acc) + 0.5 * GPQA_Diamond_acc, 0, 1)` —
  math and science are weighted equally so the loop cannot win on math alone.
- **Cost discipline:** fixed per-experiment budget (`ITS_BUDGET` ∈ {4, 8}, default 4) and a hard
  per-benchmark wall-clock cap (`ITS_EVAL_TIMEOUT`, default 1200s). The eval degrades to
  `score=0.0, passed=False` (never raises) when no model endpoint is reachable.

## Eval Weights

- **project:** 0.50  (the `accuracy` metric above — project-dominant)
- **hygiene:** 0.30  (tests, lint, coverage)
- **growth:**  0.20  (capability surface / observability)

## Research Target

- **Objective:** raise math (MATH500, AIME-2024) and hard-science (GPQA-Diamond) accuracy by
  improving *how* `its_hub` scales inference — its prompts, sampling, step configuration, and
  scaling algorithms — with no human in the loop.
- **Metric:** `accuracy` (see Project Eval).
- **Run command:** `python eval/score.py`
- **Model:** `Qwen/Qwen2.5-Math-7B-Instruct`, fixed for every experiment, served by vLLM
  (data-parallel across all available GPUs) at the endpoint named by `ITS_ENDPOINT`.

## Mutable Surfaces

The loop MAY edit these — this is where legitimate ITS research happens:

- `its_hub/core/algorithms/*` — the scaling algorithms (self-consistency, best-of-N, beam
  search, particle filtering) and any new algorithm variants.
- Prompts in `its_hub/core/utils.py` (e.g. `QWEN_SYSTEM_PROMPT`, `SAL_STEP_BY_STEP_SYSTEM_PROMPT`).
- Sampling / temperature and step configuration in `its_hub/core/lms/step_generation.py`
  (e.g. `tokens_per_step` vs `step_token`, `max_steps`, stop tokens).
- Answer-extraction and voting logic inside the algorithm files.

## Fixed Surfaces

The loop MUST NOT edit these — they define the metric, and editing them would improve the
*measurement* instead of the *algorithms*:

- `benchmarking/**` — the benchmark harness (loaders, graders, CLI).
- The datasets themselves: MATH500, AIME-2024, GPQA-Diamond.
- `tests/**` — the test suite.
- The scoring path in `eval/score.py` (`eval_accuracy` and its helpers).

### Build-time vs loop-time inversion (READ THIS)

The three surfaces above (`benchmarking/**`, `tests/**`, and the `eval/score.py` scoring path)
were **edited during the build PR that created this substrate** — they are the deliverables that
stood up the GPQA path, the `accuracy` metric, and their tests. **The moment the autoresearch
loop begins, those exact paths become FIXED / LOCKED.** The inversion is deliberate: build the
metric once, then freeze it so the loop "improves the math, not the metric." Any diff that touches
a Fixed Surface during a research cycle must be reverted.

## Seedling idea

Start with the step-generation algorithms, favoring variants that spend a **fixed number of
tokens per step** (`tokens_per_step`) over ones that split on natural step boundaries
(`step_token`). Fixed per-step budgets should generalize to domains like GPQA-Diamond that have
no clean notion of a "step", so gains ought to carry across both the math and science halves of
the metric. `benchmarking/benchmark.py` already exposes `--tokens_per_step`, and `eval_accuracy`
honors `ITS_TOKENS_PER_STEP`. This is the first direction, **not a rule** — the loop is free to
change directions based on the evidence.

## Anti-cheat rules

Before keeping any change, inspect the diff and **revert** if the agent:

- reads dataset labels / ground truth (the benchmark answers) to inform its implementation;
- hard-codes benchmark outputs or memorizes answers instead of doing legitimate ITS research;
- imports `benchmarking/` or the datasets into a mutable-surface (editable) file — the scoring
  path deliberately drives the harness only at the `subprocess` boundary to keep this leakage
  channel closed;
- tunes on the scored TEST subsets. The dev/test split is disjoint by design (documented in
  `eval/score.py`): TEST = MATH500 `:20`, GPQA `:20`, AIME `:15`; the loop may only iterate on
  the reserved DEV slices (MATH500 `20:60`, GPQA `20:60`, AIME `15:30`).

## Hypothesis Budget

- **min_growth:** 1   — each cycle must attempt at least one genuine capability/algorithm change.
- **max_new:** 3      — cap new hypotheses per cycle so iteration stays disciplined and cheap.
