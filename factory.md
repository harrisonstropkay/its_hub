# factory.md — ITS Autoresearch Loop

This file is the governance contract for the Software Factory's design-mode loop on
`its_hub`. It declares what the loop optimizes, how it is scored, what it may edit,
and what would count as cheating. Read it before running any research cycle.

## Goal

Raise composite math (MATH500, AIME-2024) + hard-science (GPQA-Diamond) accuracy by improving its_hub inference-time scaling, no human in the loop.

## Command

python eval/score.py

## Project Eval

- name: accuracy
  command: python eval/score.py
  parse: json
  weight: 1.0
  timeout: 1200
  description: composite math+science benchmark accuracy (0.5*mean(math500,aime) + 0.5*gpqa)

## Eval Weights

- project: 0.50
- hygiene: 0.30
- growth: 0.20

## Research Target

- objective: raise math (MATH500, AIME-2024) and hard-science (GPQA-Diamond) accuracy by improving how its_hub scales inference
- metric: accuracy
- run_command: python eval/score.py
- result_path: results/score.json
- target: 0.0
- timeout: 3600

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

- min_growth: 1
- max_new: 3
