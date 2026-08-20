from __future__ import annotations

import asyncio
import hashlib
import os
import random
import re
import time
from enum import Enum
from typing import TYPE_CHECKING

import click
import numpy as np

# NOTE: heavy / optional dependencies (``datasets``, ``math_verify``, ``pandas``,
# ``tqdm``, ``reward_hub``) are imported lazily inside the functions that use
# them. This keeps light-weight, pure-Python helpers such as ``grade_response``
# importable (and unit-testable) in environments that only install the base /
# ``lm`` extras — e.g. the eval harness box without the research/experimental
# extras or a GPU.
from its_hub import OpenAICompatibleLanguageModel, SelfConsistency, StepGeneration
from its_hub.core.algorithms.beam_search import BeamSearch
from its_hub.core.algorithms.particle_gibbs import (
    EntropicParticleFiltering,
    ParticleFiltering,
    _softmax,
)
from its_hub.core.utils import (
    QWEN_SYSTEM_PROMPT,
    SAL_STEP_BY_STEP_SYSTEM_PROMPT,
    extract_content_from_lm_response,
)

if TYPE_CHECKING:  # only for type annotations — kept lazy at runtime
    import pandas as pd
    from reward_hub.base import AggregationMethod


class BenchmarkDataset(Enum):
    MATH500 = "math500"
    AIME_2024 = "aime-2024"
    GPQA_DIAMOND = "gpqa-diamond"


# Option letters used to label the multiple-choice GPQA-Diamond answers.
GPQA_OPTION_LETTERS = ["A", "B", "C", "D"]


def _gpqa_field(row: dict, *names: str):
    """Return the first present, non-null value among ``names`` in a GPQA row.

    The public ``Idavidrein/gpqa`` dataset uses title-cased column names
    (``"Question"``, ``"Correct Answer"``, ...); we also accept lower/underscore
    variants so the loader is robust to minor schema drift.
    """
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    raise KeyError(f"none of {names!r} present in GPQA row")


def _normalize_gpqa_row(row: dict, idx: int) -> dict:
    """Normalize one GPQA-Diamond row into ``{problem, answer, unique_id}``.

    GPQA is multiple-choice: ``problem`` is the question followed by four labeled
    options ``A)/B)/C)/D)`` built from the correct answer plus the three
    incorrect answers. The options are shuffled with a per-item seed derived from
    ``unique_id`` so the correct choice is not positionally constant, and
    ``answer`` is set to the correct option's LETTER.
    """
    question = str(_gpqa_field(row, "Question", "question")).strip()
    correct = str(_gpqa_field(row, "Correct Answer", "correct_answer")).strip()
    incorrects = [
        str(
            _gpqa_field(
                row, f"Incorrect Answer {i}", f"incorrect_answer_{i}"
            )
        ).strip()
        for i in (1, 2, 3)
    ]
    # Prefer the dataset's stable record id; fall back to the row index.
    unique_id = str(
        row.get("Record ID") or row.get("record_id") or idx
    )

    # Deterministic per-item shuffle: seed from the unique_id so ordering is
    # reproducible across runs but the correct option's position varies by item.
    seed = int(hashlib.sha256(unique_id.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    order = list(range(4))  # 0 == correct answer, 1..3 == incorrect answers
    rng.shuffle(order)

    options = [correct, *incorrects]
    shuffled = [options[i] for i in order]
    correct_letter = GPQA_OPTION_LETTERS[order.index(0)]
    labeled = "\n".join(
        f"{GPQA_OPTION_LETTERS[i]}) {opt}" for i, opt in enumerate(shuffled)
    )
    problem = (
        f"{question}\n\n{labeled}\n\n"
        "Please reason step by step, and put the letter of the correct option "
        "(A, B, C, or D) in \\boxed{}."
    )
    return {"problem": problem, "answer": correct_letter, "unique_id": unique_id}


def load_benchmark_dataset(dataset: BenchmarkDataset):
    import datasets

    if dataset == BenchmarkDataset.MATH500:
        ds = datasets.load_dataset("HuggingFaceH4/MATH-500")["test"]
    elif dataset == BenchmarkDataset.AIME_2024:
        ds = datasets.load_dataset("Maxwell-Jia/AIME_2024")["train"]
        old_column_names = ds.column_names
        ds = ds.map(lambda x: {k.lower(): v for k, v in x.items()})
        # use existing id as unique_id
        ds = ds.rename_column("id", "unique_id")
        # convert answer to string type
        ds = ds.cast_column("answer", datasets.Value("string"))
        # remove old columns
        ds = ds.remove_columns(old_column_names)
    elif dataset == BenchmarkDataset.GPQA_DIAMOND:
        # Multiple-choice hard-science benchmark (198 items). Normalize each row
        # into the same {problem, answer, unique_id} schema the run loop consumes;
        # see _normalize_gpqa_row for the option assembly + seeded shuffle.
        raw = datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
        ds = raw.map(
            _normalize_gpqa_row,
            with_indices=True,
            remove_columns=raw.column_names,
        )
    # add unique_id if it doesn't exist
    if "unique_id" not in ds.column_names:
        ds = ds.map(lambda _, idx: {"unique_id": idx}, with_indices=True)
    return ds


class ScalingAlgorithm(Enum):
    SELF_CONSISTENCY = "self-consistency"
    BEAM_SEARCH = "beam-search"
    PARTICLE_FILTERING = "particle-filtering"
    ENTROPIC_PARTICLE_FILTERING = "entropic-particle-filtering"


def _extract_boxed(s: str) -> str:
    # find all occurrences of \boxed{...}
    boxed_matches = re.findall(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}", s)
    # return the last match if any were found
    return boxed_matches[-1] if boxed_matches else ""


def _extract_choice_letter(response: str) -> str | None:
    """Extract the chosen multiple-choice letter (A-D) from a model response.

    Tolerant of the common answer formats: ``\\boxed{C}``, ``answer: (B)``,
    ``The answer is D.``, ``(A)``, or a trailing standalone letter. Returns the
    upper-cased letter, or ``None`` when no A-D choice can be found.
    """
    if not response:
        return None
    text = str(response).strip()

    # 1. \boxed{...} — matches the answer convention used in the GPQA prompt.
    boxed = _extract_boxed(text)
    if boxed:
        m = re.search(r"([A-Da-d])", boxed)
        if m:
            return m.group(1).upper()

    # 2. explicit "answer" phrasing, e.g. "answer: (B)", "final answer is C".
    m = re.search(r"answer\b[^A-Da-d]{0,20}?\(?([A-Da-d])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 3. an option wrapped in parentheses, e.g. "... hence (D).".
    paren = re.findall(r"\(([A-Da-d])\)", text)
    if paren:
        return paren[-1].upper()

    # 4. fall back to the last standalone A-D token.
    standalone = re.findall(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])", text)
    if standalone:
        return standalone[-1].upper()

    return None


def grade_response(benchmark: BenchmarkDataset, gold, response) -> bool:
    """Grade a single model ``response`` against the ``gold`` answer.

    MATH500 / AIME_2024 keep the existing ``math_verify`` path unchanged.
    GPQA_DIAMOND is multiple-choice and is graded by letter/choice matching —
    ``math_verify`` is never called on it (choices are letters, not expressions).
    """
    if benchmark == BenchmarkDataset.GPQA_DIAMOND:
        chosen = _extract_choice_letter(response)
        if chosen is None:
            return False
        # gold is the correct letter, but be tolerant if it arrives wrapped.
        gold_letter = _extract_choice_letter(gold) or str(gold).strip().upper()
        return chosen == gold_letter

    import math_verify

    return bool(
        math_verify.verify(
            math_verify.parse(gold),
            math_verify.parse(response),
        )
    )


def init_algorithm(
    alg: ScalingAlgorithm,
    model_name: str,
    rm_name: str,
    rm_device: str,
    rm_agg_method: AggregationMethod,
    tokens_per_step: int | None = None,
):
    # Imported lazily: reward_hub / vLLM are only needed for the PRM-based
    # (beam-search / particle-filtering) algorithms.
    from its_hub.core.reward_models.local_vllm_prm import (
        LocalVllmProcessRewardModel,
    )

    if alg == ScalingAlgorithm.SELF_CONSISTENCY:
        return SelfConsistency(_extract_boxed)
    elif alg == ScalingAlgorithm.BEAM_SEARCH:
        if tokens_per_step is not None:
            # Use new tokens_per_step approach for easier usage
            sg = StepGeneration(
                max_steps=50, tokens_per_step=tokens_per_step, stop_token="\\boxed"
            )
        else:
            # Use traditional step_token approach
            step_token = "\n\n##" if "llama" in model_name.lower() else "\n\n"
            sg = StepGeneration(
                step_token=step_token, max_steps=50, stop_token="\\boxed"
            )
        prm = LocalVllmProcessRewardModel(
            model_name=rm_name, device=rm_device, aggregation_method=rm_agg_method
        )
        return BeamSearch(sg, prm, beam_width=4)
    elif alg == ScalingAlgorithm.PARTICLE_FILTERING:
        if tokens_per_step is not None:
            # Use new tokens_per_step approach for easier usage
            sg = StepGeneration(
                max_steps=50, tokens_per_step=tokens_per_step, stop_token="\\boxed"
            )
        else:
            # Use traditional step_token approach
            step_token = "\n\n##" if "llama" in model_name.lower() else "\n\n"
            sg = StepGeneration(
                step_token=step_token, max_steps=50, stop_token="\\boxed"
            )
        prm = LocalVllmProcessRewardModel(
            model_name=rm_name, device=rm_device, aggregation_method=rm_agg_method
        )
        return ParticleFiltering(sg, prm)

    elif alg == ScalingAlgorithm.ENTROPIC_PARTICLE_FILTERING:
        if tokens_per_step is not None:
            # Use new tokens_per_step approach for easier usage
            sg = StepGeneration(
                max_steps=50, tokens_per_step=tokens_per_step, stop_token="\\boxed"
            )
        else:
            # Use traditional step_token approach
            step_token = "\n\n##" if "llama" in model_name.lower() else "\n\n"
            sg = StepGeneration(
                step_token=step_token, max_steps=50, stop_token="\\boxed"
            )
        prm = LocalVllmProcessRewardModel(
            model_name=rm_name, device=rm_device, aggregation_method=rm_agg_method
        )
        return EntropicParticleFiltering(sg, prm)


def display_results(df: pd.DataFrame):
    if len(df) == 0:
        print("no results to display")
        return
    # print accuracy per budget using groupby and mean
    accuracy_by_budget = df.groupby("budget")["correct"].agg(["mean", "count"])
    for n, (accuracy, count) in accuracy_by_budget.iterrows():
        print(
            f"budget={n:3d}: accuracy={accuracy:.4f} ({int(accuracy * count):2d}/{int(count):2d})"
        )


@click.command()
@click.option(
    "--benchmark",
    type=click.Choice([e.value for e in BenchmarkDataset]),
    required=True,
    callback=lambda ctx, param, value: BenchmarkDataset(value),
    help="dataset to use for benchmarking",
)
@click.option(
    "--model_name", type=str, required=True, help="model to inference-time scale"
)
@click.option(
    "--is_async", is_flag=True, default=False, help="whether to use async mode"
)
@click.option(
    "--max_completion_tokens",
    type=int,
    default=None,
    help="max completion tokens per generation",
)
@click.option(
    "--temperature",
    type=float,
    default=None,
    help="temperature to use for inference-time scaling",
)
@click.option(
    "--max_concurrency",
    type=int,
    default=8,
    help="max concurrency to use for inference-time scaling",
)
@click.option("--endpoint", type=str, help="endpoint to use for inference-time scaling")
@click.option(
    "--api_key",
    type=str,
    default="NO_API_KEY",
    help="api key to use for inference-time scaling",
)
@click.option(
    "--rm_name",
    type=str,
    default="Qwen/Qwen2.5-Math-PRM-7B",
    help="name of reward model to use",
)
@click.option(
    "--rm_device", type=str, default="cpu", help="device to use for reward model"
)
@click.option(
    "--rm_agg_method",
    type=str,
    default="model",
    # Parsed lazily so importing this module doesn't require reward_hub.
    callback=lambda ctx, param, value: __import__(
        "reward_hub.base", fromlist=["AggregationMethod"]
    ).AggregationMethod(value),
    help="aggregation method to use for reward model (from reward_hub AggregationMethod)",
)
@click.option(
    "--alg",
    type=click.Choice([e.value for e in ScalingAlgorithm]),
    required=True,
    callback=lambda ctx, param, value: ScalingAlgorithm(value),
    help="algorithm to use for inference-time scaling",
)
@click.option(
    "--subset",
    type=str,
    default=None,
    help="subset of dataset to use, in python slice syntax (e.g. ':10', '5:', '5:10')",
)
@click.option(
    "--budgets",
    type=str,
    default="1,2,4,8",
    callback=lambda ctx, param, value: [int(b) for b in value.split(",")],
    help="comma-separated list of budgets to use for inference-time scaling",
)
@click.option(
    "--output_dir", type=str, default="results", help="directory to save results to"
)
@click.option(
    "--shuffle_seed", type=int, default=None, help="random seed to use for shuffling"
)
@click.option(
    "--force_run", is_flag=True, default=False, help="whether to force re-running"
)
@click.option(
    "--does_eval", is_flag=True, default=False, help="whether to evaluate the results"
)
@click.option(
    "--eval_expected_pass_at_one",
    is_flag=True,
    default=False,
    help="whether to evaluate expected pass at one",
)
@click.option(
    "--display_only",
    is_flag=True,
    default=False,
    help="whether to show only the results",
)
@click.option(
    "--tokens_per_step",
    type=int,
    default=None,
    help="use tokens_per_step instead of step_token for StepGeneration (easier for PF/BS algorithms)",
)
def main(
    benchmark: BenchmarkDataset,
    model_name: str,
    is_async: bool,
    max_completion_tokens: int | None,
    temperature: float,
    max_concurrency: int,
    endpoint: str,
    api_key: str,
    rm_name: str,
    rm_device: str,
    rm_agg_method: AggregationMethod,
    alg: ScalingAlgorithm,
    subset: str,
    budgets: list,
    output_dir: str,
    shuffle_seed: int,
    force_run: bool,
    does_eval: bool,
    eval_expected_pass_at_one: bool,
    display_only: bool,
    tokens_per_step: int,
):
    import pandas as pd
    from tqdm import tqdm

    # print all arguments using click context
    ctx = click.get_current_context()
    print("running with arguments:")
    for param_name, param_value in ctx.params.items():
        print(f"  {param_name}: {param_value}")

    if eval_expected_pass_at_one:
        assert alg in [
            ScalingAlgorithm.PARTICLE_FILTERING,
            ScalingAlgorithm.ENTROPIC_PARTICLE_FILTERING,
        ], "expected pass at one is only supported for particle filtering algorithms"

    print("loading existing results...")
    model_name_dashed = model_name.replace("/", "-")
    if (
        alg == ScalingAlgorithm.BEAM_SEARCH
        or alg == ScalingAlgorithm.PARTICLE_FILTERING
        or alg == ScalingAlgorithm.ENTROPIC_PARTICLE_FILTERING
    ):
        rm_name_dashed = rm_name.replace("/", "-")
        alg_str = f"{alg.value}-{rm_name_dashed}-{rm_agg_method.value}"
        # Add tokens_per_step to filename if specified
        if tokens_per_step is not None:
            alg_str += f"-tokens{tokens_per_step}"
    else:
        alg_str = alg.value
    output_file = os.path.join(
        output_dir, f"{model_name_dashed}-{alg_str}-{benchmark.value}.jsonl"
    )
    if os.path.exists(output_file):
        df_existing = pd.read_json(output_file, orient="records", lines=True)
        print(f"loaded {len(df_existing)} existing results from {output_file}")
    else:
        df_existing = pd.DataFrame()

    if display_only:
        display_results(df_existing)
        return

    print("loading benchmark dataset...")
    dataset = load_benchmark_dataset(benchmark)

    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)

    # apply subset if specified
    if subset is not None:
        try:
            # parse the slice syntax
            if ":" in subset:
                parts = subset.split(":")
                if len(parts) == 2:
                    start = int(parts[0]) if parts[0] else None
                    end = int(parts[1]) if parts[1] else None
                    dataset = dataset.select(
                        range(
                            start if start is not None else 0,
                            end if end is not None else len(dataset),
                        )
                    )
            else:
                # single index
                dataset = dataset.select([int(subset)])
            print(f"using subset of dataset: {len(dataset)} examples")
        except ValueError:
            print(f"invalid subset format: {subset}, using full dataset")

    print("creating language model...")
    if endpoint is not None:
        lm = OpenAICompatibleLanguageModel(
            endpoint=endpoint,
            api_key=api_key,
            model_name=model_name,
            system_prompt=QWEN_SYSTEM_PROMPT
            if "qwen" in model_name.lower()
            else SAL_STEP_BY_STEP_SYSTEM_PROMPT,
            is_async=is_async,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            max_concurrency=max_concurrency,
        )

    print("initializing algorithm...")
    scaling_alg = init_algorithm(
        alg,
        model_name,
        rm_name,
        rm_device,
        rm_agg_method,
        tokens_per_step,
    )

    # ensure output directory exists
    if not os.path.exists(output_dir):
        print(f"creating output directory: {output_dir}")
        os.makedirs(output_dir)

    print(f"running inference-time scaling for {budgets=}...")
    rows = []
    budget_timings = {}
    try:
        for n in tqdm(budgets):
            budget_start_time = time.time()
            for x in dataset:
                y_full = None
                y = None
                if not force_run and len(df_existing) > 0:
                    # only skip if both the unique_id and budget matches
                    match = (df_existing["unique_id"] == x["unique_id"]) & (
                        df_existing["budget"] == n
                    )
                    if match.any():
                        assert match.sum() == 1, (
                            f"expected exactly one match, got {match.sum()}"
                        )
                        if eval_expected_pass_at_one:
                            y_full = {
                                "responses": df_existing.loc[match, "responses"].values[
                                    0
                                ],
                                "log_probs": df_existing.loc[match, "log_probs"].values[
                                    0
                                ],
                            }
                        else:
                            y = df_existing.loc[match, "response"].values[0]
                if y_full is None if eval_expected_pass_at_one else y is None:
                    try:
                        if eval_expected_pass_at_one:
                            y_full = scaling_alg.infer(
                                lm, x["problem"], n, return_response_only=False
                            )
                            y_full = {
                                "responses": y_full.responses_lst[-1],
                                "log_probs": y_full.log_weights_lst[-1],
                            }
                        else:
                            y = scaling_alg.infer(lm, x["problem"], n)
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        print(f"error scaling example {x['unique_id']}: {e}")
                        continue
                if eval_expected_pass_at_one:
                    row = {
                        "unique_id": x["unique_id"],
                        "budget": n,
                        "responses": y_full["responses"],
                        "log_probs": y_full["log_probs"],
                        "correct": None,
                    }
                else:
                    row = {
                        "unique_id": x["unique_id"],
                        "budget": n,
                        "response": y,
                        "correct": None,
                    }
                if does_eval:
                    if eval_expected_pass_at_one:
                        c = [
                            grade_response(
                                benchmark,
                                x["answer"],
                                extract_content_from_lm_response(y) if isinstance(y, dict) else y,
                            )
                            for y in row["responses"]
                        ]
                        p = _softmax(row["log_probs"])
                        row["correct"] = np.dot(p, c)
                    else:
                        response_content = extract_content_from_lm_response(row["response"]) if isinstance(row["response"], dict) else row["response"]
                        row["correct"] = grade_response(
                            benchmark, x["answer"], response_content
                        )
                rows.append(row)

            # Record timing for this budget
            budget_end_time = time.time()
            budget_elapsed_time = budget_end_time - budget_start_time
            budget_timings[n] = budget_elapsed_time
            print(f"\nBudget {n} completed in {budget_elapsed_time:.2f} seconds ({budget_elapsed_time/60:.2f} minutes)")

    except KeyboardInterrupt:
        print("\nkeyboard interrupt detected, saving partial results")

    # Display timing summary
    if budget_timings:
        print("\n=== Timing Summary ===")
        total_time = sum(budget_timings.values())
        for budget, elapsed_time in budget_timings.items():
            print(f"Budget {budget:3d}: {elapsed_time:8.2f}s ({elapsed_time/60:6.2f} min)")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print("=" * 40)

    # save results to jsonl file using pandas
    print(f"\nsaving results to {output_file}...")
    df = pd.concat([df_existing, pd.DataFrame(rows)])
    # deduplicate rows with the same unique_id and budget, keeping the updated correctness
    df = df.drop_duplicates(subset=["unique_id", "budget"], keep="last")

    display_results(df)

    df.to_json(output_file, orient="records", lines=True)

    # Close lm for resource cleanup
    asyncio.run(lm.close())


if __name__ == "__main__":
    main()
