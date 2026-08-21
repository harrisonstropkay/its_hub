"""H1 live end-to-end: record -> replay(dead endpoint) -> modify, on real vLLM.

Not a committed test (requires a live GPU endpoint). Run manually:
    ITS_CACHE_MODE unset; python scripts/_h1_live_e2e.py
"""

import asyncio
import os
import shutil
from collections import Counter

from its_hub.core.algorithms.self_consistency import SelfConsistency
from its_hub.core.lms.openai_lm import CacheMissError, OpenAICompatibleLanguageModel

LIVE = "http://localhost:8000/v1"
DEAD = "http://localhost:1/v1"  # nothing listens here
MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"
CACHE_DIR = ".factory/cache/lm_responses"
PROMPT = "What is 12 * 12? Put the final answer in \\boxed{}."
BUDGET = 8


def _lm(endpoint):
    return OpenAICompatibleLanguageModel(
        endpoint=endpoint,
        api_key="NO_API_KEY",
        model_name=MODEL,
        temperature=0.8,
        max_completion_tokens=512,
        max_tries=1,
    )


async def main():
    # Clean slate for this key space so record captures fresh draws.
    if os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)

    # ---- RECORD (live endpoint) ----
    os.environ["ITS_CACHE_MODE"] = "record"
    lm = _lm(LIVE)
    sc = SelfConsistency()
    rec = await sc.ainfer(lm, PROMPT, budget=BUDGET, return_response_only=False)
    await lm.close()
    rec_counts = dict(rec.response_counts)
    print("RECORD  the_one:", repr(rec.the_one["content"][:80]))
    print("RECORD  counts :", rec_counts)

    # ---- REPLAY against a DEAD endpoint (proves no regeneration) ----
    os.environ["ITS_CACHE_MODE"] = "replay"
    import its_hub.core.lms.openai_lm as mod

    mod._cache = None  # reset replay cursors
    lm_dead = _lm(DEAD)
    sc2 = SelfConsistency()
    rep = await sc2.ainfer(lm_dead, PROMPT, budget=BUDGET, return_response_only=False)
    await lm_dead.close()
    rep_counts = dict(rep.response_counts)
    print("REPLAY  the_one:", repr(rep.the_one["content"][:80]))
    print("REPLAY  counts :", rep_counts)

    # ---- MODIFY aggregation under replay (dead endpoint) ----
    mod._cache = None
    lm_dead2 = _lm(DEAD)
    sc3 = SelfConsistency(consistency_space_projection_func=lambda _c: "ALL")
    modres = await sc3.ainfer(lm_dead2, PROMPT, budget=BUDGET, return_response_only=False)
    await lm_dead2.close()
    mod_counts = dict(modres.response_counts)
    print("MODIFY  counts :", mod_counts)

    # ---- Cache-miss proof (replay a never-recorded prompt on dead endpoint) ----
    mod._cache = None
    lm_dead3 = _lm(DEAD)
    miss = False
    try:
        await sc3.ainfer(lm_dead3, "unrecorded prompt", budget=1)
    except CacheMissError:
        miss = True
    except RuntimeError as e:
        # the orchestrator wraps the underlying CacheMissError in a RuntimeError
        miss = "CacheMissError" in str(e)
    await lm_dead3.close()

    # ---- Assertions ----
    assert rep.the_one["content"] == rec.the_one["content"], "the_one changed!"
    assert Counter(rec_counts) == Counter(rep_counts), "vote counts changed!"
    assert mod_counts == {"ALL": BUDGET}, f"modify did not collapse: {mod_counts}"
    assert mod_counts != rep_counts, "modify did not change the result!"
    assert miss, "replay of an unrecorded prompt did NOT raise CacheMissError"
    print("\nALL LIVE E2E ASSERTIONS PASSED")
    print("- replay reproduced the_one + counts byte-for-byte from cache")
    print("- replay ran against a DEAD endpoint -> zero regeneration")
    print("- modifying voting under replay changed the result (fed cached samples)")
    print("- unrecorded prompt under replay raised CacheMissError")


if __name__ == "__main__":
    asyncio.run(main())
