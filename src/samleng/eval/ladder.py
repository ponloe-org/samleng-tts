"""Self-deciding runs: a challenger replaces the champion only when it is clearly better.

Both are scored on 3 seeds per checkpoint at the same settings; per-sentence errors are averaged
over all their checkpoints and seeds, then compared on the same sentences.
"""

from pathlib import Path

import pandas as pd

from samleng.eval.cer import mixed_cer
from samleng.eval.report import category_medians, paired, seed_average

MARGIN = 0.5  # points of median all-Khmer error
ALPHA = 0.05


def result_files(
    eval_root: Path, run: str, steps: list[int], translit: bool, seeds: int = 3
) -> list[Path]:
    """results.csv of every checkpoint x seed, in the folder layout of pick_checkpoint.py."""
    tr = "_translit" if translit else ""
    return [
        eval_root / (f"{run}{tr}" + (f"_seed{k}" if k else "")) / str(s) / "results.csv"
        for s in steps
        for k in range(seeds)
    ]


def averaged(files: list[Path]) -> pd.DataFrame:
    return seed_average([pd.read_csv(f) for f in files], mixed_cer)


def verdict(champion: pd.DataFrame, challenger: pd.DataFrame) -> dict:
    """Win = at least MARGIN lower median ALL, significant, and lower on average (the test is two-sided)."""
    a, b = category_medians(champion)["ALL"], category_medians(challenger)["ALL"]
    p = paired(challenger, champion)
    lower = challenger["cer"].mean() < champion["cer"].mean()
    win = (
        round(a - b, 1) >= MARGIN
        and p["p"] < ALPHA
        and p["better"] > p["worse"]
        and lower
    )
    return {"champion": a, "challenger": b} | p | {"win": win}


def best_step(eval_root: Path, run: str, steps: list[int], seeds: int = 3) -> int:
    """The checkpoint with the lowest 3-seed median ALL (Latin input), to start the next run from."""
    return min(
        steps,
        key=lambda s: category_medians(
            averaged(result_files(eval_root, run, [s], False, seeds))
        )["ALL"],
    )
