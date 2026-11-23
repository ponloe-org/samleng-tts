"""Self-deciding runs: a challenger replaces the champion only when it is clearly better.

Both are single models (one checkpoint or one weight average). The challenger's step is picked
on seeds 0-2; the decision uses fresh seeds 3-5 for both, so picking the best step can't flatter it.
Score = mean per-sentence error capped at CAP (an ASR loop can't swing it), pooled over inputs:
mixed sentences average Latin and Khmer-script input, the rest are identical under both.
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from samleng.eval.cer import mixed_cer
from samleng.eval.report import category_medians, paired, seed_average

CAP = 0.5
MARGIN = 0.5  # points of pooled capped mean
ALPHA = 0.05
PICK_SEEDS = (0, 1, 2)
FRESH_SEEDS = (3, 4, 5)
GUARD_CATEGORIES = ("pure", "numbers", "names", "long")
GUARD_POINTS = 1.0
GUARD_MIXED_UNSEEN = 2.0
PACE_TOLERANCE = 0.03


def result_files(
    eval_root: Path,
    run: str,
    steps: list[int],
    translit: bool,
    seeds: Iterable[int] = PICK_SEEDS,
) -> list[Path]:
    """results.csv of every checkpoint x seed, in the folder layout of pick_checkpoint.py."""
    tr = "_translit" if translit else ""
    return [
        eval_root / (f"{run}{tr}" + (f"_seed{k}" if k else "")) / str(s) / "results.csv"
        for s in steps
        for k in seeds
    ]


def capped_cer(ref: str, hyp: str) -> float:
    return min(mixed_cer(ref, hyp), CAP)


def averaged(files: list[Path], metric=mixed_cer) -> pd.DataFrame:
    return seed_average([pd.read_csv(f) for f in files], metric)


def pooled(latin: pd.DataFrame, khmer: pd.DataFrame) -> pd.DataFrame:
    """Per-sentence error: mixed sentences averaged over both inputs, others from Latin input."""
    k = khmer.set_index("id")["cer"]
    mixed = latin["category"].str.startswith("mixed")
    cer = np.where(
        mixed, (latin["cer"].to_numpy() + k[latin["id"]].to_numpy()) / 2, latin["cer"]
    )
    return latin.assign(cer=cer)


def score(eval_root: Path, run: str, step: int, seeds: Iterable[int]) -> dict:
    seeds = tuple(seeds)
    files = {t: result_files(eval_root, run, [step], t, seeds) for t in (False, True)}
    seconds = pd.read_csv(files[False][0])["seconds"].sum()
    return {
        "pooled": pooled(
            averaged(files[False], capped_cer), averaged(files[True], capped_cer)
        ),
        "latin": averaged(files[False]),
        "khmer": averaged(files[True]),
        "seconds": float(seconds),
    }


def bootstrap_ci(diff: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _mean_points(
    a: pd.DataFrame, b: pd.DataFrame, category: str | None = None
) -> float:
    """b minus a, mean per-sentence error in points, on a's sentences."""
    if category:
        a, b = a[a["category"] == category], b[b["category"] == category]
    x, y = a.set_index("id")["cer"], b.set_index("id")["cer"]
    return round(100 * float((y[x.index] - x).mean()), 2)


def verdict(champion: dict, challenger: dict) -> dict:
    """Win = >= MARGIN lower pooled capped mean, CI below 0, more better than worse, no guard hit."""
    a, b = champion["pooled"], challenger["pooled"]
    x = a.set_index("id")["cer"]
    diff = (b.set_index("id")["cer"][x.index] - x).to_numpy()
    delta = round(100 * float(diff.mean()), 2)
    lo, hi = bootstrap_ci(100 * diff)
    p = paired(b, a)
    blocks = []
    for c in GUARD_CATEGORIES:
        d = _mean_points(a, b, c)
        if d > GUARD_POINTS and paired(b, a, c)["p"] < ALPHA:
            blocks.append(f"{c} worse by {d} points")
    d = _mean_points(champion["latin"], challenger["latin"], "mixed_unseen")
    if d > GUARD_MIXED_UNSEEN:
        blocks.append(f"Latin mixed_unseen worse by {d} points")
    pace = challenger["seconds"] / champion["seconds"]
    if (
        pace > 1 + PACE_TOLERANCE
    ):  # slower speech is easier for the judge; faster is no unfair help
        blocks.append(
            f"speaks slower ({pace:.3f}x the champion's time): re-score at a matched length scale"
        )
    win = delta <= -MARGIN and hi < 0 and p["better"] > p["worse"] and not blocks
    return {
        "champion": round(100 * float(a["cer"].mean()), 2),
        "challenger": round(100 * float(b["cer"].mean()), 2),
        "delta": delta,
        "ci": (round(lo, 2), round(hi, 2)),
        "pace": round(pace, 3),
        "blocks": blocks,
        "win": win,
    } | p


def best_step(eval_root: Path, run: str, steps: list[int]) -> int:
    """The checkpoint with the lowest pooled capped mean on the pick seeds."""
    return min(
        steps,
        key=lambda s: score(eval_root, run, s, PICK_SEEDS)["pooled"]["cer"].mean(),
    )


def medians(s: dict) -> dict[str, float]:
    """Reported numbers (the goal is median ALL): Latin-input category medians + Khmer-script ALL."""
    return category_medians(s["latin"]) | {
        "ALL Khmer-script": category_medians(s["khmer"])["ALL"]
    }
