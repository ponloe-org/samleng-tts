"""Did training on a word list fix those words on the bench, beyond what the held-back control words did?

    uv run python scripts/word_check.py texts/v6_control_words.txt --step 6000 \
        --new distill_v4_v6 --old distill_v3_initA [--seeds 0 1 2 3 4 5]

Pools seeds and both inputs (Latin and Khmer script) per (sentence, word) pair, then
counts pairs the new run reads more often right than the old run, and the reverse.
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

from samleng.eval.errors import word_read

EVAL = Path("outputs/train/eval")


def groups(path: Path) -> dict[str, list[str]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            name, *words = line.split()
            out[name] = words
    return out


def results(run: str, step: int | None, seeds: list[int]) -> list[pd.DataFrame]:
    found = []
    for seed in seeds:
        for t in ("", "_translit"):
            d = EVAL / f"{run}{t}{f'_seed{seed}' if seed else ''}"
            f = d / "results.csv" if step is None else d / str(step) / "results.csv"
            if f.exists():
                found.append(pd.read_csv(f, keep_default_na=False))
    return found


def hits(frames: list[pd.DataFrame], words: list[str]) -> dict[tuple[str, str], float]:
    """(sentence id, word) -> share of runs that read the word right."""
    seen: dict[tuple[str, str], list[bool]] = {}
    for df in frames:
        for sid, ref, hyp in zip(df.id, df.ref, df.hyp, strict=True):
            for w in words:
                ok = word_read(ref, hyp, w)
                if ok is not None:
                    seen.setdefault((sid, w), []).append(ok)
    return {k: sum(v) / len(v) for k, v in seen.items()}


def wrong(h: dict[tuple[str, str], float], keys: list[tuple[str, str]]) -> float:
    keys = [k for k in keys if k in h]
    return 100 * sum(1 - h[k] for k in keys) / len(keys) if keys else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("words", type=Path)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--old-step", type=int)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    new = results(args.new, args.step, args.seeds)
    old = results(args.old, args.old_step or args.step, args.seeds)
    base = results("base", None, [0])
    lines = [
        f"# Word check: {args.new} step {args.step} vs {args.old} step {args.old_step or args.step}",
        "",
        f"Runs pooled: new {len(new)}, old {len(old)} (seeds {args.seeds}, both inputs); base {len(base)}.",
        "",
        "| group | words on bench | pairs | base wrong % | old wrong % | new wrong % | new better | new worse | p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, words in groups(args.words).items():
        n, o, b = hits(new, words), hits(old, words), hits(base, words)
        keys = sorted(n.keys() & o.keys())
        if not keys:
            continue
        better = sum(n[k] > o[k] for k in keys)
        worse = sum(n[k] < o[k] for k in keys)
        p = binomtest(better, better + worse).pvalue if better + worse else 1.0
        lines.append(
            f"| {name} | {len({w for _, w in keys})} | {len(keys)} | {wrong(b, keys):.0f}"
            f" | {wrong(o, keys):.0f} | {wrong(n, keys):.0f} | {better} | {worse} | {p:.2g} |"
        )
    lines += [
        "",
        "A pair is one bench sentence and one listed word in it; wrong % = runs where the ASR did not hear it.",
    ]
    text = "\n".join(lines) + "\n"
    out = EVAL / args.new / f"words_{args.step}.md"
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
