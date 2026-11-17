"""Merge several takes of one batch into one manifest: per sentence the slowest take that is read well.

uv run python scripts/slow_take.py v5s v5m v5n [v5r] [--prefer សែសិប=v5n]

Writes outputs/distill/<out>/manifest.jsonl. Ear rejects and the stray-noise check are applied again, so a clip rejected after
its batch was checked stays out. --prefer WORD=BATCH takes that batch for sentences containing WORD.
"""

import argparse
import statistics

import soundfile as sf

from samleng.audio import stray_tail
from samleng.data import ear_rejected, pick_take
from samleng.paths import OUTPUT_DIR, ROOT
from samleng.runners import read_jsonl, write_jsonl

DISTILL = OUTPUT_DIR / "distill"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("batches", nargs="+")
    ap.add_argument("--prefer", action="append", default=[], metavar="WORD=BATCH")
    args = ap.parse_args()

    ear = ear_rejected(
        (ROOT / "texts" / "teacher_ear_rejects.txt").read_text("utf-8").splitlines()
    )
    prefer = dict(p.split("=", 1) for p in args.prefer)
    takes: dict[str, dict[str, dict]] = {}
    for batch in args.batches:
        for r in read_jsonl(DISTILL / batch / "manifest.jsonl"):
            if r["id"] in ear or f"{batch}/{r['id']}" in ear:
                r["reject"] = "ear"
            elif "reject" not in r and stray_tail(*sf.read(r["audio"])):
                r["reject"] = "noise"
            takes.setdefault(r["id"], {})[batch] = r
    rows = [
        pick_take(
            t,
            next(
                (b for w, b in prefer.items() if w in next(iter(t.values()))["text"]),
                None,
            ),
        )
        for t in takes.values()
    ]
    (DISTILL / args.out).mkdir(parents=True, exist_ok=True)
    write_jsonl(DISTILL / args.out / "manifest.jsonl", rows)
    kept = [r for r in rows if "reject" not in r]
    hours = sum(r["seconds"] for r in kept) / 3600
    print(
        f"{args.out}: kept {len(kept)} of {len(rows)}, {hours:.2f} h, "
        f"median {statistics.median(r['cps'] for r in kept):.2f} chars/s"
    )


if __name__ == "__main__":
    main()
