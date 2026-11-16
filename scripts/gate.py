"""Exit 0 if a scored run's best all-Khmer error is at or below a limit, else 3 (4: no score yet).

uv run python scripts/gate.py distill_v2_calm --max 8.0 || { print "skipped"; exit 0 }
"""

import argparse
import sys

from samleng.eval.gate import best_all
from samleng.paths import OUTPUT_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run name under outputs/train/eval")
    ap.add_argument("--max", type=float, required=True, dest="limit")
    args = ap.parse_args()
    best = best_all(OUTPUT_DIR / "train" / "eval" / args.run)
    if best is None:
        print(f"{args.run}: no score yet")
        sys.exit(4)
    print(f"{args.run}: best ALL {best[0]} ({best[1]}), limit {args.limit}")
    sys.exit(0 if best[0] <= args.limit else 3)


if __name__ == "__main__":
    main()
