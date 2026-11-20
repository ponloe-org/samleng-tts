"""Keep the current best model (the champion) and let scored runs challenge it.

uv run python scripts/ladder.py crown RUN --steps 6000 8000 10000 --checkpoint PATH --label "full calm run"
uv run python scripts/ladder.py challenge RUN --steps ... --checkpoint PATH --label "init from run A"
uv run python scripts/ladder.py checkpoint      # prints the champion's checkpoint (to start the next run from)
uv run python scripts/ladder.py best RUN --steps ...   # prints the step with the lowest 3-seed error

Both runs must already be scored on 3 seeds per step (pick_checkpoint.py, Latin and Khmer-script
input). The decision uses Latin input; a win needs >= 0.5 points lower median ALL and p < 0.05.
State: outputs/train/champion.json; one report per challenger in outputs/train/eval/ladder/.
"""

import argparse
import json
import sys
import time

import pandas as pd

from samleng.eval.ladder import averaged, best_step, result_files, verdict
from samleng.eval.report import category_medians, paired
from samleng.paths import OUTPUT_DIR

EVAL = OUTPUT_DIR / "train" / "eval"
STATE = OUTPUT_DIR / "train" / "champion.json"
REPORTS = EVAL / "ladder"


def entry(args: argparse.Namespace) -> dict:
    return {
        "run": args.run,
        "steps": args.steps,
        "checkpoint": args.checkpoint,
        "label": args.label,
        "since": time.strftime("%Y-%m-%d %H:%M"),
    }


def challenge(args: argparse.Namespace) -> None:
    champ = json.loads(STATE.read_text())
    sections, decision = [], None
    for translit in (False, True):
        a = averaged(result_files(EVAL, champ["run"], champ["steps"], translit))
        b = averaged(result_files(EVAL, args.run, args.steps, translit))
        v = verdict(a, b)
        decision = decision or v
        rows = [
            {"model": f"champion: {champ['label']}"} | category_medians(a),
            {"model": f"challenger: {args.label}"} | category_medians(b),
        ]
        pairs = [
            {"category": c or "ALL"} | paired(b, a, c)
            for c in (None, "pure", "numbers", "names", "mixed_unseen")
        ]
        sections.append(
            f"## {'Khmer-script' if translit else 'Latin'} input\n\n"
            + pd.DataFrame(rows).to_markdown(index=False)
            + "\n\nPairs (better = challenger lower):\n\n"
            + pd.DataFrame(pairs)
            .assign(p=lambda d: d["p"].map("{:.2g}".format))
            .to_markdown(index=False)
        )
    won = decision["win"]
    head = (
        f"# {args.label} ({args.run}) vs champion {champ['label']} ({champ['run']})\n\n"
        f"Decision (Latin input): {'NEW CHAMPION' if won else 'champion stays'}: "
        f"ALL {decision['challenger']} vs {decision['champion']}, "
        f"{decision['better']} sentences better / {decision['worse']} worse, p {decision['p']:.2g}.\n\n"
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{args.run}.md").write_text(head + "\n\n".join(sections) + "\n")
    with open(REPORTS / "log.md", "a") as f:
        f.write(
            f"- {time.strftime('%m-%d %H:%M')} {args.label}: ALL {decision['challenger']} vs "
            f"{decision['champion']} (p {decision['p']:.2g}) -> {'NEW CHAMPION' if won else 'champion stays'}\n"
        )
    if won:
        STATE.write_text(
            json.dumps(
                entry(args) | {"history": [*champ.get("history", []), champ]},
                ensure_ascii=False,
                indent=2,
            )
        )
    print(head.strip())
    try:
        from samleng.notify import send

        send(
            f"Ladder: {args.label}",
            f"{'NEW CHAMPION' if won else 'champion stays'}: ALL {decision['challenger']} vs {decision['champion']}",
        )
    except Exception:  # a phone alert must never fail the job
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["crown", "challenge", "checkpoint", "best"])
    ap.add_argument("run", nargs="?")
    ap.add_argument("--steps", type=int, nargs="+")
    ap.add_argument("--checkpoint", help="checkpoint to start later runs from")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    if args.action == "checkpoint":
        print(json.loads(STATE.read_text())["checkpoint"])
    elif args.action == "best":
        print(best_step(EVAL, args.run, args.steps))
    elif args.action == "crown":
        STATE.write_text(json.dumps(entry(args), ensure_ascii=False, indent=2))
        print(f"champion: {args.label} ({args.run})")
    else:
        if not (args.run and args.steps):
            sys.exit("challenge needs RUN and --steps")
        challenge(args)


if __name__ == "__main__":
    main()
