"""Keep the current best model (the champion) and let single scored models challenge it.

uv run python scripts/ladder.py crown RUN --step 6000 --run-dir DIR --checkpoint PATH --label "init from run A"
uv run python scripts/ladder.py best RUN --steps 4000 6000 8000   # step to challenge with (seeds 0-2)
uv run python scripts/ladder.py challenge RUN --step 6000 --run-dir DIR --checkpoint PATH --label "..."
uv run python scripts/ladder.py checkpoint      # the champion's checkpoint (to start the next run from)
uv run python scripts/ladder.py champion        # "RUN STEP RUN_DIR", for job scripts

`best` needs seeds 0-2 of every step; `challenge` needs seeds 3-5 of both models (pick_checkpoint.py
--seeds 6), Latin and Khmer-script input. Rule: samleng.eval.ladder. State: outputs/train/champion.json;
one report per challenger in outputs/train/eval/ladder/.
"""

import argparse
import json
import time

import pandas as pd

from samleng.eval.ladder import FRESH_SEEDS, best_step, medians, score, verdict
from samleng.paths import OUTPUT_DIR

EVAL = OUTPUT_DIR / "train" / "eval"
STATE = OUTPUT_DIR / "train" / "champion.json"
REPORTS = EVAL / "ladder"


def load() -> dict:
    champ = json.loads(STATE.read_text())
    champ.setdefault("step", champ.get("steps", [0])[-1])
    if "run_dir" not in champ:  # state written before single-model challenges
        hit = next(
            (OUTPUT_DIR / "train" / champ["run"]).glob(
                f"*/checkpoint_{champ['step']}.pth"
            )
        )
        champ["run_dir"] = str(hit.parent)
    return champ


def entry(args: argparse.Namespace) -> dict:
    return {
        "run": args.run,
        "step": args.step,
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "label": args.label,
        "since": time.strftime("%Y-%m-%d %H:%M"),
    }


def challenge(args: argparse.Namespace) -> None:
    champ = load()
    a = score(EVAL, champ["run"], champ["step"], FRESH_SEEDS)
    b = score(EVAL, args.run, args.step, FRESH_SEEDS)
    v = verdict(a, b)
    won = v["win"]
    rows = [
        {"model": f"champion: {champ['label']}", "capped mean": v["champion"]}
        | medians(a),
        {"model": f"challenger: {args.label}", "capped mean": v["challenger"]}
        | medians(b),
    ]
    head = (
        f"# {args.label} ({args.run} {args.step}) vs champion {champ['label']} "
        f"({champ['run']} {champ['step']})\n\n"
        f"Decision (fresh seeds {'/'.join(map(str, FRESH_SEEDS))}, pooled capped mean): "
        f"{'NEW CHAMPION' if won else 'champion stays'}: {v['challenger']} vs {v['champion']} "
        f"(change {v['delta']}, 95% CI {v['ci'][0]} to {v['ci'][1]}), "
        f"{v['better']} sentences better / {v['worse']} worse, p {v['p']:.2g}, pace {v['pace']}x."
        + (f" Blocked: {'; '.join(v['blocks'])}." if v["blocks"] else "")
        + "\n\n"
    )
    table = pd.DataFrame(rows).to_markdown(index=False)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{args.run}_{args.step}.md").write_text(
        head
        + "Medians (%) are Latin input unless named; the goal is median ALL.\n\n"
        + table
        + "\n"
    )
    with open(REPORTS / "log.md", "a") as f:
        f.write(
            f"- {time.strftime('%m-%d %H:%M')} {args.label}: capped mean {v['challenger']} vs "
            f"{v['champion']} (CI {v['ci'][0]} to {v['ci'][1]}), median ALL "
            f"{medians(b)['ALL']} vs {medians(a)['ALL']} -> {'NEW CHAMPION' if won else 'champion stays'}\n"
        )
    if won:
        history = [*champ.pop("history", []), champ]
        STATE.write_text(
            json.dumps(entry(args) | {"history": history}, ensure_ascii=False, indent=2)
        )
    print(head.strip())
    try:
        from samleng.notify import send

        send(
            f"Ladder: {args.label}",
            f"{'NEW CHAMPION' if won else 'champion stays'}: {v['challenger']} vs {v['champion']}",
        )
    except Exception:  # a phone alert must never fail the job
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "action", choices=["crown", "challenge", "checkpoint", "best", "champion"]
    )
    ap.add_argument("run", nargs="?")
    ap.add_argument("--step", type=int)
    ap.add_argument("--steps", type=int, nargs="+")
    ap.add_argument(
        "--run-dir", help="folder holding checkpoint_<step>.pth and config.json"
    )
    ap.add_argument("--checkpoint", help="checkpoint to start later runs from")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    if args.action == "checkpoint":
        print(load()["checkpoint"])
    elif args.action == "champion":
        c = load()
        print(c["run"], c["step"], c["run_dir"])
    elif args.action == "best":
        print(best_step(EVAL, args.run, args.steps))
    else:
        if not (args.run and args.step and args.run_dir and args.checkpoint):
            ap.error(f"{args.action} needs RUN, --step, --run-dir and --checkpoint")
        if args.action == "crown":
            old = load() if STATE.exists() else None
            history = [*old.pop("history", []), old] if old else []
            STATE.write_text(
                json.dumps(
                    entry(args) | {"history": history}, ensure_ascii=False, indent=2
                )
            )
            print(f"champion: {args.label} ({args.run} {args.step})")
        else:
            challenge(args)


if __name__ == "__main__":
    main()
