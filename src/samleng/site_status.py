"""Live training status for the project page: where the run is, when it ends, how it scores."""

import json
import re
import statistics
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from samleng.eval.cer import mixed_cer

ANSI = re.compile(r"\x1b\[[0-9;]*m")
EPOCH = re.compile(r"EPOCH: (\d+)/(\d+)")
STEP = re.compile(r"TIME: (\S+ \S+) -- STEP: (\d+)/(\d+) -- GLOBAL_STEP: (\d+)")
STEP_TIME = re.compile(r"\| > step_time: ([\d.]+)")
LOSS = re.compile(r"GLOBAL_STEP: (\d+)[^\n]*\n(?:[^\n]*\n)*?\s*\| > loss_mel: ([\d.]+)")
LOSS_POINTS = 80
CATEGORIES = ("pure", "numbers", "names", "long", "mixed_unseen")
HEADER = re.compile(r"^# site-(\w+): *(.+?) *$", re.M)


def job_info(script: str) -> dict:
    """What a queue job tells the page, from `# site-<key>: value` lines in its script.

    Keys: kind (training, testing, recording), en, km (one plain sentence each),
    run (whose scores to show), progress (`<glob> <total>`: files done out of total).
    """
    return dict(HEADER.findall(script))


KM_DIGITS = str.maketrans("0123456789", "០១២៣៤៥៦៧៨៩")
NEW_BEST = {
    "en": (
        "New best model",
        "A new model beat our best one on a fresh test: it read {better} sentences better"
        " and {worse} worse. Error rate on Khmer sentences: {before}% → {after}%.",
    ),
    "km": (
        "ម៉ូដែលល្អបំផុតថ្មី",
        "ម៉ូដែលថ្មីមួយបានឈ្នះម៉ូដែលល្អបំផុតរបស់យើង នៅក្នុងការតេស្តថ្មី៖ វាអានបានល្អជាងមុន {better} ប្រយោគ"
        " និងអាក្រក់ជាងមុន {worse} ប្រយោគ។ អត្រាកំហុសលើប្រយោគខ្មែរ ពី {before}% មក {after}%។",
    ),
}


def champions(state: dict) -> list[dict]:
    """The champion and every earlier one (older state files nest their history)."""
    out, todo = [], [state]
    while todo:
        c = dict(todo.pop())
        todo += c.pop("history", [])
        out.append(c)
    return out


def champion_summary(state: dict) -> dict | None:
    """The current champion's numbers for the page's results row, if its win recorded them."""
    s = state.get("scores")
    if not s:
        return None
    return {
        "date": state["since"][:10],
        "all": s["ALL"],
        "mixed_unseen": s.get("mixed_unseen"),
    }


def new_best_entries(state: dict, clips: Path | None = None) -> list[dict]:
    """One lab-log entry per crowned model whose win recorded its scores.

    Clips are listed when clips/<run>-<step>/ exists (exported with scripts/log_clips.py).
    """
    out = []
    for c in champions(state):
        if "scores" not in c:
            continue
        nums = {
            "better": c["better"],
            "worse": c["worse"],
            "before": f"{c['before']['ALL']:.1f}",
            "after": f"{c['scores']['ALL']:.1f}",
        }
        e = {"date": c["since"][:10], "champion": f"{c['run']} {c['step']}"}
        for lang, (title, text) in NEW_BEST.items():
            text = text.format(**nums)
            e[lang] = {
                "title": title,
                "text": text.translate(KM_DIGITS) if lang == "km" else text,
            }
        label = f"{c['run']}-{c['step']}"
        if clips and (clips / label).is_dir():
            e["audio"] = [{"src": label, "en": "Our model", "km": "ម៉ូដែលរបស់យើង"}]
        out.append(e)
    return out


def lab_log(path: Path, auto: list[dict] = ()) -> list[dict]:
    """The page's lab log, newest first: one JSON line per entry with date, en and km (title, text).

    `auto` entries (new best models) are dropped when a written entry names the same champion.
    """
    rows = (
        [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if path.exists()
        else []
    )
    written = {r.get("champion") for r in rows}
    rows += [e for e in auto if e["champion"] not in written]
    return sorted(rows, key=lambda r: r["date"], reverse=True)


def is_run_log(text: str, run: str) -> bool:
    """True if a training log writes into outputs/train/<run> (not a run whose name merely starts the same)."""
    return re.search(rf"outputs/train/{re.escape(run)}(?![\w-])", text) is not None


def parse_train_log(text: str) -> dict | None:
    """Latest epoch and step of a Coqui training log, with the recent median seconds per step."""
    text = ANSI.sub("", text)
    epochs, steps = EPOCH.findall(text), STEP.findall(text)
    if not epochs or not steps:
        return None
    epoch, last = epochs[-1]
    logged_at, in_epoch, per_epoch, step = steps[-1]
    times = [float(t) for t in STEP_TIME.findall(text)[-20:]]
    return {
        "epoch": int(epoch),
        "epochs": int(last) + 1,
        "step_in_epoch": int(in_epoch),
        "steps_per_epoch": int(per_epoch),
        "step": int(step),
        "step_seconds": round(statistics.median(times), 2) if times else None,
        "logged_at": logged_at,
        "loss": recent_loss(text),
    }


def recent_loss(text: str) -> list[list[float]]:
    """The last mel-spectrogram losses as [step, loss]; a resumed run's repeated steps keep their newest value."""
    by_step = {
        int(step): float(loss) for step, loss in LOSS.findall(ANSI.sub("", text))
    }
    return [[st, round(by_step[st], 3)] for st in sorted(by_step)[-LOSS_POINTS:]]


def _medians(df: pd.DataFrame) -> dict:
    df = df[df["category"] != "english"]
    cer = pd.Series(
        [mixed_cer(r, h) for r, h in zip(df["ref"], df["hyp"].fillna(""), strict=True)],
        index=df.index,
    )
    out = {"all": round(100 * float(cer.median()), 1)}
    for c in CATEGORIES:
        if (df["category"] == c).any():
            out[c] = round(100 * float(cer[df["category"] == c].median()), 1)
    return out


def step_scores(eval_dir: Path) -> list[dict]:
    """Median per-sentence CER (%) of every scored step folder under eval_dir, oldest first."""
    rows = []
    for d in eval_dir.iterdir() if eval_dir.exists() else []:
        if d.name.isdigit() and (d / "results.csv").exists():
            rows.append(
                {"step": int(d.name), **_medians(pd.read_csv(d / "results.csv"))}
            )
    return sorted(rows, key=lambda r: r["step"])


def base_scores(results_csv: Path) -> dict:
    return _medians(pd.read_csv(results_csv)) if results_csv.exists() else {}


def build_status(
    pos: dict | None,
    *,
    scores: list[dict],
    base: dict,
    state: str,
    run: str | None,
    now: datetime | None = None,
) -> dict:
    now = now or datetime.now(UTC)
    steps_left = finish_at = progress = None
    if pos:
        per = pos["steps_per_epoch"]
        steps_left = (per - pos["step_in_epoch"]) + (
            pos["epochs"] - pos["epoch"] - 1
        ) * per
        progress = round((pos["epoch"] + pos["step_in_epoch"] / per) / pos["epochs"], 4)
        if state == "training" and pos["step_seconds"]:
            finish_at = (
                now + timedelta(seconds=steps_left * pos["step_seconds"])
            ).isoformat()
    best = min(scores, key=lambda s: s["all"]) if scores else None
    return {
        "updated_at": now.isoformat(),
        "state": state,
        "run": run,
        **(pos or {}),
        "progress": progress,
        "steps_left": steps_left,
        "finish_at": finish_at,
        "scores": scores,
        "base": base,
        "best": {"step": best["step"], "all": best["all"]} if best else None,
    }
