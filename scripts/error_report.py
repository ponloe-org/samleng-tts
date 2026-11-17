"""List what a scored checkpoint misreads that the teacher reads correctly.

uv run python scripts/error_report.py distill_v2_calm --step 6000 [--teacher teacher_clear/0]

Reads the cached results of eval_checkpoint.py and writes errors_<step>.md next to the run's summary.
"""

import argparse

import pandas as pd

from samleng.eval.errors import hard_clusters, student_only_confusions, worst_sentences
from samleng.paths import OUTPUT_DIR

EVAL = OUTPUT_DIR / "train" / "eval"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="folder name under outputs/train/eval")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--teacher", default="teacher_clear/0")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    student = pd.read_csv(EVAL / args.run / str(args.step) / "results.csv")
    teacher = pd.read_csv(EVAL / args.teacher / "results.csv")
    worst = worst_sentences(student, teacher, args.top)
    worst[["cer", "cer_teacher", "gap"]] = (
        worst[["cer", "cer_teacher", "gap"]] * 100
    ).round(1)
    by_category = worst.category.value_counts().to_dict()
    pairs = pd.DataFrame(
        student_only_confusions(student, teacher)[:40],
        columns=["written", "heard", "sentences", "teacher_sentences"],
    )
    hard = pd.DataFrame(
        hard_clusters(student, teacher)[:40],
        columns=["written", "sentences", "teacher_sentences"],
    )
    out = EVAL / args.run / f"errors_{args.step}.md"
    out.write_text(
        f"# {args.run} step {args.step}: errors the teacher does not make\n\n"
        f"Teacher: `{args.teacher}`. Counts are sentences out of {len(student)}. "
        "The recognizer mishears too: trust patterns that repeat, not single rows.\n\n"
        f"## Written parts the student misreads most (worst first)\n\n{hard.to_markdown(index=False)}\n\n"
        f"## Repeated misreadings: written → heard\n\n{pairs.to_markdown(index=False)}\n\n"
        f"## The {len(worst)} sentences furthest behind the teacher (by category: {by_category})\n\n"
        f"{worst.to_markdown(index=False)}\n",
        encoding="utf-8",
    )
    print(f"{out}: {len(hard)} hard parts, {len(pairs)} repeated misreadings")


if __name__ == "__main__":
    main()
