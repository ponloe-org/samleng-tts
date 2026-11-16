"""A run's best all-Khmer error, so a queued job can skip itself when an earlier result says it is pointless."""

import re
from pathlib import Path

PICK_ROW = re.compile(r"^\| step\d+ .*\| *([\d.]+) *\|$", re.M)
SUMMARY_ROW = re.compile(r"^\| step\d+ +\| ALL +\|[^|]+\|[^|]+\| +([\d.]+)", re.M)


def best_all(eval_dir: Path) -> tuple[float, str] | None:
    """Lowest ALL median CER (%) of a scored run: from its 3-seed pick.md if there is one, else its summary.md."""
    for name, row, source in (
        ("pick.md", PICK_ROW, "3-seed pick"),
        ("summary.md", SUMMARY_ROW, "single-seed summary"),
    ):
        f = eval_dir / name
        # pick.md's second table (pairs of steps) ends in p-values, not errors.
        text = f.read_text().split("## Pairs")[0] if f.exists() else ""
        vals = [float(v) for v in row.findall(text)]
        if vals:
            return min(vals), source
    return None
