"""Add a "speak" field with Khmer digits to a teacher input file (in place); the training text is unchanged.

uv run python scripts/teacher_digits.py outputs/distill/v1w/tts_in.jsonl
"""

import json
import sys
from pathlib import Path

from samleng.text import teacher_digits

path = Path(sys.argv[1])
rows = [
    json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
]
changed = 0
for row in rows:
    speak = teacher_digits(row["text"])
    if speak != row["text"]:
        row["speak"] = speak
        changed += 1
path.write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
)
print(f"{path}: {changed} of {len(rows)} rows read with Khmer digits")
