"""Export one model's clips for the lab log: the same sentences for every entry, so progress can be heard.

uv run python scripts/log_clips.py run-a outputs/train/eval/distill_v1_A_dp00/17000/wav
Writes web/audio/log/<label>/<id>.m4a. An entry in ops/site_log.jsonl then lists it:
"audio": [{"src": "run-a", "en": "Our model", "km": "..."}]
"""

import argparse
from pathlib import Path

from site_samples import export

IDS = ("pure-34", "pure-01")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("folder", type=Path, help="folder of <id>.wav")
    args = ap.parse_args()
    for i in IDS:
        export(args.folder / f"{i}.wav", f"log/{args.label}", i)


if __name__ == "__main__":
    main()
