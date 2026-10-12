"""Normalize external decisions to canonical JSONL without materializing a corpus."""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

from futurix_decision.datasets import load_decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON, JSONL or Hugging Face dataset ID")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    source = load_decisions(args.input, split=args.split, streaming=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = islice(source, args.limit) if args.limit is not None else source
    count = 0
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} decision records to {output}")


if __name__ == "__main__":
    main()
