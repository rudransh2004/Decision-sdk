"""Run one large released decision model through the SDK without retaining weights.

The child walkthrough loads decisions through Futurix, performs native inference,
fine-tunes, and reloads the publisher-format export. Its private cache/export
are removed only after that child exits; pre-existing caches are never touched.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


MIN_FREE_GIB = {"clef-flash": 50, "clef": 112, "pplx-decider": 112}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MIN_FREE_GIB)
    parser.add_argument("--scratch", type=Path, default=Path("/tmp"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-examples", type=int, default=2)
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    if args.max_examples < 1:
        parser.error("--max-examples must be positive")
    available = shutil.disk_usage(args.scratch).free
    required = MIN_FREE_GIB[args.model] * 1024**3
    if available < required:
        parser.error(
            f"{args.model} requires at least {MIN_FREE_GIB[args.model]} GiB free "
            f"on {args.scratch} for its downloaded weights and full native export; "
            f"only {available / 1024**3:.1f} GiB is available"
        )

    with tempfile.TemporaryDirectory(prefix=f"futurix-{args.model}-", dir=args.scratch) as scratch:
        directory = Path(scratch)
        environment = dict(os.environ, HF_HOME=str(directory / "hf"))
        command = [
            sys.executable,
            str(Path(__file__).with_name("test_released_decisions.py")),
            args.model,
            "--device", args.device,
            "--max-examples", str(args.max_examples),
            "--train",
            "--output", str(directory / "export"),
        ]
        if args.dataset is not None:
            command.extend(["--dataset", str(args.dataset)])
        print(
            f"SDK smoke test for {args.model}; private cache and export: {directory}",
            flush=True,
        )
        subprocess.run(command, env=environment, check=True)
    print(f"SDK smoke test passed for {args.model}; only temporary weights were removed", flush=True)


if __name__ == "__main__":
    main()
