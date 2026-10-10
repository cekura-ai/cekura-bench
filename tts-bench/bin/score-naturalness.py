#!/usr/bin/env python
"""Score a run's one-shot recordings for naturalness with a listener model.

    python bin/score-naturalness.py <run>...

Appends to ``naturalness.jsonl`` in each run as results land, and skips any
cell already scored, so the command can be repeated after an interruption.
Needs torch and torchaudio, which the harness does not: run it from an
environment built with ``requirements-naturalness.txt``. The first run
fetches the model through torch.hub at a pinned tag.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tts_bench import naturalness, store  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args(argv)
    model = None
    for run in args.runs:
        root = Path(run)
        pending = [r for r in naturalness.recordings(root) if r["cell_id"] not in naturalness.scored(root)]
        if not pending:
            print(f"{root.name}: already scored")
            continue
        model = model or naturalness.load_model()
        done = naturalness.score_run(root, model)
        if (root / store.MANIFEST).exists():        # the journal is a deliberate addition to an archived run
            store.write_manifest(root)
        print(f"{root.name}: scored {done} recordings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
