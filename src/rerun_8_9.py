"""Re-run only tasks 8 and 9 (cheap; they don't recompute path lengths)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from analysis import GPATH, time_increments, task8, task9  # type: ignore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with open(GPATH, "rb") as f:
        G = pickle.load(f)
    subs = time_increments(G, 10)  # cheap induced-subgraph rebuild
    sentiment = task8(G, subs)
    sis = task9(subs, sentiment)

    out_path = ROOT / "output" / "results.json"
    if out_path.exists():
        prev = json.loads(out_path.read_text())
    else:
        prev = {}
    prev["sentiment"] = {k: dict(v) for k, v in sentiment.items()}
    prev["sis_target"] = sis.get("target")
    prev["sis_best"] = {
        k: v for k, v in (sis.get("best") or {}).items() if k != "series"
    }
    out_path.write_text(json.dumps(prev, indent=2, default=float))
    print("rerun done")


if __name__ == "__main__":
    main()
