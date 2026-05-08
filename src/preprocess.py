"""Stream TweetsCOV19.tsv.gz, filter to Oct-Nov 2019, keep rows with hashtags.

Writes a compact parquet with:
  user_id, ts (epoch), pos, neg, hashtags (list[str])
"""

from __future__ import annotations

import gzip
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "TweetsCOV19.tsv.gz"
OUT = ROOT / "output" / "tweets_oct_nov_2019.parquet"

# Twitter timestamp format: "Mon Sep 30 22:00:37 +0000 2019"
TS_FMT = "%a %b %d %H:%M:%S %z %Y"

START = datetime(2019, 10, 1, tzinfo=timezone.utc)
END = datetime(2019, 12, 1, tzinfo=timezone.utc)  # exclusive


def parse_ts(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, TS_FMT)
    except ValueError:
        return None


def main() -> None:
    rows: list[dict] = []
    seen = 0
    kept = 0

    with gzip.open(SRC, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            seen += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 12:
                continue

            ts_str = parts[2]
            ts = parse_ts(ts_str)
            if ts is None:
                continue
            if ts < START or ts >= END:
                continue

            hashtags_field = parts[10].strip()
            if not hashtags_field or hashtags_field == "null;":
                continue

            tags = [h.strip().lower() for h in hashtags_field.split() if h.strip()]
            if not tags:
                continue

            sent = parts[8].strip()
            try:
                pos_str, neg_str = sent.split()
                pos = int(pos_str)
                neg = int(neg_str)
            except ValueError:
                pos, neg = 0, 0

            rows.append(
                dict(
                    user_id=parts[1],
                    ts=int(ts.timestamp()),
                    pos=pos,
                    neg=neg,
                    hashtags=tags,
                )
            )
            kept += 1

            if seen % 1_000_000 == 0:
                print(f"  scanned={seen:,}  kept={kept:,}", file=sys.stderr)

    print(f"Scan done: scanned={seen:,}  kept={kept:,}", file=sys.stderr)

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"Wrote {len(df):,} rows -> {OUT}")


if __name__ == "__main__":
    main()
