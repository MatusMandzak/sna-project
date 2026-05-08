"""Generate auxiliary tables/figures used by the report (top hashtags, etc.)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TBL = ROOT / "output" / "tables"
FIG = ROOT / "output" / "figures"


def top_hashtags() -> None:
    df = pd.read_parquet(ROOT / "output" / "tweets_oct_nov_2019.parquet")
    cnt: Counter[str] = Counter()
    for tags in df["hashtags"]:
        cnt.update(tags)
    top = cnt.most_common(15)
    out = pd.DataFrame(top, columns=["Hashtag", "Tweets"])
    body = []
    for h, c in top:
        body.append(f"\\texttt{{\\#{h}}} & {c:,} \\\\")
    out.to_csv(TBL / "top_hashtags.csv", index=False)
    (TBL / "top_hashtags.tex").write_text(
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Most frequent hashtags in the October--November 2019 slice.}\n"
        "\\label{tab:top-hashtags}\n"
        "\\begin{tabular}{lr}\n"
        "\\toprule\n"
        "Hashtag & Tweets \\\\\n"
        "\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    # daily activity figure
    daily = (
        pd.to_datetime(df["ts"], unit="s", utc=True)
        .dt.tz_convert("UTC")
        .dt.date
    )
    counts = daily.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(counts.index, counts.values, color="#3471a8")
    ax.set_xlabel("Date")
    ax.set_ylabel("Tweets with hashtags")
    ax.set_title("Daily volume of hashtag-bearing tweets")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "daily_volume.pdf")
    plt.close(fig)


def main() -> None:
    top_hashtags()
    print("extra tables/figures generated")


if __name__ == "__main__":
    main()
