"""Re-format the auto-generated LaTeX tables with sane formatters
(integers vs decimals vs scientific) instead of pandas' default 4g format.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TBL = ROOT / "output" / "tables"


def fmt_value(metric: str, v: float | int) -> str:
    """Choose a formatter based on metric name heuristics."""
    name = metric.lower()
    if any(k in name for k in ("nodes", "edges", "components", "size")):
        return f"{int(v):,}"
    if "degree centrality" in name:
        return f"{v:.5f}"
    if "avg degree" in name and "path" not in name:
        return f"{v:.2f}"
    if "path" in name or "diameter" in name:
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    if "alpha" in name or "xmin" in name or "slope" in name:
        return f"{v:.3f}"
    if "r_squared" in name or name == "powerlaw_vs_lognormal_r":
        return f"{v:.3f}"
    if "p" == name.strip() or "_p" in name:
        return f"{v:.2e}"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _esc(s: object) -> str:
    """Escape LaTeX-special chars in literal cell text. Skip math/markup."""
    s = str(s)
    if "$" in s or "\\" in s:
        return s  # already contains math/markup
    return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def to_latex(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    cols = " & ".join(_esc(c) for c in df.columns) + " \\\\"
    body = []
    for _, row in df.iterrows():
        body.append(" & ".join(_esc(c) for c in row) + " \\\\")
    out = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{'l' + 'r' * (len(df.columns) - 1)}}}\n"
        "\\toprule\n"
        f"{cols}\n"
        "\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    path.write_text(out)


def polish_two_column(name: str, caption: str, label: str) -> None:
    csv = TBL / f"{name}.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    df["Value"] = [fmt_value(m, v) for m, v in zip(df["Metric"], df["Value"])]
    to_latex(df, TBL / f"{name}.tex", caption, label)


def polish_components() -> None:
    csv = TBL / "task3_top_components.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    df["nodes"] = df["nodes"].apply(lambda v: f"{int(v):,}")
    df["edges"] = df["edges"].apply(lambda v: f"{int(v):,}")
    df["density"] = df["density"].apply(lambda v: f"{v:.4g}")
    df = df.rename(
        columns=dict(rank="Rank", nodes="Nodes", edges="Edges", density="Density")
    )
    to_latex(
        df,
        TBL / "task3_top_components.tex",
        "Three largest connected components of $G$.",
        "tab:components",
    )


def polish_time_subgraphs() -> None:
    csv = TBL / "task6_time_subgraphs.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    df["nodes"] = df["nodes"].apply(lambda v: f"{int(v):,}")
    df["edges"] = df["edges"].apply(lambda v: f"{int(v):,}")
    df["avg_path"] = df["avg_path"].apply(
        lambda v: f"{v:.3f}" if pd.notna(v) else "-"
    )
    df["diameter"] = df["diameter"].apply(
        lambda v: f"{int(v)}" if pd.notna(v) and v >= 0 else "-"
    )
    df = df.rename(
        columns=dict(
            t="t",
            nodes="$|V|$",
            edges="$|E|$",
            avg_path="Avg path (LCC)",
            diameter="Diameter (LCC)",
        )
    )
    to_latex(
        df,
        TBL / "task6_time_subgraphs.tex",
        "Per-increment subgraph metrics (avg path / diameter computed on the LCC of each subgraph).",
        "tab:time-sub",
    )


def polish_triangles() -> None:
    csv = TBL / "task7_triangles.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    df["triangles"] = df["triangles"].apply(lambda v: f"{int(v):,}")
    df = df.rename(columns=dict(t="t", triangles="Triangles"))
    to_latex(
        df,
        TBL / "task7_triangles.tex",
        "Number of triangles per time increment.",
        "tab:triangles",
    )


def polish_sentiment() -> None:
    csv = TBL / "task8_sentiment.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    df["pos"] = df["pos"].apply(lambda v: f"{v:+.3f}")
    df["neg"] = df["neg"].apply(lambda v: f"{v:+.3f}")
    df["overall"] = df["overall"].apply(lambda v: f"{v:+.3f}")
    df["total_pos"] = df["total_pos"].apply(lambda v: f"{int(v):,}")
    df["total_neg"] = df["total_neg"].apply(lambda v: f"{int(v):,}")
    df = df.rename(
        columns=dict(
            t="t",
            pos="Avg $s^{+}$",
            neg="Avg $s^{-}$",
            overall="Avg overall",
            total_pos="$\\sum s^{+}$",
            total_neg="$\\sum s^{-}$",
        )
    )
    to_latex(
        df,
        TBL / "task8_sentiment.tex",
        "Sentiment statistics per time increment.",
        "tab:sent",
    )


def polish_sis() -> None:
    csv = TBL / "task9_sis.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    df["beta"] = df["beta"].apply(lambda v: f"{v:.3f}")
    df["lam"] = df["lam"].apply(lambda v: f"{v:.2f}")
    df["final_infected"] = df["final_infected"].apply(lambda v: f"{int(v):,}")
    df["cumulative"] = df["cumulative"].apply(lambda v: f"{int(v):,}")
    df = df.rename(
        columns=dict(
            beta="$\\beta$",
            lam="$\\lambda$",
            final_infected="Final infected",
            cumulative="Cumulative infected",
        )
    )
    to_latex(
        df,
        TBL / "task9_sis.tex",
        "Top SIS parameter pairs ranked by closeness to observed cumulative negative sentiment.",
        "tab:sis",
    )


def main() -> None:
    polish_two_column(
        "task2_global_stats",
        "Global statistics of the hashtag co-occurrence graph $G$.",
        "tab:global",
    )
    polish_components()
    polish_two_column(
        "task5_powerlaw",
        "Power-law fit of the degree distribution.",
        "tab:powerlaw",
    )
    polish_time_subgraphs()
    polish_triangles()
    polish_sentiment()
    polish_sis()
    print("polished tables")


if __name__ == "__main__":
    main()
