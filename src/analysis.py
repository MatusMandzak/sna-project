"""Tasks 2-9: graph analytics over the hashtag co-occurrence network.

Produces figures into output/figures and tables into output/tables.
Tables are written as both .csv and LaTeX (.tex) so the report can include them.
"""

from __future__ import annotations

import json
import pickle
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import powerlaw
from networkx.algorithms.approximation import average_clustering
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
GPATH = ROOT / "output" / "graph.gpickle"
FIG = ROOT / "output" / "figures"
TBL = ROOT / "output" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TBL.mkdir(parents=True, exist_ok=True)

RNG = random.Random(42)
NPRNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_graph() -> nx.Graph:
    with open(GPATH, "rb") as f:
        return pickle.load(f)


def df_to_latex(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    df.to_csv(path.with_suffix(".csv"), index=False)
    with open(path.with_suffix(".tex"), "w") as f:
        f.write(
            df.to_latex(
                index=False,
                escape=True,
                float_format=lambda x: f"{x:,.4g}",
                caption=caption,
                label=label,
            )
        )


def approx_avg_path_length(G: nx.Graph, sample: int = 80) -> float:
    """Sample-based average shortest path length (works on disconnected graphs).

    Picks `sample` source nodes, runs single-source BFS from each,
    averages over reachable target nodes.
    """
    if G.number_of_nodes() == 0:
        return float("nan")
    nodes = list(G.nodes())
    if len(nodes) <= sample:
        sources = nodes
    else:
        sources = RNG.sample(nodes, sample)
    total = 0.0
    pairs = 0
    for s in sources:
        lengths = nx.single_source_shortest_path_length(G, s)
        # exclude the source itself
        for t, d in lengths.items():
            if t == s:
                continue
            total += d
            pairs += 1
    if pairs == 0:
        return float("nan")
    return total / pairs


# ---------------------------------------------------------------------------
# Task 2 — global metrics
# ---------------------------------------------------------------------------


def task2(G: nx.Graph) -> dict:
    print("[task2] start", flush=True)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    components = list(nx.connected_components(G))
    print("[task2] components done", flush=True)
    deg_cent = nx.degree_centrality(G)
    vals = np.fromiter(deg_cent.values(), dtype=float)
    print("[task2] degree centrality done", flush=True)
    avg_path = approx_avg_path_length(G, sample=200)
    print(f"[task2] avg path: {avg_path:.3f}", flush=True)

    summary = {
        "Nodes": n,
        "Edges": m,
        "Components": len(components),
        "Largest CC size": max(len(c) for c in components),
        "Avg degree": 2 * m / n,
        "Min degree centrality": float(vals.min()),
        "Max degree centrality": float(vals.max()),
        "Mean degree centrality": float(vals.mean()),
        "Avg shortest path (sampled)": avg_path,
    }
    df = pd.DataFrame(
        [(k, v) for k, v in summary.items()], columns=["Metric", "Value"]
    )
    df_to_latex(
        df,
        TBL / "task2_global_stats",
        "Global statistics of the hashtag co-occurrence graph $G$.",
        "tab:global",
    )
    print("Task 2 done:", summary)
    return summary


# ---------------------------------------------------------------------------
# Task 3 — top 3 components
# ---------------------------------------------------------------------------


def task3(G: nx.Graph) -> list[set]:
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    rows = []
    for i, c in enumerate(comps[:3], 1):
        sub = G.subgraph(c)
        rows.append(
            dict(
                rank=i,
                nodes=sub.number_of_nodes(),
                edges=sub.number_of_edges(),
                density=nx.density(sub),
            )
        )
    df = pd.DataFrame(rows)
    df_to_latex(
        df,
        TBL / "task3_top_components",
        "Three largest connected components of $G$ ordered by size.",
        "tab:components",
    )
    print("Task 3 done:", rows)
    return comps[:3]


# ---------------------------------------------------------------------------
# Task 4 — degree centrality distribution and CDF
# ---------------------------------------------------------------------------


def task4(G: nx.Graph) -> np.ndarray:
    deg_cent = nx.degree_centrality(G)
    vals = np.array(sorted(deg_cent.values()))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(vals, bins=80, color="#3471a8", edgecolor="white")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Degree centrality")
    ax[0].set_ylabel("Count (log)")
    ax[0].set_title("Degree centrality distribution")
    ax[0].grid(True, alpha=0.3)

    cdf_y = np.arange(1, len(vals) + 1) / len(vals)
    ax[1].plot(vals, cdf_y, color="#a83434")
    ax[1].set_xlabel("Degree centrality")
    ax[1].set_ylabel("Cumulative fraction")
    ax[1].set_title("Cumulative degree centrality")
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "task4_degree_distributions.pdf")
    plt.close(fig)
    print("Task 4 done")
    return vals


# ---------------------------------------------------------------------------
# Task 5 — power-law fit with R^2 on log-log degree pdf
# ---------------------------------------------------------------------------


def task5(G: nx.Graph) -> dict:
    degrees = np.array([d for _, d in G.degree()])
    degrees = degrees[degrees > 0]
    fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
    alpha = float(fit.alpha)
    xmin = float(fit.xmin)

    # Compare power-law to lognormal with `distribution_compare` to check fit
    R, p = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)

    # R^2 on the log-binned PDF in the tail
    bins = np.unique(np.round(np.logspace(0, np.log10(degrees.max()), 60)))
    bins = bins[bins >= xmin]
    if len(bins) < 4:
        bins = np.unique(np.geomspace(max(1, xmin), degrees.max(), 30))
    hist, edges = np.histogram(degrees, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = hist > 0
    if mask.sum() < 3:
        r_squared = float("nan")
        slope = float("nan")
    else:
        log_x = np.log10(centers[mask])
        log_y = np.log10(hist[mask])
        slope, intercept, r_value, _, _ = stats.linregress(log_x, log_y)
        r_squared = float(r_value**2)

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(centers[mask], hist[mask], s=22, color="#3471a8", label="empirical")
        xs = np.linspace(centers[mask].min(), centers[mask].max(), 100)
        ax.plot(
            xs,
            10**intercept * xs**slope,
            "--",
            color="#a83434",
            label=f"power-law fit: slope={slope:.2f}",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Degree $k$")
        ax.set_ylabel("$P(k)$")
        ax.set_title(f"Power-law fit  (alpha={alpha:.2f}, $R^2$={r_squared:.3f})")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "task5_powerlaw.pdf")
        plt.close(fig)

    out = dict(
        alpha=alpha,
        xmin=xmin,
        slope_log_log=float(slope),
        r_squared=float(r_squared),
        powerlaw_vs_lognormal_R=float(R),
        powerlaw_vs_lognormal_p=float(p),
    )
    df = pd.DataFrame([(k, v) for k, v in out.items()], columns=["Metric", "Value"])
    df_to_latex(
        df,
        TBL / "task5_powerlaw",
        "Power-law fit of the degree distribution.",
        "tab:powerlaw",
    )
    print("Task 5 done:", out)
    return out


# ---------------------------------------------------------------------------
# Task 6 — split timeline into 10 increments and analyze subgraphs
# ---------------------------------------------------------------------------


def time_increments(G: nx.Graph, k: int = 10) -> list[nx.Graph]:
    ts = np.array([d["ts"] for _, d in G.nodes(data=True)])
    edges_lo = ts.min()
    edges_hi = ts.max()
    bin_edges = np.linspace(edges_lo, edges_hi, k + 1)
    subgraphs: list[nx.Graph] = []
    for i in range(k):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == k - 1:
            members = [n for n, d in G.nodes(data=True) if lo <= d["ts"] <= hi]
        else:
            members = [n for n, d in G.nodes(data=True) if lo <= d["ts"] < hi]
        sub = G.subgraph(members).copy()
        subgraphs.append(sub)
    return subgraphs


def task6(G: nx.Graph) -> list[nx.Graph]:
    subs = time_increments(G, 10)
    rows = []
    for i, S in enumerate(subs, 1):
        print(f"[task6] increment {i}: |V|={S.number_of_nodes()} "
              f"|E|={S.number_of_edges()}", flush=True)
        if S.number_of_nodes() == 0:
            rows.append(dict(t=i, nodes=0, edges=0, avg_path=float("nan"),
                             diameter=float("nan")))
            continue
        n = S.number_of_nodes()
        m = S.number_of_edges()
        comps = list(nx.connected_components(S))
        biggest = max(comps, key=len) if comps else set()
        sub = S.subgraph(biggest)
        avg_path = approx_avg_path_length(sub, sample=60)
        # diameter — sampled via eccentricity from a few sources to upper-bound
        diam = -1
        sample_n = min(20, sub.number_of_nodes())
        if sample_n > 0:
            for s in RNG.sample(list(sub.nodes()), sample_n):
                ecc = max(nx.single_source_shortest_path_length(sub, s).values())
                if ecc > diam:
                    diam = ecc
        rows.append(
            dict(
                t=i,
                nodes=n,
                edges=m,
                avg_path=avg_path,
                diameter=diam,
            )
        )
        print(f"[task6] increment {i} metrics done: avg_path={avg_path:.3f} "
              f"diam={diam}", flush=True)

    df = pd.DataFrame(rows)
    df_to_latex(
        df,
        TBL / "task6_time_subgraphs",
        "Global metrics of the ten equal-time-increment subgraphs.",
        "tab:time-sub",
    )

    # plot subgraph evolution: nodes / edges over time
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(df.t, df.nodes, "o-", color="#3471a8")
    ax[0].set_xlabel("Time increment")
    ax[0].set_ylabel("Nodes")
    ax[0].set_title("Nodes per increment")
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(df.t, df.edges, "s-", color="#a83434")
    ax[1].set_xlabel("Time increment")
    ax[1].set_ylabel("Edges")
    ax[1].set_title("Edges per increment")
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "task6_subgraph_evolution.pdf")
    plt.close(fig)

    # network-style plot of each subgraph (LCC sample)
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for i, (S, ax) in enumerate(zip(subs, axes.flat), 1):
        if S.number_of_nodes() == 0:
            ax.axis("off")
            continue
        comps = list(nx.connected_components(S))
        biggest = max(comps, key=len)
        H = S.subgraph(biggest)
        # downsample large LCCs for readability
        if H.number_of_nodes() > 250:
            seed = next(iter(biggest))
            visited = {seed}
            queue = [seed]
            while queue and len(visited) < 250:
                v = queue.pop(0)
                for w in H.neighbors(v):
                    if w not in visited:
                        visited.add(w)
                        queue.append(w)
            H = H.subgraph(visited)
        pos = nx.spring_layout(H, k=0.4, seed=7, iterations=15)
        nx.draw_networkx_nodes(H, pos, node_size=10, node_color="#3471a8", ax=ax)
        nx.draw_networkx_edges(H, pos, alpha=0.25, width=0.4, ax=ax)
        ax.set_title(f"t={i}  |V|={S.number_of_nodes()}", fontsize=9)
        ax.set_axis_off()
        print(f"[task6] increment {i} drawn", flush=True)
    fig.suptitle("LCC sample of each time increment", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "task6_subgraph_drawings.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Task 6 done")
    return subs


# ---------------------------------------------------------------------------
# Task 7 — triangle counts per increment
# ---------------------------------------------------------------------------


def task7(subs: list[nx.Graph]) -> list[int]:
    counts = []
    for i, S in enumerate(subs, 1):
        if S.number_of_edges() == 0:
            counts.append(0)
            print(f"[task7] increment {i}: 0 (empty)", flush=True)
            continue
        # nx.triangles returns dict node -> # triangles. total = sum/3.
        t = nx.triangles(S)
        c = int(sum(t.values()) // 3)
        counts.append(c)
        print(f"[task7] increment {i}: {c} triangles", flush=True)
    df = pd.DataFrame(
        {"t": list(range(1, len(counts) + 1)), "triangles": counts}
    )
    df_to_latex(
        df,
        TBL / "task7_triangles",
        "Triangle counts per time increment.",
        "tab:triangles",
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df.t, df.triangles, "o-", color="#3a8a3a", linewidth=2)
    ax.set_xlabel("Time increment")
    ax.set_ylabel("Number of triangles")
    ax.set_title("Triangle count over time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "task7_triangles.pdf")
    plt.close(fig)
    print("Task 7 done:", counts)
    return counts


# ---------------------------------------------------------------------------
# Task 8 — sentiment evolution
# ---------------------------------------------------------------------------


def task8(G: nx.Graph, subs: list[nx.Graph]) -> dict:
    """Per-increment sentiment.

    The project spec defines node sentiment via the user's own positive and
    negative sentiment scores; we aggregate per (user, increment) so that a
    user contributes to the increment they are tweeting in.
    """
    df = pd.read_parquet(ROOT / "output" / "tweets_oct_nov_2019.parquet")
    ts = df["ts"].to_numpy()
    lo, hi = ts.min(), ts.max()
    bin_edges = np.linspace(lo, hi, 11)
    df["bin"] = np.minimum(np.searchsorted(bin_edges, ts, side="right") - 1, 9)

    rows = []
    for i, S in enumerate(subs, 1):
        active_users = set(S.nodes())
        sub = df[(df["bin"] == (i - 1)) & (df["user_id"].isin(active_users))]
        if sub.empty:
            rows.append(dict(t=i, pos=0.0, neg=0.0, overall=0.0,
                             total_pos=0, total_neg=0))
            continue
        per_user = sub.groupby("user_id")[["pos", "neg"]].sum()
        rows.append(
            dict(
                t=i,
                pos=float(per_user["pos"].mean()),
                neg=float(per_user["neg"].mean()),
                overall=float(per_user["pos"].mean()
                              + per_user["neg"].mean()),
                total_pos=int(per_user["pos"].sum()),
                total_neg=int(per_user["neg"].sum()),
            )
        )
    df = pd.DataFrame(rows)
    df_to_latex(
        df,
        TBL / "task8_sentiment",
        "Average sentiment per time increment "
        "(neg is negative-valued so an overall rise indicates more positive mood).",
        "tab:sent",
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df.t, df.pos, "o-", color="#2a8a2a", label="avg positive")
    ax.plot(df.t, df.neg, "s-", color="#a82a2a", label="avg negative")
    ax.plot(df.t, df.overall, "d--", color="#444",
            label="overall (pos+neg)")
    ax.set_xlabel("Time increment")
    ax.set_ylabel("Average sentiment")
    ax.set_title("Sentiment evolution across the 10 time increments")
    ax.axhline(0, linewidth=0.5, color="black")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "task8_sentiment.pdf")
    plt.close(fig)
    print("Task 8 done")
    return {r["t"]: r for r in rows}


# ---------------------------------------------------------------------------
# Task 9 — SIS propagation on first increment
# ---------------------------------------------------------------------------


def task9(subs: list[nx.Graph], sentiment: dict) -> dict:
    import ndlib.models.ModelConfig as mc
    import ndlib.models.epidemics as ep

    G1 = subs[0]
    if G1.number_of_nodes() == 0:
        print("Task 9: skip, empty t=1 subgraph")
        return {}
    # Use the LCC of t=1 for stable simulation
    biggest = max(nx.connected_components(G1), key=len)
    H = G1.subgraph(biggest).copy()
    print(f"SIS host graph: |V|={H.number_of_nodes()} |E|={H.number_of_edges()}")

    # observed total negative sentiment across the whole timeline (target)
    target = sum(abs(s["total_neg"]) for s in sentiment.values())
    print(f"observed |total_neg| target (sum across all increments): {target}")

    # choose initial fraction of infected: the fraction of nodes in t=1 with
    # negative sentiment > positive sentiment.
    init_frac = float(
        np.mean(
            [
                d["neg"] < 0 and abs(d["neg"]) > d["pos"]
                for _, d in H.nodes(data=True)
            ]
        )
    )
    init_frac = max(0.01, min(0.4, init_frac))
    print(f"initial infected fraction = {init_frac:.3f}")

    iterations = 10  # one per time increment
    sweep = []
    for beta in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        for lam in (0.02, 0.05, 0.1, 0.2, 0.4):
            model = ep.SISModel(H)
            cfg = mc.Configuration()
            cfg.add_model_parameter("beta", beta)
            cfg.add_model_parameter("lambda", lam)
            cfg.add_model_parameter("fraction_infected", init_frac)
            model.set_initial_status(cfg)
            it = model.iteration_bunch(iterations)
            infected_per_step = [step["node_count"][1] for step in it]
            cumulative_infected = sum(infected_per_step)
            sweep.append(
                dict(
                    beta=beta,
                    lam=lam,
                    final_infected=infected_per_step[-1],
                    cumulative=cumulative_infected,
                    series=infected_per_step,
                )
            )

    sweep.sort(key=lambda r: abs(r["cumulative"] - target))
    best = sweep[0]
    print(f"best (beta={best['beta']}, lambda={best['lam']}): "
          f"cumulative={best['cumulative']} target={target}")

    # plot top 3 series + horizontal target line
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in sweep[:3]:
        ax.plot(
            range(1, iterations + 1),
            r["series"],
            "-o",
            label=f"beta={r['beta']}, lam={r['lam']}",
        )
    ax.set_xlabel("Time step")
    ax.set_ylabel("# infected nodes (negative-sentiment carriers)")
    ax.set_title("SIS propagation on t=1 subgraph")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "task9_sis_topfits.pdf")
    plt.close(fig)

    df = pd.DataFrame(
        [
            dict(beta=r["beta"], lam=r["lam"],
                 final_infected=r["final_infected"],
                 cumulative=r["cumulative"])
            for r in sweep[:8]
        ]
    )
    df_to_latex(
        df,
        TBL / "task9_sis",
        "SIS sweep ranked by closeness to the observed cumulative negative sentiment.",
        "tab:sis",
    )
    return dict(target=target, best=best, sweep=sweep[:8])


# ---------------------------------------------------------------------------


def main() -> None:
    G = load_graph()

    summary2 = task2(G)
    task3(G)
    task4(G)
    fit5 = task5(G)
    subs = task6(G)
    triangles = task7(subs)
    sentiment = task8(G, subs)
    sis = task9(subs, sentiment)

    with open(ROOT / "output" / "results.json", "w") as f:
        json.dump(
            dict(
                global_summary=summary2,
                powerlaw=fit5,
                triangles=triangles,
                sentiment={k: dict(v) for k, v in sentiment.items()},
                sis_target=sis.get("target"),
                sis_best={k: v for k, v in (sis.get("best") or {}).items()
                          if k != "series"},
            ),
            f,
            indent=2,
            default=float,
        )
    print("All tasks done.")


if __name__ == "__main__":
    main()
