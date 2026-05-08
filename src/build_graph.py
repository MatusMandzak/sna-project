"""Build hashtag co-occurrence graph from filtered tweets.

Nodes = Twitter user IDs.
Edge (u, v) iff u and v posted at least one tweet sharing the same hashtag.
Edge weight = number of distinct hashtags they share.

To keep the graph tractable while still meaningful, hashtags whose user-count
exceeds HASHTAG_USER_CAP are dropped (these are mega-trending tags that produce
star-like cliques and dominate edge count without adding social signal).
"""

from __future__ import annotations

import pickle
import sqlite3
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "output" / "tweets_oct_nov_2019.parquet"
OUT_GRAPH = ROOT / "output" / "graph.gpickle"
OUT_INDEX = ROOT / "output" / "node_index.pkl"
OUT_DB = ROOT / "output" / "adjacency.sqlite"

# A hashtag with > CAP distinct users is considered too "viral" and is dropped.
HASHTAG_USER_CAP = 150


def main() -> None:
    df = pd.read_parquet(SRC)
    print(f"Loaded {len(df):,} tweets")

    # hashtag -> set of users who used it (and earliest timestamp per user)
    tag_users: dict[str, set[str]] = defaultdict(set)
    # for time-stamped graph we also need per-user earliest tweet ts
    user_first_ts: dict[str, int] = {}
    user_pos: dict[str, int] = defaultdict(int)
    user_neg: dict[str, int] = defaultdict(int)
    user_count: dict[str, int] = defaultdict(int)

    for u, ts, pos, neg, tags in zip(
        df.user_id, df.ts, df.pos, df.neg, df.hashtags
    ):
        for h in tags:
            tag_users[h].add(u)
        if u not in user_first_ts or ts < user_first_ts[u]:
            user_first_ts[u] = int(ts)
        user_pos[u] += int(pos)
        user_neg[u] += int(neg)
        user_count[u] += 1

    print(f"hashtags: {len(tag_users):,}")
    sizes = [len(s) for s in tag_users.values()]
    print(
        f"hashtag-user fanout: max={max(sizes)} | "
        f">CAP={sum(1 for s in sizes if s > HASHTAG_USER_CAP)} dropped"
    )

    # edge weight = # of co-occurring hashtags
    edge_w: dict[tuple[str, str], int] = defaultdict(int)
    n_pairs = 0
    for h, users in tag_users.items():
        if len(users) < 2 or len(users) > HASHTAG_USER_CAP:
            continue
        users_sorted = sorted(users)
        for u, v in combinations(users_sorted, 2):
            edge_w[(u, v)] += 1
            n_pairs += 1

    print(f"raw pair touches: {n_pairs:,} | unique edges: {len(edge_w):,}")

    G = nx.Graph()
    # add all nodes that appear in at least one kept hashtag (otherwise isolates)
    kept_users: set[str] = set()
    for h, users in tag_users.items():
        if 2 <= len(users) <= HASHTAG_USER_CAP:
            kept_users.update(users)
    # also include singletons (hashtag fanout=1) for completeness — they become isolates
    # but are real nodes per task spec. We keep only users involved in at least one
    # *kept* hashtag for tractable analysis.
    for u in kept_users:
        G.add_node(
            u,
            ts=user_first_ts[u],
            pos=user_pos[u],
            neg=user_neg[u],
            count=user_count[u],
        )

    G.add_weighted_edges_from(((u, v, w) for (u, v), w in edge_w.items()))
    print(f"graph: |V|={G.number_of_nodes():,} |E|={G.number_of_edges():,}")

    # Persist graph
    with open(OUT_GRAPH, "wb") as f:
        pickle.dump(G, f)
    print(f"wrote graph -> {OUT_GRAPH}")

    # Build sparse adjacency and dump to SQLite.
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    for u, v, w in G.edges(data="weight", default=1):
        i, j = idx[u], idx[v]
        rows.append(i)
        cols.append(j)
        data.append(int(w))
        rows.append(j)
        cols.append(i)
        data.append(int(w))
    n = len(nodes)
    A = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A.eliminate_zeros()
    print(f"adjacency: {A.shape}, nnz={A.nnz:,}")

    with open(OUT_INDEX, "wb") as f:
        pickle.dump(nodes, f)

    if OUT_DB.exists():
        OUT_DB.unlink()
    con = sqlite3.connect(OUT_DB)
    cur = con.cursor()
    cur.executescript(
        """
        CREATE TABLE nodes (idx INTEGER PRIMARY KEY, user_id TEXT NOT NULL);
        CREATE TABLE adjacency (
            row INTEGER NOT NULL,
            col INTEGER NOT NULL,
            weight INTEGER NOT NULL,
            PRIMARY KEY (row, col)
        );
        CREATE INDEX idx_adj_row ON adjacency(row);
        """
    )
    cur.executemany("INSERT INTO nodes VALUES (?, ?)", list(enumerate(nodes)))
    coo = A.tocoo()
    cur.executemany(
        "INSERT INTO adjacency VALUES (?, ?, ?)",
        zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()),
    )
    con.commit()
    con.close()
    print(f"wrote adjacency database -> {OUT_DB}")


if __name__ == "__main__":
    main()
