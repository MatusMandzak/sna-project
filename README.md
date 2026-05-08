# SNA Project 13 — Twitter COVID-19 Network Analysis

Implementation of *Project 13: Twitter COVID19 Network Analysis* on the
[TweetsCOV19](https://zenodo.org/record/4593591) corpus, restricted to
October–November 2019.

## Layout

```
project_task.txt                   # original assignment text
TweetsCOV19.tsv.gz                 # raw dataset (input)
src/
  preprocess.py                    # filter to Oct-Nov 2019 (Task 1)
  build_graph.py                   # build hashtag co-occurrence graph + adjacency DB (Task 2)
  analysis.py                      # Tasks 2–9 (figures + tables)
  polish_tables.py                 # post-format LaTeX tables for the report
output/
  tweets_oct_nov_2019.parquet      # filtered tweets
  graph.gpickle                    # pickled NetworkX graph
  adjacency.sqlite                 # adjacency matrix as SQLite
  results.json                     # JSON summary of all numeric results
  figures/                         # PDF figures referenced from the report
  tables/                          # CSV + .tex tables referenced from the report
report/
  IEEEtran.cls
  main.tex                         # IEEE-format report
```

## Reproducing

```bash
uv sync                            # install dependencies via pyproject.toml
uv run python src/preprocess.py    # ~1 minute
uv run python src/build_graph.py   # ~1 minute
uv run python src/analysis.py      # ~10 minutes
uv run python src/polish_tables.py # < 1 second
cd report && pdflatex main.tex && pdflatex main.tex   # twice for refs
```

## Methodology highlights

* **Edge cap.** A hashtag with more than 150 distinct users is dropped before
  edge generation. Without this cap a handful of viral tags
  (`#china`, `#healthcare`, `#hongkong`) produce ~39 M
  star-cliques that would dominate the graph without adding social signal.
* **Approximate path-length and diameter.** Computed by sampling source
  nodes and running single-source BFS, with sample sizes set so the
  pipeline finishes in roughly 10 minutes.
* **Power-law fit.** `powerlaw.Fit` (Clauset–Shalizi–Newman MLE) plus a
  log–log regression in the tail to provide the requested $R^2$.
* **SIS calibration.** Sweeps $\beta \in \{0.001,\dots,0.05\}$ and
  $\lambda \in \{0.05,\dots,0.4\}$ on the LCC of the first-increment
  subgraph and ranks candidates by closeness of cumulative infection to
  cumulative observed negative sentiment.
