# logs_and_audits

Working logs from experiments, kept for transparency. **Numbers across these files are not comparable**: they come from different validation splits, configurations and metric implementations, and the exact provenance of each split was not documented. The authoritative local comparison is `strategy_comparison.csv` (133 validation rows), and the scores that matter are the Zindi leaderboard ones in the main README.

| File | What it is (from its columns) |
|---|---|
| `strategy_comparison.csv` | Strategies A–E per subset; source of the README table (133 rows) |
| `lexical_matching_distributions.csv` | Per-subset validation size, mean blended ROUGE and a retrieval-match flag under one configuration (larger validation sets than above; the flag is not an overlap rate) |
| `agent_run_log.csv` | Per-subset best configuration from an automated optimizer run (BM25 k1/b, copy threshold, normalization switches) |
| `grid_search_alpha_results.csv` | Grid over hybrid weight alpha, k1, b per subset (ROUGE-1) |
| `thorough_grid_search_results.csv` | Grid over retrieval approach, threshold and fallback length per subset |
| `overnight_sweep_matrix.csv` | Embedding model / variant / alpha sweep |
| `cluster_labels_90.csv` | Training-row IDs with the ~90 HDBSCAN cluster labels (labels only, no text) |
| `cluster_labels_eng14.csv` | Same IDs mapped to the 14 clinical groups (G01–G14) and a myth-framing flag |

A per-subset score file that showed ROUGE = 1.0 for every subset was removed: it was a scoring error, not a result.
