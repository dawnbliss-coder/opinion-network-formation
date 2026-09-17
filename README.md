# Opinion Network Formation

Dataset: survey of 96 respondents across 60 Likert statements in 4 categories — Technology (T), Education (E), Society (S), Environment/Values (V).

**Preprocessing (`src/data_prep.py`, shared by all three networks):**
- Responses encoded to `-2..+2` (Strongly Disagree → Strongly Agree); "No Comments"/blank → `NaN`.
- Respondents who answered <50% of statements are dropped → **87 respondents** retained.
- A per-respondent std-rescaled matrix (divide each row by its own std, no mean-centering) is also produced, to correct for "extreme responders" vs "mild responders" before computing similarity/correlation.

Three complementary networks were built on this data:

## 1. User–User Network (`src/user_network.py`)
- **Nodes:** respondents. **Edges:** cosine similarity between respondents' std-rescaled response vectors.
- Sparsified via **mutual top-8 nearest neighbors** (only kept if each respondent is in the other's top-8 most-similar list) — fixed similarity thresholds were unstable (density swung from 88% to 1.6% between threshold 0.5 and 0.9).
- Also explored: a **directed** top-8 version (only ~22% of nominations are mutual, so direction is informative), a **signed** variant (adds negative edges for opposite-opinion pairs), per-category multiplex layers, and an engagement-weighted variant.
- Figures: `figures/user_user_mutual.png`, `figures/user_user_directed.png`.

## 2. Statement–Statement Network (`src/statement_network.py`)
- **Nodes:** the 60 statements. **Edges:** Spearman correlation between statements across respondents (computed on the rescaled matrix; min 15 overlapping responses required).
- Edge kept if `|r| ≥ 0.30`. At this threshold: **810 edges**, density **45.8%**, avg degree **27**, all edges positive, **4 components** (largest = 57/60 nodes; isolated: E02, E03, E04).
- Most central statements: **S09** (inclusive university environments) — highest degree & eigenvector centrality; **S15** (public trust in tech) — highest betweenness. High-centrality statements span multiple categories (S, V, E, T), i.e. structure cuts across the survey's own thematic labels.
- Full write-up: `src/statement_network_report.md`. Figure: `figures/statement_network.png`.

## 3. User–Statement Network (`src/user_statement_network.py`)
- **Bipartite graph:** one node set = 87 respondents, other = 60 statements.
- **Edge weight = the raw encoded response itself** (not a derived similarity): `+2` Strongly Agree, `+1` Agree, `0` Neutral, `-1` Disagree, `-2` Strongly Disagree. An edge exists only if the respondent actually answered that statement (a skipped question ≠ a Neutral edge).
- This is the most direct "who said what" representation — the other two networks are effectively derived views of this same underlying response matrix (User–User via row similarity, Statement–Statement via column correlation).
- **Result:** 5,158 / 5,220 possible edges (density 98.8%). Sign split: 80% positive, 13% neutral, 7% negative — the class leans strongly agree overall (mean weight ≈ +1.14).
- Per-user metrics: `mean_response` (agree/disagree lean) and `mean_abs_response` (opinion intensity) identify the most agreeable/disagreeable and most intense respondents.
- Per-statement metrics: `mean_response` + `std_response` separate **consensus statements** (high mean, low std — e.g. E15, V13) from **polarizing statements** (high std — e.g. E04, E03, E02, all Education-category), independent of the Statement–Statement correlation view.
- Two figures:
  - `figures/user_statement_heatmap.png` — full respondent × statement response matrix (respondents sorted by lean, statements grouped by category). Exactly the 5 true response colors + a distinct gray for "no response" are shown (`interpolation="nearest"` + explicit `set_bad` color — plain `imshow` otherwise blends colors across cell borders and fabricates in-between shades that aren't real data).
  - `figures/user_statement_network.png` — node-link view of the **full** bipartite graph, all 5,158 edges, all 5 response types (including Neutral) drawn — nothing filtered out. Edge color encodes sign (green = Agree/Strongly Agree, red = Disagree/Strongly Disagree, gray = Neutral); edge width/opacity scale with `|weight|` so strong opinions stand out from the mass of Agree/Neutral edges without discarding any of them.
- **Theory note — why sparsify-by-threshold doesn't make sense here** (`edge_threshold_table()` in the script, reference only, not used to build the plotted graph): unlike the other two networks, there's no similarity/correlation score to threshold — the edge weight already *is* the raw response — so the only conceivable knob is "how decisive must a response be to keep as an edge":

  | cutoff | edges | density |
  |---|---:|---:|
  | all answered, incl. Neutral | 5,158 | 98.8% |
  | drop Neutral only | 4,491 | 86.0% |
  | Strongly Agree/Disagree only | 2,349 | 45.0% |

  Even the strictest cut (Strongly Agree/Disagree only) only reaches 45% density — the same ballpark the Statement–Statement network treated as "selective enough" at its own 0.30 threshold — and that's before accounting for how much real signal (every Agree/Disagree/Neutral response) it would throw away. This is why the actual figure keeps every edge and encodes weight via color/width/opacity instead of filtering.
- Note: the response-weighted scheme (+2/+1/0/-1/-2) is a starting/tentative encoding — could be revisited later (e.g. distance-based weights).

## Repo layout
```
src/            data_prep.py, user_network.py, statement_network.py, user_statement_network.py
figures/        all output PNGs
notebooks/      exploration.ipynb (initial EDA)
```
