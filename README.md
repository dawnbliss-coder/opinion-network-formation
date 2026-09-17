# Opinion Network Formation

Dataset: survey of 96 respondents across 60 Likert statements in 4 categories — Technology (T), Education (E), Society (S), Environment/Values (V).

## 3. User–Statement Network (`src/user_statement_network.py`)
- **Bipartite graph:** one node set = 87 respondents, other = 60 statements.
- **Edge weight = the raw encoded response itself** (not a derived similarity): `+2` Strongly Agree, `+1` Agree, `0` Neutral, `-1` Disagree, `-2` Strongly Disagree. An edge exists only if the respondent actually answered that statement (a skipped question ≠ a Neutral edge).
- This is the most direct "who said what" representation 
- **Result:** 5,158 / 5,220 possible edges (density 98.8%). Sign split: 80% positive, 13% neutral, 7% negative — the class leans strongly agree overall (mean weight ≈ +1.14).
- Per-user metrics: `mean_response` (agree/disagree lean) and `mean_abs_response` (opinion intensity) identify the most agreeable/disagreeable and most intense respondents.
- Per-statement metrics: `mean_response` + `std_response` separate **consensus statements** (high mean, low std — e.g. E15, V13) from **polarizing statements** (high std — e.g. E04, E03, E02, all Education-category), independent of the Statement–Statement correlation view.
- Two figures:
  - `figures/user_statement_heatmap.png` — full respondent × statement response matrix (respondents sorted by lean, statements grouped by category). Exactly the 5 true response colors + a distinct gray for "no response" are shown (`interpolation="nearest"` + explicit `set_bad` color — plain `imshow` otherwise blends colors across cell borders and fabricates in-between shades that aren't real data).
  - `figures/user_statement_network.png` — node-link view of the **full** bipartite graph, all 5,158 edges, all 5 response types (including Neutral) drawn — nothing filtered out. Edge color encodes sign (green = Agree/Strongly Agree, red = Disagree/Strongly Disagree, gray = Neutral); edge width/opacity scale with `|weight|` so strong opinions stand out from the mass of Agree/Neutral edges without discarding any of them.
- **edge_threshold_table()** :

  | cutoff | edges | density |
  |---|---:|---:|
  | all answered, incl. Neutral | 5,158 | 98.8% |
  | drop Neutral only | 4,491 | 86.0% |
  | Strongly Agree/Disagree only | 2,349 | 45.0% |


## Repo layout
```
src/            data_prep.py, user_network.py, statement_network.py, user_statement_network.py
figures/        all output PNGs
notebooks/      exploration.ipynb (initial EDA)
```
