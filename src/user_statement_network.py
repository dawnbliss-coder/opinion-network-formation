"""
user_statement_network.py

Builds and analyzes the User-Statement opinion network:
a bipartite graph where one set of nodes is respondents and the other
is survey statements. An edge exists whenever a respondent answered a
statement, weighted by their encoded Likert response:

    +2  Strongly Agree
    +1  Agree
     0  Neutral
    -1  Disagree
    -2  Strongly Disagree

Unlike the User-User (cosine similarity) and Statement-Statement
(correlation) networks, this graph keeps the raw response value as the
edge weight instead of a derived similarity/correlation score, so it
is the most direct representation of "who said what" in the dataset.

Expects the artifacts produced by data_prep.run_pipeline():
    - response_matrix_raw: per-respondent encoded response matrix (-2..2),
      NaN for unanswered/"No Comments", already engagement-filtered
    - statement_metadata: statement_id -> category, full_text

Typical usage:
    from data_prep import run_pipeline
    from user_statement_network import (
        build_bipartite_graph, compute_user_metrics, compute_statement_metrics,
        network_summary, print_report, plot_heatmap, plot_network,
    )

    artifacts = run_pipeline("Survey_Results_UC.csv")
    G = build_bipartite_graph(artifacts["response_matrix_raw"], artifacts["statement_metadata"])
    user_metrics = compute_user_metrics(artifacts["response_matrix_raw"])
    statement_metrics = compute_statement_metrics(artifacts["response_matrix_raw"], artifacts["statement_metadata"])
    summary = network_summary(G, artifacts["response_matrix_raw"])
    print_report(summary, user_metrics, statement_metrics)
    plot_heatmap(artifacts["response_matrix_raw"], artifacts["statement_metadata"], user_metrics)
    plot_network(G)
"""

import numpy as np
import pandas as pd
import networkx as nx


USER_PREFIX = "U::"          # namespacing so user IDs can never collide with statement IDs


def _uid(respondent_id) -> str:
    return f"{USER_PREFIX}{respondent_id}"


# ---------------------------------------------------------------------------
# 1. GRAPH CONSTRUCTION
# ---------------------------------------------------------------------------

def build_bipartite_graph(response_matrix_raw: pd.DataFrame, statement_meta: pd.DataFrame) -> nx.Graph:
    """
    Build the User-Statement bipartite graph.

    Nodes: respondents (bipartite=0) and statements (bipartite=1, tagged
    with category + full text).
    Edges: one per answered (respondent, statement) pair, weight = encoded
    Likert value in {-2,-1,0,1,2}. Unanswered / "No Comments" cells (NaN
    in response_matrix_raw) get no edge at all, rather than a weight-0
    edge — a skipped question is not the same as a Neutral response.
    """
    G = nx.Graph()

    for uid in response_matrix_raw.index:
        G.add_node(_uid(uid), bipartite=0, node_type="user")

    for sid, row in statement_meta.iterrows():
        if sid in response_matrix_raw.columns:
            G.add_node(sid, bipartite=1, node_type="statement",
                       category=row["category"], text=row["full_text"])

    values = response_matrix_raw.values
    users = response_matrix_raw.index
    statements = response_matrix_raw.columns
    for i, uid in enumerate(users):
        for j, sid in enumerate(statements):
            w = values[i, j]
            if pd.notna(w):
                G.add_edge(_uid(uid), sid, weight=float(w))

    return G


# ---------------------------------------------------------------------------
# 2. METRICS
# ---------------------------------------------------------------------------

def compute_user_metrics(response_matrix_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Per-respondent metrics, computed directly off the response matrix
    (equivalent to, but simpler than, walking bipartite edges):
      - degree: number of statements answered
      - mean_response: average stance (comparable across respondents
        regardless of how many statements they answered)
      - mean_abs_response: average |weight| ("intensity" - how strongly a
        respondent tends to react, independent of direction)
    """
    degree = response_matrix_raw.notna().sum(axis=1)
    mean_response = response_matrix_raw.sum(axis=1, skipna=True) / degree
    mean_abs_response = response_matrix_raw.abs().mean(axis=1, skipna=True)

    df = pd.DataFrame({
        "degree": degree,
        "mean_response": mean_response,
        "mean_abs_response": mean_abs_response,
    })
    return df.sort_values("mean_response", ascending=False)


def compute_statement_metrics(response_matrix_raw: pd.DataFrame, statement_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Per-statement metrics:
      - degree: number of respondents who answered it
      - mean_response: average encoded response - consensus level AND
        direction (near +2 = broad agreement, near -2 = broad disagreement,
        near 0 = split/neutral)
      - std_response: spread of responses - low std + high |mean| =
        consensus statement; high std = polarizing/contested statement
    """
    degree = response_matrix_raw.notna().sum(axis=0)
    mean_response = response_matrix_raw.mean(axis=0, skipna=True)
    std_response = response_matrix_raw.std(axis=0, skipna=True)

    df = pd.DataFrame({
        "degree": degree,
        "mean_response": mean_response,
        "std_response": std_response,
    })
    df["category"] = statement_meta.reindex(df.index)["category"]
    return df.sort_values("std_response", ascending=False)


def network_summary(G: nx.Graph, response_matrix_raw: pd.DataFrame) -> dict:
    """High-level descriptive stats for the bipartite network."""
    n_users = response_matrix_raw.shape[0]
    n_statements = response_matrix_raw.shape[1]
    n_possible = n_users * n_statements
    n_edges = G.number_of_edges()

    weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])

    return {
        "n_users": n_users,
        "n_statements": n_statements,
        "n_possible_edges": n_possible,
        "n_edges": n_edges,
        "density": n_edges / n_possible if n_possible else 0,
        "mean_weight": float(weights.mean()) if len(weights) else np.nan,
        "pct_positive": float((weights > 0).mean()) if len(weights) else np.nan,
        "pct_negative": float((weights < 0).mean()) if len(weights) else np.nan,
        "pct_neutral": float((weights == 0).mean()) if len(weights) else np.nan,
    }


def edge_threshold_table(response_matrix_raw: pd.DataFrame) -> pd.DataFrame:
    """
    THEORY / REFERENCE ONLY - not used to build the plotted graph, which
    keeps every answered edge (see plot_network()). This exists purely to
    document, with numbers, why sparsifying this particular bipartite graph
    by |weight| would not actually produce a clean network the way the
    Statement-Statement network's |r| >= 0.30 correlation cutoff does.

    Unlike a similarity/correlation network, there is no natural r/cosine
    score to threshold here - the edge weight already IS the raw response -
    so the only conceivable sparsification knob is "how decisive must a
    response be to count as an edge" (i.e. drop Neutral, then drop Agree/
    Disagree too, keeping only Strongly Agree/Disagree). This table shows
    that even the strictest version of that knob barely dents the density:
    respondents answered almost everything, and most answers lean toward
    agreement, so there is no cutoff that turns this into a sparse graph.
    That is precisely why the actual visualization (plot_network) draws
    every edge and instead encodes weight via color/width/opacity, rather
    than filtering edges out as the other two networks do.
    """
    vals = response_matrix_raw.values
    answered = ~np.isnan(vals)
    n_possible = vals.size

    rows = []
    for label, mask in [
        (">=0 (all answered, incl. Neutral)", answered),
        (">=1 (drop Neutral)", answered & (np.abs(vals) >= 1)),
        ("==2 (Strongly Agree/Disagree only)", answered & (np.abs(vals) == 2)),
    ]:
        rows.append({"cutoff": label, "edges": int(mask.sum()), "density": mask.sum() / n_possible})
    return pd.DataFrame(rows)


def print_report(summary: dict, user_metrics: pd.DataFrame, statement_metrics: pd.DataFrame, top_n: int = 5) -> None:
    print("NETWORK SUMMARY")
    print(f"Users (respondents): {summary['n_users']}")
    print(f"Statements: {summary['n_statements']}")
    print(f"Edges (answered questions): {summary['n_edges']} / {summary['n_possible_edges']} possible "
          f"(density {summary['density']:.1%})")
    print(f"Mean edge weight: {summary['mean_weight']:.3f}")
    print(f"Sign split: {summary['pct_positive']:.1%} positive, "
          f"{summary['pct_neutral']:.1%} neutral, {summary['pct_negative']:.1%} negative")
    print()

    print(f"TOP {top_n} MOST AGREEABLE USERS (highest mean response)")
    print(user_metrics.sort_values("mean_response", ascending=False).head(top_n))
    print()
    print(f"TOP {top_n} MOST DISAGREEABLE USERS (lowest mean response)")
    print(user_metrics.sort_values("mean_response").head(top_n))
    print()
    print(f"TOP {top_n} MOST INTENSE USERS (highest mean |response|)")
    print(user_metrics.sort_values("mean_abs_response", ascending=False).head(top_n))
    print()

    print(f"TOP {top_n} MOST POLARIZING STATEMENTS (highest std)")
    print(statement_metrics.sort_values("std_response", ascending=False).head(top_n)[
        ["category", "mean_response", "std_response", "degree"]])
    print()
    print(f"TOP {top_n} MOST CONSENSUS STATEMENTS (lowest std)")
    print(statement_metrics.sort_values("std_response").head(top_n)[
        ["category", "mean_response", "std_response", "degree"]])
    print()
    print(f"TOP {top_n} MOST AGREED-WITH STATEMENTS (highest mean)")
    print(statement_metrics.sort_values("mean_response", ascending=False).head(top_n)[
        ["category", "mean_response", "std_response"]])
    print()
    print(f"TOP {top_n} MOST DISAGREED-WITH STATEMENTS (lowest mean)")
    print(statement_metrics.sort_values("mean_response").head(top_n)[
        ["category", "mean_response", "std_response"]])


# ---------------------------------------------------------------------------
# 3. VISUALIZATION
# ---------------------------------------------------------------------------

def plot_heatmap(response_matrix_raw: pd.DataFrame, statement_meta: pd.DataFrame,
                  user_metrics: pd.DataFrame, figsize=(16, 12)):
    """
    Users x Statements heatmap - the most legible view of a dense bipartite
    network with ~87 x 60 possible edges (too many to draw as a node-link
    graph without it turning into a hairball).

    Rows (users) sorted by mean_response (most disagreeable -> most
    agreeable), columns (statements) grouped by T/E/S/V category, so
    both per-user lean and per-statement consensus are visible at a glance.

    There are only 5 possible response values (-2..+2) plus "missing".
    imshow's default interpolation blends colors across cell boundaries
    when the 87x60 data grid is scaled up to a much higher-resolution
    image, which fabricates in-between shades that aren't real data -
    interpolation="nearest" is used to keep exactly 5 flat colors. Missing
    responses are rendered as a distinct mid-gray (set_bad) rather than
    left blank/white, since white is also the Neutral (0) color on the
    RdBu scale and the two must not be visually confused.
    """
    import matplotlib.pyplot as plt

    row_order = user_metrics.sort_values("mean_response").index
    col_order = statement_meta.sort_values("category").index
    col_order = [c for c in col_order if c in response_matrix_raw.columns]

    mat = response_matrix_raw.loc[row_order, col_order]

    fig, ax = plt.subplots(figsize=figsize)
    masked = np.ma.masked_invalid(mat.values)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad(color="#999999")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-2, vmax=2, interpolation="nearest")

    # category separators + labels along the top
    categories = statement_meta.loc[col_order, "category"]
    boundaries = np.where(categories.values[:-1] != categories.values[1:])[0] + 1
    for b in boundaries:
        ax.axvline(b - 0.5, color="black", linewidth=0.8)

    cat_ticks, cat_labels = [], []
    start = 0
    for i in range(1, len(categories) + 1):
        if i == len(categories) or categories.values[i] != categories.values[start]:
            cat_ticks.append((start + i - 1) / 2)
            cat_labels.append(categories.values[start])
            start = i
    ax.set_xticks(cat_ticks)
    ax.set_xticklabels(cat_labels)
    ax.set_xlabel("Statements (grouped by category)")
    ax.set_ylabel("Respondents (sorted: most disagreeable → most agreeable)")
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, ticks=[-2, -1, 0, 1, 2])
    cbar.ax.set_yticklabels(["-2\nStrongly\nDisagree", "-1\nDisagree", "0\nNeutral",
                             "+1\nAgree", "+2\nStrongly\nAgree"])
    cbar.set_label("Response")

    import matplotlib.patches as mpatches
    missing_patch = mpatches.Patch(color="#999999", label="No response / skipped")
    ax.legend(handles=[missing_patch], loc="upper left", bbox_to_anchor=(1.02, 1.0),
              bbox_transform=ax.transAxes, frameon=True)

    ax.set_title("User-Statement Network: Response Heatmap\n(5 discrete response values + missing, in gray)")
    plt.tight_layout()
    return fig


def plot_network(G: nx.Graph, figsize=(14, 14)):
    """
    Node-link view of the full bipartite network - every answered
    (respondent, statement) edge is drawn, including Neutral (weight 0)
    responses, not just the strong-opinion ones.

    Edge color encodes sign (green = Agree/Strongly Agree, red = Disagree/
    Strongly Disagree, gray = Neutral) and alpha/width scale with |weight|,
    so the 5 response types stay visually distinguishable even though all
    ~5,158 edges are drawn at once.

    Node color: statements colored by category, users colored gray.
    Node size: degree (number of answered statements / respondents).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    pos = nx.spring_layout(G, weight="weight", seed=42, k=0.4)

    categories = sorted(set(nx.get_node_attributes(G, "category").values()))
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(categories), 1))
    cat_color = {cat: cmap(i) for i, cat in enumerate(categories)}

    node_colors, node_sizes = [], []
    for n, d in G.nodes(data=True):
        if d.get("node_type") == "statement":
            node_colors.append(cat_color[d["category"]])
            node_sizes.append(80 + 4 * G.degree(n))
        else:
            node_colors.append("lightgray")
            node_sizes.append(20 + 2 * G.degree(n))

    edges = list(G.edges(data=True))

    def _edge_color(w):
        if w > 0:
            return "#2ca02c"
        if w < 0:
            return "#d62728"
        return "#999999"

    edge_colors = [_edge_color(d["weight"]) for _, _, d in edges]
    edge_widths = [0.2 + 0.5 * abs(d["weight"]) for _, _, d in edges]
    edge_alphas = [0.03 + 0.12 * abs(d["weight"]) for _, _, d in edges]

    stmt_labels = {n: n for n, d in G.nodes(data=True) if d.get("node_type") == "statement"}

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, _ in edges],
                            edge_color=edge_colors, width=edge_widths, alpha=edge_alphas, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                            linewidths=0.3, edgecolors="black", ax=ax)
    nx.draw_networkx_labels(G, pos, labels=stmt_labels, font_size=7, ax=ax)

    legend_handles = [mpatches.Patch(color=cat_color[cat], label=cat) for cat in categories]
    legend_handles.append(mpatches.Patch(color="lightgray", label="user"))
    legend_handles.append(mpatches.Patch(color="#2ca02c", label="Agree / Strongly Agree edge"))
    legend_handles.append(mpatches.Patch(color="#999999", label="Neutral edge"))
    legend_handles.append(mpatches.Patch(color="#d62728", label="Disagree / Strongly Disagree edge"))
    ax.legend(handles=legend_handles, title="Node type / category / edge sign",
              loc="upper left", frameon=True)

    ax.set_title(
        f"User-Statement Network (all response types)\n"
        f"n={G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
        f"(edge width/opacity scale with |response|)"
    )
    ax.axis("off")
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    from data_prep import run_pipeline

    artifacts = run_pipeline("../Survey_Results_UC.csv")
    raw = artifacts["response_matrix_raw"]
    meta = artifacts["statement_metadata"]

    G = build_bipartite_graph(raw, meta)
    user_metrics = compute_user_metrics(raw)
    statement_metrics = compute_statement_metrics(raw, meta)
    summary = network_summary(G, raw)
    print_report(summary, user_metrics, statement_metrics)

    print("DENSITY AT INCREASING |WEIGHT| CUTOFFS (reference only - the plotted graph below keeps every edge)")
    print(edge_threshold_table(raw).to_string(index=False))
    print()

    fig1 = plot_heatmap(raw, meta, user_metrics)
    fig1.savefig("../figures/user_statement_heatmap.png", dpi=150)

    fig2 = plot_network(G)
    fig2.savefig("../figures/user_statement_network.png", dpi=150)
