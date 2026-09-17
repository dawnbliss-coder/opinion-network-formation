"""
statement_network.py

Builds and analyzes the statement-statement opinion network:
nodes = survey statements, edges = similarity in how respondents
answered pairs of statements (correlation-based). Reveals latent
clusters and tensions in class opinion, including whether they
cut across the survey's own T/E/S/V category labels.

Expects the artifacts produced by data_prep.run_pipeline():
    - response_matrix_rescaled: per-respondent std-rescaled response matrix
    - statement_metadata: statement_id -> category, full_text

Typical usage:
    from data_prep import run_pipeline
    from statement_network import (
        build_statement_graph, compute_network_metrics,
        network_summary, print_report, plot_statement_network,
    )

    artifacts = run_pipeline("Survey_Results_UC.csv")
    G, corr = build_statement_graph(
        artifacts["response_matrix_rescaled"],
        artifacts["statement_metadata"],
    )
    metrics = compute_network_metrics(G)
    summary = network_summary(G)
    print_report(G, metrics, summary)
    plot_statement_network(G)
"""

import numpy as np
import pandas as pd
import networkx as nx


CORR_METHOD = "spearman"     # ordinal-appropriate; use "pearson" if you'd rather assume interval spacing
MIN_PERIODS = 15             # minimum overlapping respondents required to trust a correlation
EDGE_THRESHOLD = 0.30        # |r| cutoff for keeping an edge


def compute_correlation_matrix(rescaled: pd.DataFrame, method: str = CORR_METHOD,
                                min_periods: int = MIN_PERIODS) -> pd.DataFrame:
    """
    Pairwise statement-statement correlation across respondents.

    Uses the *rescaled* (per-respondent std-normalized) matrix rather than
    raw encoded values: without that correction, "extreme responders" who
    mark everything Strongly Agree/Disagree inflate correlations between
    otherwise unrelated statements, since their rows push every column pair
    toward the same corners. Rescaling removes that response-style artifact
    before we measure genuine co-variation in opinion.

    pandas' .corr() is pairwise-complete by default (uses only respondents
    who answered both statements in a pair); min_periods guards against
    trusting a correlation built from too few overlapping responses.
    """
    return rescaled.corr(method=method, min_periods=min_periods)


def build_statement_graph(
    rescaled: pd.DataFrame,
    statement_meta: pd.DataFrame,
    method: str = CORR_METHOD,
    min_periods: int = MIN_PERIODS,
    threshold: float = EDGE_THRESHOLD,
):
    """
    Build the statement-statement opinion network.

    Nodes: one per survey statement (T01, E03, ...), tagged with its
    category and full text from statement_meta.
    Edges: weighted by correlation r between how respondents answered
    the two statements; sign is preserved (positive edge = statements
    tend to be endorsed together, negative = opposing opinions). An
    edge is kept only if |r| >= threshold.

    Returns (G, corr) — the networkx graph and the full correlation matrix
    (the matrix is also useful on its own for a heatmap figure).
    """
    corr = compute_correlation_matrix(rescaled, method=method, min_periods=min_periods)

    G = nx.Graph()
    for sid, row in statement_meta.iterrows():
        if sid in corr.columns:
            G.add_node(sid, category=row["category"], text=row["full_text"])

    cols = corr.columns
    for i, s1 in enumerate(cols):
        for s2 in cols[i + 1:]:
            r = corr.loc[s1, s2]
            if pd.notna(r) and abs(r) >= threshold:
                G.add_edge(s1, s2, weight=r, sign="pos" if r > 0 else "neg")

    return G, corr


def compute_network_metrics(G: nx.Graph) -> pd.DataFrame:
    """
    Per-statement metrics for the report's Analysis section:
      - degree (unweighted — number of statements it's meaningfully
        correlated with, not how strongly)
      - pos_degree (unweighted count of positively-correlated neighbors —
        how many other statements this one's responses move *with*
        across respondents; each qualifying edge counts once regardless
        of the correlation's magnitude, to keep pos_degree <= degree
        directly comparable and interpretable)
      - eigenvector_centrality (computed on |weight|, since eigenvector
        centrality requires non-negative weights, where higher weight
        correctly means "more influence")
      - betweenness_centrality (computed on a distance transform of
        |weight|, since nx.betweenness_centrality's `weight` parameter is
        interpreted as a traversal cost, not a strength — a raw r=0.9 edge
        would otherwise be treated as "farther apart" than a weak r=0.31
        edge, which is backwards for a correlation network)
    """
    abs_G = nx.Graph()
    abs_G.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        abs_G.add_edge(u, v, weight=abs(d["weight"]))

    degree = dict(G.degree())

    # pos_degree: how many neighbors a statement's opinions move *with*
    # (positive correlation — co-endorsed or co-rejected together).
    pos_degree = {n: 0 for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        if d["weight"] > 0:
            pos_degree[u] += 1
            pos_degree[v] += 1

    try:
        eigen = nx.eigenvector_centrality(abs_G, weight="weight", max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigen = {n: np.nan for n in G.nodes()}

    # Distance graph for betweenness: strong correlation (|r| close to 1)
    # -> short distance (close to 0); weak correlation (|r| close to 0)
    # -> long distance (close to 1). Shortest paths then correctly favor
    # routing through strongly-correlated statements.
    dist_G = nx.Graph()
    dist_G.add_nodes_from(abs_G.nodes())
    for u, v, d in abs_G.edges(data=True):
        dist_G.add_edge(u, v, distance=1 - d["weight"])
    betweenness = nx.betweenness_centrality(dist_G, weight="distance")

    df = pd.DataFrame({
        "degree": degree,
        "pos_degree": pos_degree,
        "eigenvector_centrality": eigen,
        "betweenness_centrality": betweenness,
    })
    df["category"] = [G.nodes[n]["category"] for n in df.index]
    return df.sort_values("degree", ascending=False)


def network_summary(G: nx.Graph) -> dict:
    """
    High-level descriptive stats to quote directly in the report.

    Also reports connectivity diagnostics (n_connected_components,
    largest_component_size, isolated_nodes) so a given threshold's effect
    on graph fragmentation is visible before you commit to it — at a
    high threshold, statements can lose every edge and silently drop out
    of the "network" part of the analysis.
    """
    n_pos = sum(1 for _, _, d in G.edges(data=True) if d["weight"] > 0)
    n_neg = G.number_of_edges() - n_pos

    components = list(nx.connected_components(G))
    isolated_nodes = sorted(n for n in G.nodes() if G.degree(n) == 0)

    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "n_connected_components": len(components),
        "largest_component_size": max((len(c) for c in components), default=0),
        "isolated_nodes": isolated_nodes,
        "avg_degree": (sum(dict(G.degree()).values()) / G.number_of_nodes()
                       if G.number_of_nodes() else 0),
        "n_positive_edges": n_pos,
        "n_negative_edges": n_neg,
    }


def print_report(G: nx.Graph, metrics: pd.DataFrame, summary: dict, top_n: int = 5) -> None:
    """Print the network summary plus the top-N statements by each metric."""
    print("NETWORK SUMMARY")
    print(f"Nodes: {summary['n_nodes']}")
    print(f"Edges: {summary['n_edges']}")
    print(f"Density: {summary['density']:.4f}")
    print(f"Average degree: {summary['avg_degree']:.2f}")
    print(f"Positive edges: {summary['n_positive_edges']}")
    print(f"Negative edges: {summary['n_negative_edges']}")
    print(f"Connected components: {summary['n_connected_components']}")
    print(f"Largest component size: {summary['largest_component_size']} / {summary['n_nodes']} nodes")
    if summary["isolated_nodes"]:
        print(f"Isolated statements: {', '.join(summary['isolated_nodes'])}")
    else:
        print("Isolated statements: none")
    print()

    def _print_top(column: str, header: str, metric_label: str, fmt: str):
        print(header)
        print(f"Statement | Category | {metric_label} | Text")
        top = metrics.sort_values(column, ascending=False).head(top_n)
        for sid, row in top.iterrows():
            text = G.nodes[sid]["text"]
            print(f"{sid} | {row['category']} | {fmt.format(row[column])} | {text}")
        print()

    _print_top(
        "degree",
        "TOP STATEMENTS BY DEGREE",
        "Degree",
        "{:d}",
    )
    _print_top(
        "eigenvector_centrality",
        "TOP STATEMENTS BY EIGENVECTOR CENTRALITY",
        "Eigenvector",
        "{:.4f}",
    )
    _print_top(
        "betweenness_centrality",
        "TOP STATEMENTS BY BETWEENNESS",
        "Betweenness",
        "{:.4f}",
    )


def plot_statement_network(G: nx.Graph, figsize=(14, 12), seed: int = 42):
    """
    Force-directed visualization:
      - node color: category (T/E/S/V), with a legend
      - node size: degree (more connections = bigger)
      - edge color: sign (green = positive co-endorsement, red = opposing opinions)
      - edge width: |correlation|
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    pos = nx.spring_layout(G, weight="weight", seed=seed, k=0.6)

    degree = dict(G.degree())
    max_deg = max(degree.values(), default=1) or 1
    sizes = [200 + 800 * degree[n] / max_deg for n in G.nodes()]

    categories = sorted(set(nx.get_node_attributes(G, "category").values()))
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(categories), 1))
    cat_color = {cat: cmap(i) for i, cat in enumerate(categories)}
    colors = [cat_color[G.nodes[n]["category"]] for n in G.nodes()]

    edge_colors = ["#2ca02c" if G.edges[e]["weight"] > 0 else "#d62728" for e in G.edges()]
    edge_widths = [1 + 4 * abs(G.edges[e]["weight"]) for e in G.edges()]

    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, edgecolors="black", linewidths=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)

    legend_handles = [mpatches.Patch(color=cat_color[cat], label=cat) for cat in categories]
    plt.legend(handles=legend_handles, title="Category", loc="upper left", frameon=True)

    plt.title("Statement-Statement Opinion Network")
    plt.axis("off")
    plt.tight_layout()
    return plt.gcf()


if __name__ == "__main__":
    from data_prep import run_pipeline

    artifacts = run_pipeline("../Survey_Results_UC.csv")
    G, corr = build_statement_graph(
        artifacts["response_matrix_rescaled"],
        artifacts["statement_metadata"],
    )
    metrics = compute_network_metrics(G)
    summary = network_summary(G)
    print_report(G, metrics, summary)
    fig = plot_statement_network(G)
    fig.savefig("../figures/statement_network.png", dpi=150)