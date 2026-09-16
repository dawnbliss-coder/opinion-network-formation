import itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import community as community_louvain  # python-louvain

from data_prep import run_pipeline


MIN_OVERLAP = 5  # minimum jointly-answered statements required to trust a similarity score


# ---------------------------------------------------------------------------
# 1. SIMILARITY METRICS
# ---------------------------------------------------------------------------

def _pairwise(df: pd.DataFrame, pair_fn, min_overlap: int = MIN_OVERLAP) -> pd.DataFrame:
    """
    Generic pairwise-complete similarity computation.
    pair_fn(a, b) -> float, where a, b are 1D arrays with NaNs already removed
    (only positions both respondents answered are passed in).
    """
    ids = df.index.tolist()
    n = len(ids)
    values = df.values
    sim = np.full((n, n), np.nan)

    for i, j in itertools.combinations(range(n), 2):
        vi, vj = values[i], values[j]
        mask = ~np.isnan(vi) & ~np.isnan(vj)
        if mask.sum() < min_overlap:
            continue
        a, b = vi[mask], vj[mask]
        sim[i, j] = sim[j, i] = pair_fn(a, b)

    np.fill_diagonal(sim, 1.0)
    return pd.DataFrame(sim, index=ids, columns=ids)


def cosine_similarity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    def fn(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return np.dot(a, b) / denom if denom > 0 else np.nan
    return _pairwise(df, fn)


def pearson_similarity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    def fn(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return np.corrcoef(a, b)[0, 1]
    return _pairwise(df, fn)


def spearman_similarity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    def fn(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        rho, _ = spearmanr(a, b)
        return rho
    return _pairwise(df, fn)


def agreement_ratio_matrix(df_raw_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Crude, highly interpretable metric: fraction of jointly-answered statements
    where both respondents gave the EXACT SAME encoded Likert value.
    Use response_matrix_raw (not rescaled) for this — rescaling breaks exact equality.
    """
    def fn(a, b):
        return np.mean(a == b)
    return _pairwise(df_raw_labels, fn)


# ---------------------------------------------------------------------------
# 2. SPARSIFICATION STRATEGIES
# ---------------------------------------------------------------------------

def sparsify_threshold(sim: pd.DataFrame, threshold: float) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(sim.index)
    for i, j in itertools.combinations(sim.index, 2):
        w = sim.loc[i, j]
        if pd.notna(w) and w >= threshold:
            G.add_edge(i, j, weight=float(w))
    return G

def sparsify_topk(sim: pd.DataFrame, k: int = 8, mutual: bool = True) -> nx.Graph:
    """
    Top-k nearest neighbours per node. If mutual=True, only keep an edge when
    BOTH respondents list each other in their own top-k (stricter, cleaner graph).
    Recommended default for this dataset since fixed thresholds are unstable here
    (density swings from 88% to 1.6% between threshold=0.5 and threshold=0.9).
    """
    G = nx.Graph()
    G.add_nodes_from(sim.index)

    topk_sets = {}
    for r in sim.index:
        row = sim.loc[r].drop(r).dropna()
        topk_sets[r] = set(row.nlargest(k).index)

    for i, j in itertools.combinations(sim.index, 2):
        w = sim.loc[i, j]
        if pd.isna(w) or w <= 0:
            continue
        if mutual:
            if j in topk_sets[i] and i in topk_sets[j]:
                G.add_edge(i, j, weight=float(w))
        else:
            if j in topk_sets[i] or i in topk_sets[j]:
                G.add_edge(i, j, weight=float(w))
    return G

# ---------------------------------------------------------------------------
# 3. PRIMARY NETWORK
# ---------------------------------------------------------------------------


def sparsify_topk_directed(sim: pd.DataFrame, k: int = 8) -> nx.DiGraph:
    """
    Directed top-k KNN graph: an edge i -> j means "j is one of i's k most
    similar peers", weight = similarity(i, j). This is the more honest encoding
    of top-k similarity, since the relation is NOT inherently symmetric -
    empirically only ~22% of top-8 nominations in this dataset are mutual
    (see reciprocity() below). Forcing it into an undirected graph via OR/AND
    logic either inflates or discards most of this real asymmetric structure.
    """
    D = nx.DiGraph()
    D.add_nodes_from(sim.index)
    for i in sim.index:
        row = sim.loc[i].drop(i).dropna()
        for j, w in row.nlargest(k).items():
            if w > 0:
                D.add_edge(i, j, weight=float(w))
    return D
 
 
def compute_directed_metrics(D: nx.DiGraph) -> dict:
    """
    Directed-graph metrics for the top-k similarity network.
 
    in_strength  : sum of incoming edge weights - how often & how strongly a
                   respondent is nominated as similar by OTHERS. High in-strength
                   = a "consensus"/mainstream opinion holder that many people's
                   profiles resemble.
    out_strength : sum of outgoing edge weights - since every node has exactly
                   k outgoing edges, this measures how similar a respondent's
                   BEST matches are. Low out-strength = an opinion outlier whose
                   closest peers still aren't very close (idiosyncratic profile).
    pagerank     : directed analogue of eigenvector centrality; a respondent is
                   important if nominated by OTHER important respondents, not
                   just by many people.
    reciprocity  : fraction of edges i->j that have a return edge j->i. Low
                   overall reciprocity (as here) explains why forcing mutual
                   edges fragments the graph.
    strongly/weakly connected components: strongly connected requires mutual
    reachability (rare when reciprocity is low); weakly connected ignores
    direction, closer to what the undirected version showed.
    """
    metrics = {}
    metrics["n_nodes"] = D.number_of_nodes()
    metrics["n_edges"] = D.number_of_edges()
 
    metrics["in_strength"] = dict(D.in_degree(weight="weight"))
    metrics["out_strength"] = dict(D.out_degree(weight="weight"))
    metrics["in_degree_centrality"] = nx.in_degree_centrality(D)
    metrics["out_degree_centrality"] = nx.out_degree_centrality(D)
    metrics["pagerank"] = nx.pagerank(D, weight="weight")
    metrics["reciprocity"] = nx.reciprocity(D)
 
    scc = list(nx.strongly_connected_components(D))
    wcc = list(nx.weakly_connected_components(D))
    metrics["n_strongly_connected_components"] = len(scc)
    metrics["largest_scc_size"] = max(map(len, scc), default=0)
    metrics["largest_scc_frac"] = metrics["largest_scc_size"] / D.number_of_nodes() if D.number_of_nodes() else 0
    metrics["n_weakly_connected_components"] = len(wcc)
    metrics["largest_wcc_size"] = max(map(len, wcc), default=0)
    metrics["largest_wcc_frac"] = metrics["largest_wcc_size"] / D.number_of_nodes() if D.number_of_nodes() else 0
    metrics["n_unconnected_nodes"] = D.number_of_nodes() - metrics["largest_wcc_size"]
 
    # Community detection needs an undirected graph. Symmetrize by summing
    # weights in both directions (a mutual pair gets a boosted combined weight,
    # a one-sided nomination keeps its single weight) rather than dropping
    # one-sided edges, so we don't lose the ~78% of structure that's one-sided.
    if community_louvain is not None and D.number_of_edges() > 0:
        undirected_for_community = nx.Graph()
        undirected_for_community.add_nodes_from(D.nodes())
        for u, v, d in D.edges(data=True):
            w = d["weight"]
            if undirected_for_community.has_edge(u, v):
                undirected_for_community[u][v]["weight"] += w
            else:
                undirected_for_community.add_edge(u, v, weight=w)
        partition = community_louvain.best_partition(undirected_for_community, weight="weight")
        metrics["communities"] = partition
        metrics["modularity"] = community_louvain.modularity(
            partition, undirected_for_community, weight="weight"
        )
        metrics["n_communities"] = len(set(partition.values()))
    else:
        metrics["communities"] = {}
        metrics["modularity"] = np.nan
        metrics["n_communities"] = 0
 
    return metrics
 
 
def top_n_directed(metrics: dict, metric_name: str, n: int = 5) -> pd.Series:
    return pd.Series(metrics[metric_name]).sort_values(ascending=False).head(n)
 
 
def plot_directed_user_user(D: nx.DiGraph, metrics: dict, title: str, save_path: str = None):
    """Force-directed layout for the directed graph: node size = in-strength
    (how often nominated as similar), color = community, arrows show direction."""
    fig, ax = plt.subplots(figsize=(11, 11))
    pos = nx.spring_layout(D, weight="weight", seed=42, k=1.5 / np.sqrt(max(D.number_of_nodes(), 1)))
 
    communities = metrics.get("communities", {})
    node_colors = [communities.get(n, 0) for n in D.nodes()]
    in_strength = metrics.get("in_strength", {})
    max_in = max(in_strength.values()) if in_strength else 1
    node_sizes = [200 + 2500 * (in_strength.get(n, 0) / max_in) for n in D.nodes()]
 
    nx.draw_networkx_edges(D, pos, alpha=0.12, arrows=True, arrowsize=6,
                            connectionstyle="arc3,rad=0.05", ax=ax)
    nx.draw_networkx_nodes(D, pos, node_color=node_colors, node_size=node_sizes, cmap="tab20", ax=ax)
    nx.draw_networkx_labels(D, pos, font_size=6, ax=ax)
 
    ax.set_title(
        f"{title}\n"
        f"n={metrics['n_nodes']}, edges={metrics['n_edges']}, "
        f"reciprocity={metrics['reciprocity']:.1%}, "
        f"largest_SCC={metrics['largest_scc_frac']:.1%}, "
        f"communities={metrics['n_communities']} (modularity={metrics['modularity']:.3f})"
    )
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    plt.close(fig)
    return fig


def build_primary_user_user(response_matrix_rescaled: pd.DataFrame, k: int = 8) -> nx.Graph:
    """
    Cosine similarity on std-rescaled vectors + top-k sparsification (mutual).
    """
    sim = cosine_similarity_matrix(response_matrix_rescaled)
    G = sparsify_topk(sim, k=k, mutual=True)
    return G, sim


# ---------------------------------------------------------------------------
# 4. CONTRAST
# ---------------------------------------------------------------------------

def build_signed_user_user(
    response_matrix_rescaled: pd.DataFrame, pos_threshold: float = 0.6, neg_threshold: float = -0.2
) -> nx.Graph:
    """
    Signed network: positive edges for similar respondents, NEGATIVE edges for
    systematically dissimilar (opposite-opinion) pairs. Enables structural
    balance analysis and reveals polarization structure that a purely
    positive-edge graph would hide.
    """
    sim = cosine_similarity_matrix(response_matrix_rescaled)
    G = nx.Graph()
    G.add_nodes_from(sim.index)
    for i, j in itertools.combinations(sim.index, 2):
        w = sim.loc[i, j]
        if pd.isna(w):
            continue
        if w >= pos_threshold:
            G.add_edge(i, j, weight=float(w), sign="positive")
        elif w <= neg_threshold:
            G.add_edge(i, j, weight=float(w), sign="negative")
    return G


def build_category_layers(
    response_matrix_rescaled: pd.DataFrame, statement_metadata: pd.DataFrame, k: int = 8
) -> dict:
    """
    Multiplex layers: one User-User network PER category (T/E/S/V), same node
    set, different edge sets. Lets you test whether opinion clusters on one
    topic (e.g. Technology) predict clustering on another (e.g. Environment).
    Returns {category: (Graph, similarity_matrix)}.
    """
    layers = {}
    for cat in sorted(statement_metadata["category"].unique()):
        cols = statement_metadata.index[statement_metadata["category"] == cat]
        cols = [c for c in cols if c in response_matrix_rescaled.columns]
        sub = response_matrix_rescaled[cols]
        sim = cosine_similarity_matrix(sub)
        G = sparsify_topk(sim, k=k, mutual=False)
        layers[cat] = (G, sim)
    return layers


def build_engagement_weighted(
    sim: pd.DataFrame, engagement_scores: pd.Series, k: int = 8
) -> nx.Graph:
    """
    Reweight similarity by min(engagement_i, engagement_j) before sparsifying,
    so a similarity score computed from a partially-engaged respondent counts
    for less than one computed from a fully-engaged respondent.
    """
    adjusted = sim.copy()
    ids = sim.index
    for i, j in itertools.combinations(ids, 2):
        w = sim.loc[i, j]
        if pd.notna(w):
            factor = min(engagement_scores.get(i, 0), engagement_scores.get(j, 0))
            adjusted.loc[i, j] = adjusted.loc[j, i] = w * factor
    return sparsify_topk(adjusted, k=k, mutual=False)

# ---------------------------------------------------------------------------
# 5. METRICS
# ---------------------------------------------------------------------------

def compute_metrics(G: nx.Graph) -> dict:
    """Compute the standard set of User-User network metrics for the report."""
    metrics = {}

    metrics["n_nodes"] = G.number_of_nodes()
    metrics["n_edges"] = G.number_of_edges()
    n_possible = G.number_of_nodes() * (G.number_of_nodes() - 1) / 2
    metrics["density"] = G.number_of_edges() / n_possible if n_possible else 0

    metrics["degree_centrality"] = nx.degree_centrality(G)
    try:
        metrics["eigenvector_centrality"] = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        metrics["eigenvector_centrality"] = {n: np.nan for n in G.nodes()}
    metrics["betweenness_centrality"] = nx.betweenness_centrality(G, weight="weight")
    metrics["clustering_coefficient"] = nx.clustering(G, weight="weight")
    metrics["avg_clustering"] = nx.average_clustering(G, weight="weight")

    has_negative_weight = any(
        data.get("weight", 1) < 0 for _, _, data in G.edges(data=True)
    )
    if community_louvain is not None and G.number_of_edges() > 0 and not has_negative_weight:
        partition = community_louvain.best_partition(G, weight="weight")
        metrics["communities"] = partition
        metrics["modularity"] = community_louvain.modularity(partition, G, weight="weight")
        metrics["n_communities"] = len(set(partition.values()))
    else:
        metrics["communities"] = {}
        metrics["modularity"] = np.nan
        metrics["n_communities"] = 0

    # connectivity summary
    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        metrics["largest_component_size"] = len(largest_cc)
        metrics["largest_component_frac"] = len(largest_cc) / G.number_of_nodes()
    else:
        metrics["largest_component_size"] = 0
        metrics["largest_component_frac"] = 0
    metrics["n_unconnected_nodes"] = G.number_of_nodes() - metrics["largest_component_size"]

    return metrics


def top_n_by_metric(metrics: dict, metric_name: str, n: int = 5) -> pd.Series:
    """Convenience: top-n respondents by a given centrality metric."""
    return pd.Series(metrics[metric_name]).sort_values(ascending=False).head(n)


# ---------------------------------------------------------------------------
# 6. VISUALIZATION
# ---------------------------------------------------------------------------

def plot_user_user_network(G: nx.Graph, metrics: dict, title: str, save_path: str = None):
    """Force-directed layout, nodes colored by Louvain community, sized by degree centrality."""
    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(G, weight="weight", seed=42, k=1.5 / np.sqrt(max(G.number_of_nodes(), 1)))

    communities = metrics.get("communities", {})
    node_colors = [communities.get(n, 0) for n in G.nodes()]
    degree_cent = metrics.get("degree_centrality", {})
    node_sizes = [300 + 3000 * degree_cent.get(n, 0) for n in G.nodes()]

    edges = G.edges(data=True)
    edge_weights = [d.get("weight", 0.5) for _, _, d in edges]

    nx.draw_networkx_edges(G, pos, alpha=0.15, width=[max(w, 0.1) for w in edge_weights], ax=ax)
    nodes = nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes, cmap="tab20", ax=ax
    )
    nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)

    ax.set_title(
        f"{title}\n"
        f"n={metrics['n_nodes']}, edges={metrics['n_edges']}, "
        f"density={metrics['density']:.1%}, modularity={metrics['modularity']:.3f}, "
        f"communities={metrics['n_communities']}"
    )
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# MAIN: build every variant against the real dataset and compare
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    artifacts = run_pipeline("Survey_Results_UC.csv")
    raw = artifacts["response_matrix_raw"]
    rescaled = artifacts["response_matrix_rescaled"]
    meta = artifacts["statement_metadata"]
    engagement = artifacts["engagement_scores"].reindex(raw.index)

    print("\n=== PRIMARY NETWORK: cosine + top-k(8, mutual) on rescaled data ===")
    G_primary, sim_cosine = build_primary_user_user(rescaled, k=8)
    m_primary = compute_metrics(G_primary)
    print({k: v for k, v in m_primary.items() if k not in
           ("degree_centrality", "eigenvector_centrality", "betweenness_centrality",
            "clustering_coefficient", "communities")})
    print("Top 5 by eigenvector centrality (most 'representative' respondents):")
    print(top_n_by_metric(m_primary, "eigenvector_centrality"))
    print(f"Largest connected component: {m_primary['largest_component_size']} nodes; "
          f"unconnected nodes: {m_primary['n_unconnected_nodes']}")
    plot_user_user_network(G_primary, m_primary, "User-User Network (Primary: Cosine + Mutual Top-8)",
                            save_path="figures/user_user_mutual.png")

    print("\n=== DIRECTED PRIMARY NETWORK: cosine + directed top-k(8) ===")
    D = sparsify_topk_directed(sim_cosine, k=8)
    m_directed = compute_directed_metrics(D)
    print({k: v for k, v in m_directed.items() if k not in
           ("in_strength", "out_strength", "in_degree_centrality",
            "out_degree_centrality", "pagerank", "communities")})
    print("Top 5 by in-strength (most nominated as similar -> 'consensus' respondents):")
    print(top_n_directed(m_directed, "in_strength"))
    print("Top 5 by PageRank (nominated by other well-connected respondents):")
    print(top_n_directed(m_directed, "pagerank"))
    print("Bottom 5 by out-strength (opinion outliers - even their best matches aren't close):")
    print(pd.Series(m_directed["out_strength"]).sort_values().head(5))
    print(f"Largest weakly connected component: {m_directed['largest_wcc_size']} nodes; "
          f"unconnected nodes: {m_directed['n_unconnected_nodes']}")
    plot_directed_user_user(D, m_directed, "User-User Network (Directed Top-8)",
                             save_path="figures/user_user_directed.png")


    print("\n=== CONTRAST 1: Signed network (agreement + disagreement) ===")
    G_signed = build_signed_user_user(rescaled, pos_threshold=0.9, neg_threshold=-0.1)
    m_signed = compute_metrics(G_signed)
    pos_edges = sum(1 for _, _, d in G_signed.edges(data=True) if d["sign"] == "positive")
    neg_edges = sum(1 for _, _, d in G_signed.edges(data=True) if d["sign"] == "negative")
    print(f"Positive edges: {pos_edges}, Negative edges: {neg_edges}; "
          f"largest connected component: {m_signed['largest_component_size']} nodes; "
          f"unconnected nodes: {m_signed['n_unconnected_nodes']}")

    print("\n=== CONTRAST 2: Per-category multiplex layers ===")
    layers = build_category_layers(rescaled, meta, k=8)
    for cat, (G_layer, _) in layers.items():
        m_layer = compute_metrics(G_layer)
        print(f"  Category {cat}: {m_layer['n_edges']} edges, "
              f"modularity={m_layer['modularity']:.3f}, communities={m_layer['n_communities']}, "
              f"largest connected component={m_layer['largest_component_size']} nodes, "
              f"unconnected nodes={m_layer['n_unconnected_nodes']}")

    print("\n=== CONTRAST 3: Engagement-weighted network ===")
    G_engagement = build_engagement_weighted(sim_cosine, engagement, k=8)
    m_engagement = compute_metrics(G_engagement)
    print(f"  {m_engagement['n_edges']} edges, modularity={m_engagement['modularity']:.3f}, "
          f"largest connected component={m_engagement['largest_component_size']} nodes, "
          f"unconnected nodes={m_engagement['n_unconnected_nodes']}")

    for name, sim_fn, data_input in [
        ("pearson", pearson_similarity_matrix, rescaled),
        ("spearman", spearman_similarity_matrix, rescaled),
        ("agreement_ratio", agreement_ratio_matrix, raw),
    ]:
        sim_alt = sim_fn(data_input)
        G_alt = sparsify_topk(sim_alt, k=8, mutual=False)
        m_alt = compute_metrics(G_alt)
        print(f"  {name}: {m_alt['n_edges']} edges, modularity={m_alt['modularity']:.3f}, "
            f"avg_clustering={m_alt['avg_clustering']:.3f}, "
            f"largest connected component={m_alt['largest_component_size']} nodes, "
            f"unconnected nodes={m_alt['n_unconnected_nodes']}")