import networkx as nx
import numpy as np


def is_effectively_undirected(
    G: nx.MultiDiGraph,
    weight: str = "length",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    verbose: bool = True,
) -> bool:
    """
    Check if a directed MultiDiGraph is effectively undirected for a given weight.

    Conditions:
      - For every edge (u, v, k) there exists at least one reverse edge (v, u, k2).
      - At least one reverse edge has a matching `weight` value within tolerance.

    Returns:
      True if the graph behaves like an undirected graph for that weight, False otherwise.
    """
    for u, v, k, data in G.edges(keys=True, data=True):
        # Skip self-loops; they are symmetric by definition
        if u == v:
            continue

        if not G.has_edge(v, u):
            if verbose:
                print(f"Missing reverse edge for ({u}, {v}, {k})")
            return False

        w = data.get(weight)
        # If the forward edge has no weight, we can't compare meaningfully
        if w is None:
            if verbose:
                print(f"Edge ({u}, {v}, {k}) missing weight '{weight}'")
            return False

        # Look for at least one reverse edge with matching weight
        reverse_ok = False
        for k2, data2 in G[v][u].items():
            w2 = data2.get(weight)
            if w2 is None:
                continue
            if np.isclose(w, w2, rtol=rtol, atol=atol):
                reverse_ok = True
                break

        if not reverse_ok:
            if verbose:
                print(f"No reverse edge with matching '{weight}' for ({u}, {v}, {k})")
            return False

    return True
