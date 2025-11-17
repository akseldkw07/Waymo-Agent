import networkx as nx

from .utils import _max_speed_int


def preprocess_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Preprocesses the graph by cleaning edge attributes."""
    G_cleaned = G.copy()
    _max_speed(G_cleaned)
    return G_cleaned


def _max_speed(G: nx.MultiDiGraph) -> None:
    """Cleans the 'maxspeed' attribute of all edges in the graph by converting them to integers."""
    for u, v, k, data in G.edges(keys=True, data=True):
        maxspeed = data.get("maxspeed")
        if maxspeed is not None:
            data["maxspeed"] = _max_speed_int(maxspeed)
