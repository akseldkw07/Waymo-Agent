import networkx as nx


def post_load(G: nx.MultiDiGraph) -> None:
    """
    Post-process the loaded graph by ensuring 'maxspeed' attributes are integers.

    Args:
        G (nx.MultiDiGraph): The loaded graph.
    Returns:
        nx.MultiDiGraph: The processed graph with cleaned 'maxspeed' attributes.
    """
    for u, v, k, data in G.edges(keys=True, data=True):
        maxspeed = data.get("maxspeed")
        assert isinstance(maxspeed, (int, float, str)), f"Unexpected maxspeed type: {type(maxspeed)}. {data=}"
        if isinstance(maxspeed, str):
            maxspeed = int(maxspeed)

        data["maxspeed"] = maxspeed * 1.60934  # Convert mph to kph
        data["unit"] = "kph"  # Indicate the unit of maxspeed
