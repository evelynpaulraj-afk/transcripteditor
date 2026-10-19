import networkx as nx


def build_visual_graph(shots):
    """
    Build a graph where each shot is a node.

    Edges are added when shots share visual concepts.
    """

    G = nx.Graph()

    for shot in shots:

        shot_id = shot["shot_id"]

        analysis = shot.get("analysis", {})

        concepts = set()

        for key in [
            "people",
            "objects",
            "actions",
            "visual_content",
            "screen_or_slide_content"
        ]:

            concepts.update(analysis.get(key, []))

        G.add_node(
            shot_id,
            shot=shot,
            concepts=concepts
        )

    ids = list(G.nodes)

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):

            a = ids[i]
            b = ids[j]

            ca = G.nodes[a]["concepts"]
            cb = G.nodes[b]["concepts"]

            overlap = len(ca & cb)

            if overlap > 0:
                G.add_edge(
                    a,
                    b,
                    weight=overlap
                )

    return G


def compress_visual_graph(shots, keep_ratio=0.5):
    """
    Compress visual context using graph centrality.

    Returns the most important shots.
    """

    if len(shots) <= 3:
        return shots

    G = build_visual_graph(shots)

    scores = nx.degree_centrality(G)

    ranked = sorted(
        G.nodes,
        key=lambda n: scores[n],
        reverse=True
    )

    keep = max(
        3,
        int(len(ranked) * keep_ratio)
    )

    selected = ranked[:keep]

    selected_shots = []

    for sid in selected:
        selected_shots.append(
            G.nodes[sid]["shot"]
        )

    return sorted(
        selected_shots,
        key=lambda s: s["start"]
    )