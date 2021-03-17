"""Functions for creating a NetworkX graph representation of a dot-bracket string."""

from collections import Counter


def validate(dot_bracket: str, sequence: str):
    """Validate the dot-bracket notation.

    The checks here are:

    1. The lengths of dot_bracket and sequence are equal.
    2. The number of opening and closing parentheses are equal.
    """

    # Check sequence lengths
    if len(dot_bracket) != len(sequence):
        raise ValueError(
            "The lengths of dot_bracket and sequence are not identical when they should be. \n"
            f"dot_bracket is of length {len(dot_bracket)} "
            f"while sequence is of length {len(sequence)}. \n"
            "Please check your input sequences. "
            "For reference, they are: \n"
            f"- dot_bracket: {dot_bracket} \n"
            f"- sequence: {sequence}"
        )

    # Check parentheses
    letters = Counter(dot_bracket)
    if letters["("] != letters[")"]:
        raise ValueError(
            "The number of opening and closing parentheses are not identical when they should be. \n"
            "For reference, here is what we counted:\n"
            f"{letters}"
        )


def base_pair_edges(dot_bracket: str):
    """Return the base pairing edges of the dot-bracket string."""
    openings = []
    edge_list = []
    for i, l in enumerate(dot_bracket):
        if l == "(":
            openings.append(i)
        if l == ")":
            position = openings.pop(-1)
            edge_list.append((position, i))
    return edge_list


def backbone_edges(dot_bracket: str):
    """Return the RNA backbone edges of the dot-bracket string."""
    num_pos = len(dot_bracket)
    n1 = range(num_pos - 1)
    n2 = range(1, num_pos)
    return list(zip(n1, n2))


import networkx as nx


def to_networkx(dot_bracket: str):
    """Convert the dot-bracket string into a NetworkX graph.

    No metadata is added to the graph, as this should be done afterwards.
    """
    G = nx.Graph()

    G.add_nodes_from(range(len(dot_bracket)))

    G.add_edges_from(base_pair_edges(dot_bracket))
    G.add_edges_from(backbone_edges(dot_bracket))
    return G
