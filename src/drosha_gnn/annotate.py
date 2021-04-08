"""Annotate graph datastructure with metadata."""
import networkx as nx 
import numpy as np 

def node_nucleotide(G: nx.Graph, sequence: str):
    """Annotate nucleotide and nucleotide index on every node."""
    nucleotides = sorted("AUGC")
    for i, letter in enumerate(sequence):
        G.nodes[i]["nucleotide"] = letter
        G.nodes[i]["nucleotide_idx"] = nucleotides.index(letter) + 1
    return G


def node_entropy(G: nx.Graph, entropy_vector: np.ndarray):
    """Annotate entropy on every node."""
    for node, entropy in zip(G.nodes(), entropy_vector):
        G.nodes[node]["entropy"] = float(entropy)
    return G
