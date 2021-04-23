"""Data handlers."""

import jax.numpy as np
from jax import jit

def prep_feats(F, size):
    # F is of shape (n_nodes, n_feats)
    return np.pad(
        F,
        [
            (0, size - F.shape[0]),
            (0, 0)
        ],
    )


def prep_adjs(A, size):
    # A is of shape (n_nodes, n_nodes)
    return np.pad(
        A,
        [
            (0, size-A.shape[0]),
            (0, size-A.shape[0]),
        ],
    )


def feat_matrix(G):
    """Return feature matrix of a graph.
    
    - `nucleotide_idx` is nothing more than an indexer for node embeddings.
    - `entropy` is the entropy of the node itself.
    """
    feats = []
    for n, d in G.nodes(data=True):
        feat_vect = np.array([d["nucleotide_idx"], d["entropy"]])
        feats.append(feat_vect)
    feats = np.stack(feats)
    return feats
