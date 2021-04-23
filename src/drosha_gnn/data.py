"""Data handlers."""

import jax.numpy as np
from jax import jit
from .graph import to_networkx
from . import annotate
import networkx as nx


def prep_feats(F, size):
    # F is of shape (n_nodes, n_feats)
    return np.pad(
        F,
        [(0, size - F.shape[0]), (0, 0)],
    )


def prep_adjs(A, size):
    # A is of shape (n_nodes, n_nodes)
    return np.pad(
        A,
        [
            (0, size - A.shape[0]),
            (0, size - A.shape[0]),
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


import pandas as pd
import jax.numpy as np
from jax import random
from typing import Dict
from drosha_gnn.training import train_test_split
from loguru import logger


def split_graph_data(
    data_key: random.PRNGKey, graph_matrices: Dict[int, np.ndarray], df: pd.DataFrame
):
    """Custom train test split to split our graph matrices up."""
    logger.info("Splitting graph data.")
    train_idxs, test_idxs = train_test_split(data_key, df)
    graph_series = pd.Series(graph_matrices)
    X_train = np.stack(graph_series[train_idxs].values)
    X_test = np.stack(graph_series[test_idxs].values)
    y_train = df.loc[train_idxs, "logit"].values.reshape(-1, 1)
    y_test = df.loc[test_idxs, "logit"].values.reshape(-1, 1)
    return X_train, X_test, y_train, y_test


def split_entropy_data(data_key: random.PRNGKey, df, entropy):
    logger.info("Splitting entropy data.")
    train_idxs, test_idxs = train_test_split(data_key, df)
    X_train = entropy.loc[train_idxs]
    X_test = entropy.loc[test_idxs]
    y_train = df.loc[train_idxs, "logit"].values
    y_test = df.loc[test_idxs, "logit"].values
    return X_train, X_test, y_train, y_test


def make_graph(sample_idx, df: pd.DataFrame, entropy: pd.DataFrame):
    """Make graph with nucleotide and entropy annotations."""
    g = to_networkx(df.loc[sample_idx]["dot_bracket"])
    seq = df.loc[sample_idx]["seq"]
    g = annotate.node_nucleotide(g, seq)
    entropy_vec = entropy.loc[sample_idx]
    g = annotate.node_entropy(g, entropy_vec)
    return g


def make_graph_matrices(sample_idx, df, entropy):
    g = make_graph(sample_idx, df, entropy)
    F = prep_feats(feat_matrix(g), 170)
    A = prep_adjs(np.array(nx.adjacency_matrix(g).todense()), 170)
    return np.concatenate([A, F], axis=1)


from scipy import interpolate as interp


def interpolate(array: np.ndarray, length: int):
    """Interpolate a sequence to an array of particular length."""
    ref_array = np.ones(length)
    arr_interp = interp.interp1d(np.arange(array.size), array)
    interpolated = arr_interp(np.linspace(0, array.size - 1, ref_array.size))
    return interpolated


def entropy_sequence(row: pd.Series):
    entropy = np.array([v for v in row if v != -1.0])
    return interpolate(entropy, len(row))


def align_entropy(entropy: pd.DataFrame):
    """Return an entropy dataframe that has entropy interpolated and aligned."""
    entropy_columns = [c for c in entropy.columns if "shannon_" in c]
    entropy_aligned = []
    for row, data in entropy[entropy_columns].iterrows():
        aligned = entropy_sequence(data.values)
        entropy_aligned.append(aligned)
    entropy_aligned_df = pd.DataFrame(np.vstack(entropy_aligned), index=entropy.index)
    entropy_aligned_df.columns = entropy_columns
    return entropy_aligned_df
