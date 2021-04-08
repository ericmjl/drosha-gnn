import jax.numpy as np
from jax import random


def select_feats(graph_mat, num_nodes: int):
    return graph_mat[:, num_nodes:]

def select_adj(graph_mat, num_nodes: int):
    return graph_mat[:, :num_nodes]

def RnaGraphEmbedding(num_nodes: int, embedding_size: int):
    vocab_size = 4
    def init_fun(rng, input_shape):
        """
        :param input_shape: (num_nodes, num_nodes + num_features)
        """
        num_nodes, num_nodes_features = input_shape
        num_features = num_nodes_features - num_nodes

        embedding_matrix = random.normal(rng, shape=(vocab_size, embedding_size))
        # Add a zeros vector to the beginning for padded vector.
        embedding_matrix = np.concatenate([np.zeros((1, embedding_size)), embedding_matrix])
        return (num_nodes, num_nodes + embedding_size,), embedding_matrix
    
    def apply_fun(params, inputs, **kwargs):
        """
        :param inputs: The node feature matrix.
            We assume that the node feature matrix's first column
            is the embedding index.
        """
        embedding_matrix = params
        adj = select_adj(inputs, num_nodes)
        feats = select_feats(inputs, num_nodes)

        indices = np.take(feats, 0, axis=1).astype(int)
        embedding = np.take(embedding_matrix, indices, axis=0)
        
        output = np.concatenate([adj, embedding], axis=1)
        return output
        
    return init_fun, apply_fun


from jax import lax
def NodeFeatureExtractor(num_nodes: int):
    def init_fun(rng, input_shape):
        num_nodes, num_feats = input_shape
        return (num_nodes, num_nodes + input_shape[-1] - 1,), ()
    
    def apply_fun(params, inputs, **kwargs):
        adj = select_adj(inputs, num_nodes)
        feats = select_feats(inputs, num_nodes)

        return np.concatenate([adj, feats[:, 1:]], axis=1)
    
    return init_fun, apply_fun