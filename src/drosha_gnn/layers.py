import jax.numpy as np
from jax import random
from jax import nn

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

from jax.tree_util import tree_map
from functools import partial
from drosha_gnn.layers import select_adj, select_feats


def GraphFanInConcat(num_nodes: int, axis: int = -1):
    def init_fun(rng, input_shape):
        
        num_feats = np.sum(np.array([i[1] - num_nodes for i in input_shape]))
        return (num_nodes, num_nodes + num_feats), ()

    def apply_fun(params, inputs, **kwargs):
        adj = tree_map(partial(select_adj, num_nodes=170), inputs)
        feats = tree_map(partial(select_feats, num_nodes=170), inputs)
        feats = np.concatenate(feats, axis=1)
        return np.concatenate([adj[0], feats], axis=1)

    return init_fun, apply_fun



from functools import partial
from jax import vmap

def concat_nodes(node1, node2):
    """Concatenate two nodes together."""
    return np.concatenate([node1, node2])


def concatenate(node: np.ndarray, node_feats: np.ndarray):
    """Concatenate node with each node in node_feats.

    Behaviour is as follows.
    Given a node with features `f_0` and stacked node features
    `[f_0, f_1, f_2, ..., f_N]`,
    return a stacked concatenated feature array:
    `[(f_0, f_0), (f_0, f_1), (f_0, f_2), ..., (f_0, f_N)]`.
    
    :param node: A vector embedding of a single node in the graph.
        Should be of shape (n_input_features,)
    :param node_feats: Stacked vector embedding of all nodes in the graph.
        Should be of shape (n_nodes, n_input_features)
    :returns: A stacked array of concatenated node features.
    """
    return vmap(partial(concat_nodes, node))(node_feats)


def concatenate_node_features(node_feats):
    """Return node-by-node concatenated features.
    
    Given a node feature matrix of shape (n_nodes, n_features),
    this returns a matrix of shape (n_nodes, n_nodes, 2*n_features).
    """
    outputs = vmap(partial(concatenate, node_feats=node_feats))(node_feats)
    return outputs


def attentive_adjacency(params, adj, feats, num_nodes):
    w1, b1, w2, b2, wa = params
    node_by_node_concat = concatenate_node_features(feats)

    # Neural network piece here.
    a1 = np.tanh(np.dot(node_by_node_concat, w1) + b1)
    # a2 = np.dot(a1, w2) + b2
    a2 = nn.relu(np.dot(a1, w2) + b2)
    return a2 * adj

    # Finally, we multiply by a matrix of the same shape
    # to get an attention matrix that doesn't have banding.
    # attentive_adj = adj * a2 * wa
    # return attentive_adj

def AttentiveMessagePassingLayer(num_nodes: int, hidden_dims: int = 256):
    """Attentive message passing on a graph.
    
    We use a feed forward neural network to learn
    the weights on which a message passing operator should work.
    
    The input is the graph matrix. Should be of size (num_nodes, num_nodes + num_feats).
    The output is also of the size (num_nodes, num_nodes + num_feats).
    """

    def init_fun(rng, input_shape):
        num_nodes, n_node_feats = input_shape
        num_feats = n_node_feats - num_nodes
        k1, k2, k3, k4, k5 = random.split(rng, 5)
        
        # Params for neural network transformation of node concatenated features.
        w1 = random.normal(k1, shape=(2 * num_feats, hidden_dims)) * 0.001
        b1 = random.normal(k2, shape=(hidden_dims,)) * 0.001
        w2 = random.normal(k3, shape=(hidden_dims,)) * 0.001
        b2 = random.normal(k4, shape=(1,)) * 0.001
        wa = random.normal(k5, shape=(num_nodes, num_nodes))
        
        params = w1, b1, w2, b2, wa
        output_shape = (num_nodes, num_nodes + num_feats)
        return output_shape, params
    
    def apply_fun(params, inputs, **kwargs):
        ### START ATTENTIVE MATRIX CALCULATION ###
        adj = select_adj(inputs, num_nodes)
        feats = select_feats(inputs, num_nodes)
        attentive_adj = attentive_adjacency(params, adj, feats, num_nodes)
        ### END ATTENTIVE MATRIX CALCULATION ###
        mp = np.dot(attentive_adj, feats)
        return np.concatenate([adj, mp], axis=1)

    return init_fun, apply_fun


def GetGraphAdjAttentionMatrix(num_nodes: int, hidden_dims: int = 256):
    """Return just the attentive matrix.
    
    Intended to be the final layer of a model.
    
    We use a feed forward neural network to learn
    the weights on which a message passing operator should work.
    
    The input is the graph matrix. Should be of size (num_nodes, num_nodes + num_feats).
    The output is also of the size (num_nodes, num_nodes + num_feats).
    """

    def init_fun(rng, input_shape):
        num_nodes, n_node_feats = input_shape
        num_feats = n_node_feats - num_nodes
        k1, k2, k3, k4, k5 = random.split(rng, 5)
        
        # Params for neural network transformation of node concatenated features.
        w1 = random.normal(k1, shape=(2 * num_feats, hidden_dims)) * 0.001
        b1 = random.normal(k2, shape=(hidden_dims,)) * 0.001
        w2 = random.normal(k3, shape=(hidden_dims,)) * 0.001
        b2 = random.normal(k4, shape=(1,)) * 0.001
        wa = random.normal(k5, shape=(num_nodes, num_nodes))
        
        params = w1, b1, w2, b2, wa
        output_shape = (num_nodes, num_nodes + num_feats)
        return output_shape, params

    
    def apply_fun(params, inputs, **kwargs):
        adj = select_adj(inputs, num_nodes)
        feats = select_feats(inputs, num_nodes)
        attentive_adj = attentive_adjacency(params, adj, feats, num_nodes)
        return attentive_adj

    return init_fun, apply_fun


def compute_node_attention_weights(params, feats):
    w1, b1, w2, b2 = params
    a1 = np.tanh(np.dot(feats, w1) + b1)
    a2 = nn.relu(np.dot(a1, w2) + b2)
    return np.squeeze(a2)



def AttentiveGraphSummation(num_nodes, hidden_dims: int = 2048):
    def init_fun(rng, input_shape):
        num_nodes, num_node_feats = input_shape
        num_feats = num_node_feats - num_nodes
        
        k1, k2, k3, k4 = random.split(rng, 4)
        # Params for neural network transformation of node concatenated features.
        w1 = random.normal(k1, shape=(num_feats, hidden_dims)) * 0.001
        b1 = random.normal(k2, shape=(hidden_dims,)) * 0.001
        w2 = random.normal(k3, shape=(hidden_dims, 1)) * 0.001
        b2 = random.normal(k4, shape=(1,)) * 0.001
        params = w1, b1, w2, b2
        output_shape = (num_feats,)
        return output_shape, params
    
    def apply_fun(params, inputs, **kwargs):
        feats = select_feats(inputs, num_nodes)
        
        # Neural network piece here.
        node_attn_weights = compute_node_attention_weights(params, feats)

        # Weighted summation happens here
        out = np.dot(node_attn_weights, feats)
        return out

    return init_fun, apply_fun


def GetNodeAttention(num_nodes, hidden_dims:int = 16):
    def init_fun(rng, input_shape):
        num_nodes, num_node_feats = input_shape
        num_feats = num_node_feats - num_nodes
        
        k1, k2, k3, k4 = random.split(rng, 4)
        # Params for neural network transformation of node concatenated features.
        w1 = random.normal(k1, shape=(num_feats, hidden_dims)) * 0.001
        b1 = random.normal(k2, shape=(hidden_dims,)) * 0.001
        w2 = random.normal(k3, shape=(hidden_dims, 1)) * 0.001
        b2 = random.normal(k4, shape=(1,)) * 0.001
        params = w1, b1, w2, b2
        output_shape = (num_feats,)
        return output_shape, params
    
    def apply_fun(params, inputs, **kwargs):
        feats = select_feats(inputs, num_nodes)
        
        # Neural network piece here.
        node_attn_weights = compute_node_attention_weights(params, feats)
        return node_attn_weights
    return init_fun, apply_fun


