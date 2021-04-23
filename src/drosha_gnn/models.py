from typing import Callable

from jax.experimental import stax

from drosha_gnn.layers import (
    AttentiveGraphSummation,
    AttentiveMessagePassingLayer,
    GetGraphAdjAttentionMatrix,
    GetNodeAttention,
    GraphFanInConcat,
    NodeFeatureExtractor,
    RnaGraphEmbedding,
)


def AttentionEverywhereGNN(num_nodes: int):

    init_fun, apply_fun = stax.serial(
        stax.FanOut(2),
        stax.parallel(
            RnaGraphEmbedding(num_nodes=num_nodes, embedding_size=16),
            NodeFeatureExtractor(num_nodes=num_nodes),
        ),
        GraphFanInConcat(num_nodes=num_nodes),
        AttentiveMessagePassingLayer(num_nodes=num_nodes, hidden_dims=16),
        AttentiveGraphSummation(num_nodes=num_nodes),
        stax.Dense(256),
        stax.Relu,
        stax.Dense(1),
    )

    return init_fun, apply_fun


def GraphAdjacencyAttention(num_nodes: int):
    init_fun, apply_fun = stax.serial(
        stax.FanOut(2),
        stax.parallel(
            RnaGraphEmbedding(num_nodes=num_nodes, embedding_size=16),
            NodeFeatureExtractor(num_nodes=num_nodes),
        ),
        GraphFanInConcat(num_nodes=num_nodes),
        GetGraphAdjAttentionMatrix(num_nodes=num_nodes, hidden_dims=16),
    )
    return init_fun, apply_fun


def GraphNodeAttention(num_nodes: int):
    init_fun, apply_fun = stax.serial(
        stax.FanOut(2),
        stax.parallel(
            RnaGraphEmbedding(num_nodes=num_nodes, embedding_size=16),
            NodeFeatureExtractor(num_nodes=num_nodes),
        ),
        GraphFanInConcat(num_nodes=num_nodes),
        AttentiveMessagePassingLayer(num_nodes=num_nodes, hidden_dims=16),
        GetNodeAttention(num_nodes=num_nodes),
    )
    return init_fun, apply_fun


def make_model_and_params(key, Model: Callable, input_shape: tuple, **model_kwargs):
    init_fun, model = Model(**model_kwargs)
    _, params = init_fun(key, input_shape=input_shape)
    return model, params
