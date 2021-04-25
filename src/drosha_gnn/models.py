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
        stax.Elu,
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


from drosha_gnn.training import fit, best_params, mseloss
from jax import vmap
from functools import partial


class GATModel:
    """sklearn-compatible version of GAT model."""

    def __init__(self, key):
        model, params = make_model_and_params(
            key, AttentionEverywhereGNN, input_shape=(170, 2), num_nodes=170
        )
        self.key = key
        self.model = model
        self.initial_params = params
        self.best_params = None
        self.best_idx = None
        self.loss_history = []
        self.state_history = []
        self.opt_get_params = None

    def fit(self, X, y, num_iters=200):
        losses_train, states, opt_get_params = fit(
            self.model, self.initial_params, X, y, num_iters=num_iters
        )
        self.loss_history = losses_train
        self.state_history = states
        self.opt_get_params = opt_get_params

    def set_best_params(self, X, y):
        """X, y should be the test set, not the training set!"""
        self.best_params, self.best_idx = best_params(
            self.state_history, self.model, X, y, self.opt_get_params, mseloss
        )

    def predict(self, X):
        return vmap(partial(self.model, self.best_params))(X)
