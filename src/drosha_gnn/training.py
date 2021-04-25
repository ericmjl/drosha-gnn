from typing import Callable

import pandas as pd
from jax import jit
from jax import numpy as np
from jax import random, value_and_grad, vmap
from jax.experimental.optimizers import adam
from jax.tree_util import Partial
from loguru import logger
from tqdm.auto import tqdm


def train_test_split(rng: random.PRNGKey, df: pd.DataFrame, train_fraction=0.7):
    """This function guarantees that we always have the same train/test split."""
    indices = np.array(df.index)
    indices = random.permutation(rng, indices)
    num_train = int(len(df) * train_fraction)
    train_idxs = indices[:num_train]
    test_idxs = indices[num_train:]
    logger.info(f"First 10 entries of training set idxs: {train_idxs[:10]}")
    logger.info(f"First 10 entries of test set idxs: {test_idxs[:10]}")
    return train_idxs, test_idxs


def mse(y_true: np.array, y_pred: np.array):
    return np.mean(np.power(y_true.squeeze() - y_pred.squeeze(), 2))


@jit
def mseloss(params, model, X, y):
    """MSE loss."""
    y_pred = vmap(Partial(model, params))(X)
    return mse(y, y_pred)


dmseloss = jit(value_and_grad(mseloss))


def step(
    i,
    state,
    model: Callable,
    X,
    y,
    dmseloss: Callable,
    get_params: Callable,
    update: Callable,
):
    """Full step function.

    - X, y: Our data.
    - dmseloss: Gradient of loss function.
    - get_params, update: get_params and update function from optimizer triple.
    """
    params = get_params(state)
    model = Partial(model)
    l, g = dmseloss(params, model, X, y)
    state = update(i, g, state)
    return (l, state)


def fit(model_func, params, X_train, y_train, num_iters=200, step_size=5e-3):
    opt_init, opt_update, opt_get_params = adam(step_size)
    state = opt_init(params)
    opt_get_params = jit(opt_get_params)
    model_func = jit(model_func)

    adam_step = Partial(
        step,
        model=model_func,
        get_params=opt_get_params,
        update=opt_update,
        dmseloss=dmseloss,
    )
    adam_step = jit(adam_step)

    losses_train = []
    iterator = tqdm(range(num_iters), desc="Training Iteration")
    states = []
    try:
        for i in iterator:
            l_train, state = adam_step(i, state, X=X_train, y=y_train)
            losses_train.append(l_train)
            states.append(state)
    except KeyboardInterrupt:
        pass
    return losses_train, states, opt_get_params


def states_losses(
    states, model: Callable, X, y, get_params: Callable, lossfunc: Callable
):
    """Return all test losses across all parameter states from a training run.

    Intended to get back test losses, but can also be used to reconstruct training losses.
    """
    lossfunc = jit(lossfunc)
    get_params = jit(get_params)
    model = Partial(model)

    losses = []
    for state in tqdm(states, desc=""):
        param = get_params(state)
        losses.append(lossfunc(param, model, X, y))
    return np.stack(losses)


def best_params(states, model, X, y, get_params, lossfunc):
    """Return the best parameter (that minimizes a loss) from training run."""
    test_losses = states_losses(states, model, X, y, get_params, lossfunc)
    best_idx = np.argmin(test_losses)
    logger.info(f"Best parameters found at training epoch {best_idx}")
    best_param = get_params(states[best_idx])
    return best_param, best_idx
