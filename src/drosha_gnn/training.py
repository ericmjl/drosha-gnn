from jax import random, numpy as np 
import pandas as pd 
from jax.tree_util import Partial



def train_test_split(rng: random.PRNGKey, df: pd.DataFrame, train_fraction=0.7):
    """This function guarantees that we always have the same train/test split."""
    indices = np.array(df.index)
    indices = random.permutation(rng, indices)
    num_train = int(len(df) * train_fraction)
    train_idxs = indices[:num_train]
    test_idxs = indices[num_train:]
    return train_idxs, test_idxs


from jax import value_and_grad, vmap
from jax import jit

def mse(y_true: np.array, y_pred: np.array):
    return np.mean(np.power(y_true - y_pred, 2))


@jit
def mseloss(params, model, X, y):
    """MSE loss."""
    y_pred = vmap(Partial(model, params))(X)
    return mse(y, y_pred)


dmseloss = jit(value_and_grad(mseloss))
