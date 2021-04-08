from jax import random, numpy as np 
import pandas as pd 


def train_test_split(rng: PRNGKey, df: pd.DataFrame, train_fraction=0.7):
    """This function guarantees that we always have the same train/test split."""
    indices = np.array(df.index)
    indices = random.permutation(rng, indices)
    num_train = int(len(df) * train_fraction)
    train_idxs = indices[:num_train]
    test_idxs = indices[num_train:]
    return train_idxs, test_idxs
