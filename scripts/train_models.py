"""
This is the master script that trains all models and pickles them down for inspection later.

In this script, we do a number of things
in order to ensure reproducibility and comparability
between model training runs.

1. We use the exact same train/test splits for each model that is used.
The splits are goverened by a PRNGKey.
Each split gets one PRNGKey splitted off from the first PRNGKey.
2. We also fit models for each of the refined mutational data subsets.

Just to recap, the data flow looks roughly like this.

We have the following raw dataframes:

- `combined`: contains the dot-bracket and logit values, which are crucial.
- `entropy`: contains node-wise entropy for each graph.
- `onehot`: contains the one-hot encoding for each RNA sequence.

All of these tables are indexed identically, because they were processed by the same Jupyter notebook
`datasette-formatting.ipynb`.

Graph representations require the `combined` and `entropy` tables.
Simple models (linear regression, random forest) only require the `entropy` table or the `onehot` table.

The outputs are files on disk. (Unfortunately, there's no more flexible database that this one.)
The way things are named are as follows:

```
model-{model_name}_key-{cv_key}_dataset-{dataset}.pkl
```
"""

# Start by setting keys.

from functools import partial

import janitor
import pandas as pd
from drosha_gnn.data import (
    make_graph_matrices,
    split_entropy_data,
    split_graph_data,
    align_entropy,
)
from drosha_gnn.models import AttentionEverywhereGNN, make_model_and_params
from drosha_gnn.training import best_params, fit, mse, mseloss
from jax import random, vmap
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm
import pickle as pkl
from pyprojroot import here
import typer

base_datasets = {
    "biochem": "https://drosha-data.fly.dev/drosha/combined.csv?_stream=on&_sort=rowid&replicate__exact=1&_size=max",
    "mir125": "https://drosha-data.fly.dev/drosha/combined.csv?_labels=on&_stream=on&replicate=2&basename=58&_size=max",
    "mir150": "https://drosha-data.fly.dev/drosha/combined.csv?_labels=on&_stream=on&replicate=2&basename=529&_size=max",
}


def read_data(dataset: str):
    """Return base dataset and entropy table."""

    entropy_url = (
        "https://drosha-data.fly.dev/drosha/entropy.csv?_labels=on&_stream=on&_size=max"
    )

    logger.info("Reading base dataframe")
    df = pd.read_csv(base_datasets[dataset]).set_index("rowid")

    logger.info("Reading entropy dataframe.")
    entropy = (
        pd.read_csv(entropy_url)
        .set_index("rowid")
        .select_columns(["shannon_*"])
        .pipe(align_entropy)
    )
    return df, entropy


def construct_graphs(df, entropy):
    logger.info("Constructing graph matrices.")
    graph_matrices = dict()
    for sample_idx in tqdm(df.index):
        graph_matrices[sample_idx] = make_graph_matrices(sample_idx, df, entropy)
    return graph_matrices


class GATModel:
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


def train_gnn(cv_key, df, entropy, gnn_model, num_iters=200):
    logger.info("Training GNN model.")
    model_and_params_key, data_key = random.split(cv_key)
    graph_matrices = construct_graphs(df, entropy)
    X_train, X_test, y_train, y_test = split_graph_data(data_key, graph_matrices, df)
    train_test_data = X_train, X_test, y_train, y_test
    gnn_model.fit(X_train, y_train, num_iters=num_iters)
    gnn_model.set_best_params(X_test, y_test)
    return gnn_model, train_test_data


def train_entropy(cv_key, df, entropy, model_object):
    logger.info(f"Training a {model_object.__class__.__name__} model on entropy data.")
    _, data_key = random.split(cv_key)

    logger.info(f"Data key value: {data_key}")
    X_train, X_test, y_train, y_test = split_entropy_data(data_key, df, entropy)
    train_test_data = X_train, X_test, y_train, y_test
    model_object.fit(X_train, y_train)
    preds = model_object.predict(X_test)
    score = mse(preds, y_test)
    logger.info(f"Model performance (MSE): {score}")
    return model_object, train_test_data


def pickle_package(model_object, train_test_data, cv_key, dataset_name):
    """Pickle model and data to disk.

    I've chosen to pickle everything together as a single package
    because that makes looking at the data a bit more easy later on.

    As mentioned above, the naming convention for the file is:

    ```
    model-{model_name}_key-{cv_key}_dataset-{dataset}.pkl
    ```

    For the GAT neural network,
    the object that gets saved to disk is the params,
    not the full object.
    """
    package = dict()
    package["data"] = dict()
    X_train, X_test, y_train, y_test = train_test_data
    package["data"]["X_train"] = X_train
    package["data"]["X_test"] = X_test
    package["data"]["y_train"] = y_train
    package["data"]["y_test"] = y_test

    package["model"] = dict()
    package["model"]["name"] = model_object.__class__.__name__
    if isinstance(model_object, GATModel):
        model_object = model_object.best_params
    package["model"]["object"] = model_object

    with open(
        here()
        / "data"
        / "pickles"
        / f"model-{model_object.__class__.__name__}_key-{cv_key}_dataset-{dataset_name}.pkl",
        "wb",
    ) as f:
        pkl.dump(package, f)


def main(dataset_name: str):
    """Main function."""

    # Set up models to be trained
    master_key = random.PRNGKey(99)
    cv_keys = random.split(master_key, 5)
    df, entropy = read_data(dataset_name)
    for cv_key in cv_keys:
        linear_model = LinearRegression()
        rf_model = RandomForestRegressor(n_estimators=300, n_jobs=-1)
        gnn_model = GATModel(cv_key)

        model_object, train_test_data = train_entropy(cv_key, df, entropy, linear_model)
        pickle_package(model_object, train_test_data, cv_key, dataset_name)

        model_object, train_test_data = train_entropy(cv_key, df, entropy, rf_model)
        pickle_package(model_object, train_test_data, cv_key, dataset_name)

        model_object, train_test_data = train_gnn(
            cv_key, df, entropy, gnn_model, num_iters=200
        )
        pickle_package(model_object, train_test_data, cv_key, dataset_name)


if __name__ == "__main__":
    typer.run(main)
