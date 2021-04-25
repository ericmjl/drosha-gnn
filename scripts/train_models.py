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
from jax._src.random import PRNGKey
import pandas as pd
from drosha_gnn.data import (
    make_graph_matrices,
    split_entropy_data,
    split_graph_data,
    align_entropy,
)
from drosha_gnn.training import mse
from jax import random
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm
import pickle as pkl
from pyprojroot import here
import typer
from drosha_gnn.models import GATModel


app = typer.Typer()

base_datasets = {
    "biochem": "https://drosha-data.fly.dev/drosha/combined.csv?_stream=on&_sort=rowid&replicate__exact=1&_size=max",
    "mir150": "https://drosha-data.fly.dev/drosha/combined.csv?_labels=on&_stream=on&replicate=2&basename=529&_size=max",
    "mir16": "https://drosha-data.fly.dev/drosha/combined.csv?_labels=on&_stream=on&replicate=2&basename=123&_size=max",
    "mir190": "https://drosha-data.fly.dev/drosha/combined.csv?_labels=on&_stream=on&replicate=2&basename=170&_sort_desc=frac_avg&_size=max",
}


def read_data(dataset: str):
    """Return base dataset and entropy table."""

    entropy_url = (
        "https://drosha-data.fly.dev/drosha/entropy.csv?_labels=on&_stream=on&_size=max"
    )

    logger.info("Reading base dataframe")
    df = pd.read_csv(base_datasets[dataset]).set_index("rowid")

    logger.info("Reading entropy dataframe.")
    entropy = pd.read_csv(entropy_url).set_index("rowid").select_columns(["shannon_*"])
    return df, entropy


def construct_graphs(df, entropy):
    logger.info("Constructing graph matrices.")
    graph_matrices = dict()
    logger.info(f"First 10 values of first row of entropy: {entropy.iloc[0][0:10]}")
    logger.info(f"Last 10 values of first row of entropy: {entropy.iloc[0][-10:]}")
    for sample_idx in tqdm(df.index):
        graph_matrices[sample_idx] = make_graph_matrices(sample_idx, df, entropy)
    return graph_matrices


def train_gnn(
    cv_key: PRNGKey, df: pd.DataFrame, entropy: pd.DataFrame, gnn_model, num_iters=300
):
    logger.info("Training GNN model.")
    model_and_params_key, data_key = random.split(cv_key)
    graph_matrices = construct_graphs(df, entropy)
    X_train, X_test, y_train, y_test = split_graph_data(data_key, graph_matrices, df)
    train_test_data = X_train, X_test, y_train, y_test
    gnn_model.fit(X_train, y_train, num_iters=num_iters)
    gnn_model.set_best_params(X_test, y_test)
    preds = gnn_model.predict(X_test)
    score = mse(preds, y_test)
    logger.info(f"Model performance (MSE): {score}")
    return gnn_model, train_test_data


def train_entropy(
    cv_key: PRNGKey, df: pd.DataFrame, entropy: pd.DataFrame, model_object: GATModel
):
    logger.info(f"Training a {model_object.__class__.__name__} model on entropy data.")
    _, data_key = random.split(cv_key)

    entropy_cleaned = entropy.pipe(align_entropy)
    logger.info(f"Data key value: {data_key}")
    X_train, X_test, y_train, y_test = split_entropy_data(data_key, df, entropy_cleaned)
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


@app.command()
def main(
    dataset_name: str = typer.Option(
        "all", help="The name of the dataset to fit model on. Defaults to 'all'."
    ),
    num_gnn_iters: int = typer.Option(
        default=300, help="Number of iterations to train GNN model"
    ),
):
    """Main function."""

    dataset_names = [dataset_name]
    if dataset_name == "all":
        dataset_names = sorted(base_datasets.keys())

    # Set up models to be trained
    master_key = random.PRNGKey(99)
    cv_keys = random.split(master_key, 5)
    for dataset_name in dataset_names:
        logger.info(f"Fitting models for {dataset_name}")
        df, entropy = read_data(dataset_name)
        for cv_key in cv_keys:
            linear_model = LinearRegression()
            rf_model = RandomForestRegressor(n_estimators=300, n_jobs=-1)
            gnn_model = GATModel(cv_key)

            model_object, train_test_data = train_entropy(
                cv_key, df, entropy, linear_model
            )
            pickle_package(model_object, train_test_data, cv_key, dataset_name)

            model_object, train_test_data = train_entropy(cv_key, df, entropy, rf_model)
            pickle_package(model_object, train_test_data, cv_key, dataset_name)

            model_object, train_test_data = train_gnn(
                cv_key, df, entropy, gnn_model, num_iters=num_gnn_iters
            )
            pickle_package(model_object, train_test_data, cv_key, dataset_name)


if __name__ == "__main__":
    app()
