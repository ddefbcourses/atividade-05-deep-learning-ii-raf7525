from __future__ import annotations

from typing import Iterable


DEFAULT_EXPERIMENT_NAME = "atividade_05_cifar100_mlp"


def _reshape_images(images):

    return images.reshape(images.shape[0], -1)


def load_data(seed, val_size=0.2, label_mode="fine"):

    from sklearn.model_selection import train_test_split
    from tensorflow.keras.datasets import cifar100

    from src.utils import normalize_images, set_seed

    set_seed(seed)

    (X_train, y_train), _ = cifar100.load_data(
        label_mode=label_mode
    )

    X_train = normalize_images(
        _reshape_images(X_train).astype("float32")
    )
    y_train = y_train.reshape(-1)

    return train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=seed,
        stratify=y_train
    )


def load_test_data(label_mode="fine"):

    from tensorflow.keras.datasets import cifar100

    from src.utils import normalize_images

    _, (X_test, y_test) = cifar100.load_data(
        label_mode=label_mode
    )

    X_test = normalize_images(
        _reshape_images(X_test).astype("float32")
    )

    return X_test, y_test.reshape(-1)


def train_mlp(
    X_train,
    y_train,
    activation,
    hidden_layers,
    learning_rate,
    seed,
    max_iter=25,
    batch_size=128,
    early_stopping=True,
    verbose=False
):

    from sklearn.neural_network import MLPClassifier

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        learning_rate_init=learning_rate,
        random_state=seed,
        max_iter=max_iter,
        batch_size=batch_size,
        early_stopping=early_stopping,
        n_iter_no_change=5,
        solver="adam",
        verbose=verbose
    )

    model.fit(X_train, y_train)

    return model


def evaluate(model, X_test, y_test):

    from src.metrics import classification_metrics

    predictions = model.predict(X_test)
    metrics = classification_metrics(
        y_test,
        predictions
    )

    return metrics, predictions


def first_layer_parameter_count(input_dim, hidden_units):

    return (input_dim * hidden_units) + hidden_units


def run_tracked_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    activation,
    hidden_layers,
    learning_rate,
    seed,
    max_iter=25,
    batch_size=128,
    experiment_name=DEFAULT_EXPERIMENT_NAME,
    run_name=None,
    verbose=False
):

    from src.experiment import (
        log_metrics,
        log_params,
        measure_training_time,
        setup_experiment,
        start_run,
    )

    params = {
        "activation": activation,
        "hidden_layers": str(hidden_layers),
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "batch_size": batch_size,
        "seed": seed,
    }

    setup_experiment(experiment_name)

    with start_run(run_name=run_name):

        log_params(params)

        model, training_time = measure_training_time(
            train_mlp,
            X_train,
            y_train,
            activation,
            hidden_layers,
            learning_rate,
            seed,
            max_iter=max_iter,
            batch_size=batch_size,
            verbose=verbose,
        )

        metrics, predictions = evaluate(
            model,
            X_test,
            y_test
        )
        metrics["training_time"] = training_time
        metrics["final_loss"] = model.loss_
        metrics["n_iterations"] = float(model.n_iter_)

        log_metrics(metrics)

    return {
        "model": model,
        "metrics": metrics,
        "predictions": predictions,
        "params": params,
    }


def benchmark_configurations(
    configurations: Iterable[dict],
    X_train,
    y_train,
    X_test,
    y_test,
    experiment_name=DEFAULT_EXPERIMENT_NAME,
):

    import pandas as pd

    rows = []

    for config in configurations:

        result = run_tracked_experiment(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            experiment_name=experiment_name,
            **config,
        )

        rows.append(
            {
                **result["params"],
                **result["metrics"],
            }
        )

    results = pd.DataFrame(rows)

    return results.sort_values(
        by="accuracy",
        ascending=False,
    ).reset_index(drop=True)