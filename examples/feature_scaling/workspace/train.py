"""Fixed-budget pilot. Fit preprocessing on training data; evaluate validation only."""

import argparse
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning

from preprocess import make_preprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("wine", "breast_cancer", "digits"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = {"wine": load_wine, "breast_cancer": load_breast_cancer, "digits": load_digits}[args.dataset]()
    indices = np.arange(len(data.target))
    development, test = train_test_split(indices, test_size=0.2, random_state=args.split_seed,
                                         stratify=data.target)
    train, validation = train_test_split(development, test_size=0.25, random_state=args.split_seed,
                                         stratify=data.target[development])
    splits = {"train": train.tolist(), "validation": validation.tolist(), "test_reserved": test.tolist()}
    split_hash = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
    settings = dict(hidden_layer_sizes=(32,), activation="relu", solver="adam", alpha=0.0001,
                    batch_size=64, learning_rate_init=0.001, max_iter=80, tol=0,
                    n_iter_no_change=81, early_stopping=False)
    classifier = MLPClassifier(random_state=args.seed, **settings)
    pipeline = make_pipeline(make_preprocessor(), classifier)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(data.data[train], data.target[train])
    fit_seconds = time.perf_counter() - started
    prediction = pipeline.predict(data.data[validation])
    output = {
        "schema_version": 1, "dataset": args.dataset, "seed": args.seed,
        "split_seed": args.split_seed, "partition": "validation", "split_hash": split_hash,
        "metrics": {"accuracy": float(accuracy_score(data.target[validation], prediction)),
                    "macro_f1": float(f1_score(data.target[validation], prediction, average="macro"))},
        "protocol": {
            "estimator": "MLPClassifier", "settings": settings, "split_seed": args.split_seed,
            "dataset_sha256": hashlib.sha256(data.data.tobytes() + data.target.tobytes()).hexdigest(),
            "library_versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                                 "sklearn": sklearn.__version__},
            "train_n": len(train), "validation_n": len(validation), "test_reserved_n": len(test),
            "preprocessing_fit_partition": "train", "test_evaluated": False,
        },
        "fit_seconds": fit_seconds, "training_iterations": int(classifier.n_iter_),
        "warnings": [str(item.message) for item in caught],
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "validation_predictions": prediction.tolist(),
        "validation_targets": data.target[validation].tolist(),
    }
    destination = Path(args.output)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"dataset": args.dataset, "seed": args.seed, **output["metrics"]}))


if __name__ == "__main__":
    main()
