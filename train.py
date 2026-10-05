"""Retrains the 7 trait models (same logic as train.ipynb) and enforces an accuracy gate.

Usage:  python train.py [--out models.pkl] [--min-accuracy 0.85]
Exit code 1 if any trait model falls below the threshold (used as a CI quality gate).
"""
import argparse
import json
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

TRAITS = ["conf", "disc", "lead", "neuro", "open", "agree", "extra"]


def make_data(n=2000, seed=42):
    np.random.seed(seed)
    X = np.random.uniform(0, 5, (n, 15))
    y = {
        "conf": (X[:, 3] + (5 - X[:, 1]) / 5 + X[:, 5] > 2.2),
        "disc": (X[:, 0] + X[:, 2] + X[:, 7] / 5 + X[:, 12] > 2.0),
        "lead": (X[:, 5] + X[:, 8] + X[:, 13] > 2.3),
        "neuro": ((5 - X[:, 1]) / 5 + (5 - X[:, 14]) / 5 < 1.2),
        "open": (X[:, 3] + X[:, 6] + X[:, 10] > 2.4),
        "agree": (X[:, 11] > 2.5),
        "extra": (X[:, 5] + X[:, 10] > 2.0),
    }
    return X, {k: v.astype(int) for k, v in y.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models.pkl")
    parser.add_argument("--metrics", default="metrics.json")
    parser.add_argument("--min-accuracy", type=float, default=0.85)
    args = parser.parse_args()

    X, labels = make_data()
    models, accuracies = {}, {}
    for trait in TRAITS:
        X_tr, X_te, y_tr, y_te = train_test_split(X, labels[trait], test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_tr, y_tr)
        accuracies[trait] = float(accuracy_score(y_te, model.predict(X_te)))
        models[trait] = model
        print(f"{trait:>6}: {accuracies[trait]:.1%}")

    avg = float(np.mean(list(accuracies.values())))
    print(f"average: {avg:.1%}")
    joblib.dump(models, args.out)
    with open(args.metrics, "w") as f:
        json.dump({"accuracies": accuracies, "average": avg}, f, indent=2)

    failing = {t: a for t, a in accuracies.items() if a < args.min_accuracy}
    if failing:
        print(f"FAILED accuracy gate ({args.min_accuracy:.0%}): {failing}")
        sys.exit(1)
    print("Accuracy gate passed.")


if __name__ == "__main__":
    main()
