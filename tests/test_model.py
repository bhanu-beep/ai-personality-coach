"""Guards the ML artifact: right version, right shape, sane outputs."""
import os
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models.pkl")
TRAITS = ["conf", "disc", "lead", "neuro", "open", "agree", "extra"]


def test_model_loads_without_version_mismatch():
    # Fails if the scikit-learn version in requirements.txt differs from the one used to train models.pkl
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        joblib.load(MODEL_PATH)
    assert sklearn.__version__ == "1.8.0"


def test_all_seven_trait_models_present():
    models = joblib.load(MODEL_PATH)
    assert sorted(models.keys()) == sorted(TRAITS)


def test_models_expect_15_features_and_return_probabilities():
    models = joblib.load(MODEL_PATH)
    x = np.full((1, 15), 3.0)
    for trait in TRAITS:
        assert models[trait].n_features_in_ == 15
        proba = models[trait].predict_proba(x)[0]
        assert abs(proba.sum() - 1.0) < 1e-6
        assert 0.0 <= proba[1] <= 1.0


def test_prediction_is_deterministic():
    models = joblib.load(MODEL_PATH)
    x = np.array([[1, 2, 1, 1, 0.5, 1, 1, 4, 0.5, 1, 1, 1, 5, 1, 4]])
    first = [models[t].predict_proba(x)[0][1] for t in TRAITS]
    second = [models[t].predict_proba(x)[0][1] for t in TRAITS]
    assert first == second
