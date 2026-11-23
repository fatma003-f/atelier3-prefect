"""Tests unitaires de model_pipeline."""

import pandas as pd

from model_pipeline import regrouper_diagnostic, train_model


def test_regrouper_diagnostic():
    assert regrouper_diagnostic("250.83") == "Diabetes"
    assert regrouper_diagnostic("428") == "Circulatory"
    assert regrouper_diagnostic("V57") == "Other"
    assert regrouper_diagnostic("Unknown") == "Unknown"


def test_train_model_predict():
    X = pd.DataFrame({"age": [45, 55, 65, 75] * 5, "gender": ["M", "F"] * 10})
    y = pd.Series([0, 1, 0, 1] * 5)
    model = train_model(X, y)
    assert len(model.predict(X)) == len(X)
