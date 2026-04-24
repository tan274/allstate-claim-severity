import numpy as np
import pandas as pd
import joblib
import pytest
from pathlib import Path
from sklearn.dummy import DummyRegressor

import src.predict as predict_module
from src.predict import predict
from src.features import fit_preprocessor


@pytest.fixture
def fake_artifacts(tmp_path):
    preprocessor = {
        "cat_cols": ["cat1"],
        "num_cols": ["cont1"],
        "freq_maps": {"cat1": {"A": 3, "B": 1}},
        "fill_values": {"cont1": 0.5},
        "feature_order": ["cat1", "cont1"],
    }
    X = pd.DataFrame({"cat1": [3, 3, 1], "cont1": [0.1, 0.2, 0.3]})
    y_log = np.log1p([100.0, 200.0, 300.0])
    model = DummyRegressor(strategy="mean")
    model.fit(X, y_log)

    joblib.dump(model, tmp_path / "model.joblib")
    joblib.dump(preprocessor, tmp_path / "preprocessor.joblib")
    return tmp_path


def test_predict_returns_shape(fake_artifacts, monkeypatch):
    monkeypatch.setattr(predict_module, "ARTIFACTS", fake_artifacts)
    df = pd.DataFrame([{"cat1": "A", "cont1": 0.3}])
    result = predict(df)
    assert result.shape == (1,)


def test_predict_returns_original_scale(fake_artifacts, monkeypatch):
    monkeypatch.setattr(predict_module, "ARTIFACTS", fake_artifacts)
    df = pd.DataFrame([{"cat1": "A", "cont1": 0.3}])
    result = predict(df)
    # DummyRegressor mean of log1p([100, 200, 300]) ~ 5.21, expm1 ~ 182
    # result should be in dollar scale (well above log space values of ~5-8)
    assert result[0] > 50.0


def test_predict_unknown_category_does_not_crash(fake_artifacts, monkeypatch):
    monkeypatch.setattr(predict_module, "ARTIFACTS", fake_artifacts)
    df = pd.DataFrame([{"cat1": "UNKNOWN_XYZ", "cont1": 0.3}])
    result = predict(df)
    assert result.shape == (1,)


def test_predict_missing_columns_do_not_crash(fake_artifacts, monkeypatch):
    monkeypatch.setattr(predict_module, "ARTIFACTS", fake_artifacts)
    df = pd.DataFrame([{}])  # completely empty payload
    result = predict(df)
    assert result.shape == (1,)
