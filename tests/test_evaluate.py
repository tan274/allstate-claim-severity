import numpy as np
import pandas as pd
import joblib
import pytest
from pathlib import Path
from sklearn.dummy import DummyRegressor

import src.evaluate as evaluate_module
from src.evaluate import evaluate


@pytest.fixture
def fake_artifacts(tmp_path):
    # Minimal preprocessor matching transform_features() contract
    preprocessor = {
        "cat_cols": ["cat1"],
        "num_cols": ["cont1"],
        "freq_maps": {"cat1": {"A": 3, "B": 1}},
        "fill_values": {"cont1": 0.5},
        "feature_order": ["cat1", "cont1"],
    }

    # Fake model trained in log space
    X = pd.DataFrame({"cat1": [3, 3, 1], "cont1": [0.1, 0.2, 0.3]})
    y_log = np.log1p([100.0, 200.0, 300.0])
    model = DummyRegressor(strategy="mean")
    model.fit(X, y_log)

    joblib.dump(model, tmp_path / "model.joblib")
    joblib.dump(preprocessor, tmp_path / "preprocessor.joblib")
    return tmp_path


@pytest.fixture
def fake_csv(tmp_path):
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "cat1": ["A", "B", "A"],
        "cont1": [0.1, 0.2, 0.3],
        "loss": [100.0, 200.0, 300.0],
    })
    path = tmp_path / "val.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_evaluate_prints_metrics(fake_artifacts, fake_csv, capsys, monkeypatch):
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error

    monkeypatch.setattr(evaluate_module, "ARTIFACTS", fake_artifacts)
    evaluate(fake_csv)
    out = capsys.readouterr().out

    # Compute expected values using the same fake model/preprocessor
    from src.features import transform_features, split_features_target
    model = joblib.load(fake_artifacts / "model.joblib")
    preprocessor = joblib.load(fake_artifacts / "preprocessor.joblib")
    df = pd.read_csv(fake_csv)
    X, y = split_features_target(df)
    X = transform_features(X, preprocessor)
    preds = np.expm1(model.predict(X))
    expected_mae = f"{mean_absolute_error(y, preds):.2f}"
    expected_rmse = f"{root_mean_squared_error(y, preds):.2f}"

    assert f"MAE:  {expected_mae}" in out
    assert f"RMSE: {expected_rmse}" in out


def test_evaluate_saves_residuals_png(fake_artifacts, fake_csv, monkeypatch):
    monkeypatch.setattr(evaluate_module, "ARTIFACTS", fake_artifacts)
    evaluate(fake_csv)
    assert (fake_artifacts / "residuals.png").exists()


def test_evaluate_drops_null_loss(fake_artifacts, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(evaluate_module, "ARTIFACTS", fake_artifacts)
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "cat1": ["A", "B", "A"],
        "cont1": [0.1, 0.2, 0.3],
        "loss": [100.0, None, 300.0],
    })
    path = tmp_path / "val_nulls.csv"
    df.to_csv(path, index=False)
    evaluate(str(path))
    out = capsys.readouterr().out
    assert "Dropped 1 row(s)" in out
    assert "MAE:" in out
