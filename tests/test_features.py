import pandas as pd
import pytest
from src.features import fit_preprocessor, transform_features, split_features_target


@pytest.fixture
def small_train():
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cat1": ["A", "A", "B", "A"],
        "cont1": [0.1, 0.2, None, 0.4],
        "loss": [100.0, 200.0, 300.0, 400.0],
    })


def test_no_nulls_after_transform(small_train):
    preprocessor = fit_preprocessor(small_train)
    X, _ = split_features_target(small_train)
    result = transform_features(X, preprocessor)
    assert result.isnull().sum().sum() == 0


def test_feature_order_stable(small_train):
    preprocessor = fit_preprocessor(small_train)

    X_train, _ = split_features_target(small_train)
    X_shuffled = X_train[X_train.columns[::-1]]  # reverse column order

    t1 = transform_features(X_train, preprocessor)
    t2 = transform_features(X_shuffled, preprocessor)

    assert list(t1.columns) == preprocessor["feature_order"]
    assert list(t2.columns) == preprocessor["feature_order"]


def test_unknown_categories_map_to_zero(small_train):
    preprocessor = fit_preprocessor(small_train)
    X, _ = split_features_target(small_train)

    X_new = X.copy()
    X_new["cat1"] = "UNKNOWN_XYZ"

    result = transform_features(X_new, preprocessor)
    assert (result["cat1"] == 0).all()
