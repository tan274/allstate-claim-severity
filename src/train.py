import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from load_data import load_train
from features import fit_preprocessor, transform_features, split_features_target

ARTIFACTS = Path("artifacts")
DATA_PATH = "data/raw/train.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def train():
    # Load and split
    df = load_train(DATA_PATH)
    train_df, val_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Preprocess
    preprocessor = fit_preprocessor(train_df)
    X_train, y_train = split_features_target(train_df)
    X_val, y_val = split_features_target(val_df)
    X_train = transform_features(X_train, preprocessor)
    X_val = transform_features(X_val, preprocessor)
    y_train_log = np.log1p(y_train)

    # Train both models
    models = {
        "LightGBM": LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
        "XGBoost": XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
    }

    metrics = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train_log)
        preds = np.expm1(model.predict(X_val))
        metrics[name] = {
            "MAE": round(mean_absolute_error(y_val, preds), 2),
            "RMSE": round(root_mean_squared_error(y_val, preds), 2),
        }
        trained[name] = model
        print(f"{name}: MAE={metrics[name]['MAE']}, RMSE={metrics[name]['RMSE']}")

    # Pick best by MAE
    best_name = min(metrics, key=lambda k: metrics[k]["MAE"])
    best_model = trained[best_name]
    print(f"\nBest model: {best_name}")

    # Save artifacts
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(best_model, ARTIFACTS / "model.joblib")
    joblib.dump(preprocessor, ARTIFACTS / "preprocessor.joblib")

    metrics_out = {"models": metrics, "winner": best_name}
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics_out, indent=2))

    split_meta = {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
    }
    (ARTIFACTS / "split_meta.json").write_text(json.dumps(split_meta, indent=2))

    # Save validation set for Phase 8/9
    processed = Path("data/processed")
    processed.mkdir(exist_ok=True)
    val_df.to_parquet(processed / "val.parquet", index=False)

    print("Artifacts saved to artifacts/")


if __name__ == "__main__":
    train()
