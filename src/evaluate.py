import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from features import transform_features, split_features_target

ARTIFACTS = Path("artifacts")


def load_file(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, on_bad_lines="skip")


def evaluate(file_path: str):
    model = joblib.load(ARTIFACTS / "model.joblib")
    preprocessor = joblib.load(ARTIFACTS / "preprocessor.joblib")

    df = load_file(file_path)
    has_labels = "loss" in df.columns

    if has_labels:
        null_loss = df["loss"].isnull().sum()
        if null_loss > 0:
            df = df.dropna(subset=["loss"])
            print(f"Dropped {null_loss} row(s) with null loss")
        X, y = split_features_target(df)
    else:
        X = df.drop(columns=["id"], errors="ignore")
        y = None

    X = transform_features(X, preprocessor)
    preds = np.expm1(model.predict(X))

    if has_labels:
        mae = mean_absolute_error(y, preds)
        rmse = root_mean_squared_error(y, preds)
        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")

        residuals = np.array(y) - preds
        plt.figure(figsize=(8, 5))
        plt.scatter(preds, residuals, alpha=0.2, s=5)
        plt.axhline(0, color="red", linewidth=1)
        plt.xlabel("Predicted loss ($)")
        plt.ylabel("Residual (actual - predicted)")
        plt.title("Residuals vs Predicted")
        plt.tight_layout()
        plt.savefig(ARTIFACTS / "residuals.png", dpi=100)
        plt.close()
        print("Saved artifacts/residuals.png")
    else:
        print("No loss column found — skipping metrics and residuals plot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to CSV or parquet file to evaluate")
    args = parser.parse_args()
    evaluate(args.file)
