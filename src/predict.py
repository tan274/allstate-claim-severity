import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import joblib
import numpy as np
import pandas as pd

from features import transform_features

ARTIFACTS = Path("artifacts")


def load_artifacts():
    model = joblib.load(ARTIFACTS / "model.joblib")
    preprocessor = joblib.load(ARTIFACTS / "preprocessor.joblib")
    return model, preprocessor


def predict(df: pd.DataFrame) -> np.ndarray:
    model, preprocessor = load_artifacts()
    X = df.drop(columns=["id", "loss"], errors="ignore")
    X = transform_features(X, preprocessor)
    return np.expm1(model.predict(X))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py '{\"cat1\": \"A\", \"cont1\": 0.5, ...}'")
        sys.exit(1)

    payload = json.loads(sys.argv[1])
    df = pd.DataFrame([payload])
    preds = predict(df)
    print(f"Predicted loss: {preds[0]:.2f}")
