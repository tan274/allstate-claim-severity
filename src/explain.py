import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from features import transform_features, split_features_target

ARTIFACTS = Path("artifacts")
VAL_PATH = Path("data/processed/val.parquet")
SAMPLE_N = 2000


def explain():
    model = joblib.load(ARTIFACTS / "model.joblib")
    preprocessor = joblib.load(ARTIFACTS / "preprocessor.joblib")

    val_df = pd.read_parquet(VAL_PATH)
    sample = val_df.sample(n=min(SAMPLE_N, len(val_df)), random_state=42)

    X, _ = split_features_target(sample)
    X = transform_features(X, preprocessor)

    print("Computing SHAP values (log-loss space)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # shap_summary.png
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "shap_summary.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("Saved artifacts/shap_summary.png")

    # shap_importance.csv
    importance = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance.to_csv(ARTIFACTS / "shap_importance.csv", index=False)
    print("Saved artifacts/shap_importance.csv")

    # Top 10
    print("\nTop 10 features by mean |SHAP| (log-loss space):")
    print(importance.head(10).to_string(index=False))

    # One high and one low prediction row
    preds = model.predict(X)
    high_idx = np.argmax(preds)
    low_idx = np.argmin(preds)

    def top_row_shap(idx, n=5):
        row_shap = pd.DataFrame({"feature": X.columns, "shap": shap_values[idx]})
        return row_shap.reindex(row_shap["shap"].abs().sort_values(ascending=False).index).head(n)

    print(f"\nHighest prediction row (log-loss={preds[high_idx]:.3f}):")
    print(top_row_shap(high_idx).to_string(index=False))

    print(f"\nLowest prediction row (log-loss={preds[low_idx]:.3f}):")
    print(top_row_shap(low_idx).to_string(index=False))


if __name__ == "__main__":
    explain()
