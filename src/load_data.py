import pandas as pd
from pathlib import Path


def load_train(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Train file not found: {path}")
    df = pd.read_csv(p, on_bad_lines="skip")
    print(f"train: {df.shape}")
    if "loss" not in df.columns:
        raise ValueError("Expected 'loss' column in train data")
    null_loss = df["loss"].isnull().sum()
    if null_loss > 0:
        df = df.dropna(subset=["loss"])
        print(f"  dropped {null_loss} row(s) with null loss")
    return df


def load_test(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Test file not found: {path}")
    df = pd.read_csv(p)
    print(f"test: {df.shape}")
    return df
