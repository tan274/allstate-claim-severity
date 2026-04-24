import pandas as pd


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=["loss", "id"], errors="ignore")
    y = df["loss"]
    return X, y


def get_column_groups(df: pd.DataFrame):
    cat_cols = [c for c in df.columns if c.startswith("cat")]
    num_cols = [c for c in df.columns if c.startswith("cont")]
    return cat_cols, num_cols


def fit_preprocessor(train_df: pd.DataFrame) -> dict:
    X, _ = split_features_target(train_df)
    cat_cols, num_cols = get_column_groups(X)

    freq_maps = {col: X[col].value_counts().to_dict() for col in cat_cols}
    fill_values = {col: X[col].median() for col in num_cols}
    feature_order = list(X.columns)

    return {
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "freq_maps": freq_maps,
        "fill_values": fill_values,
        "feature_order": feature_order,
    }


def transform_features(df: pd.DataFrame, preprocessor: dict) -> pd.DataFrame:
    result = {}

    for col in preprocessor["cat_cols"]:
        if col not in df.columns:
            result[col] = 0
        else:
            result[col] = df[col].map(preprocessor["freq_maps"][col]).fillna(0)

    for col in preprocessor["num_cols"]:
        if col not in df.columns:
            result[col] = preprocessor["fill_values"][col]
        else:
            result[col] = df[col].fillna(preprocessor["fill_values"][col])

    return pd.DataFrame(result, index=df.index)[preprocessor["feature_order"]]
