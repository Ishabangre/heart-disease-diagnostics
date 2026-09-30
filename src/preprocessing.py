"""Data loading, cleaning and splitting."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "heart.csv"
TARGET = "target"
RANDOM_STATE = 2  # same as the original notebook
DROP_DUPLICATES = True  # honest baseline (no train/test leakage)

RAW_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def load_data(path=DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put heart.csv inside the dataset/ folder."
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame, drop_duplicates: bool = DROP_DUPLICATES) -> pd.DataFrame:
    """Basic cleaning. Duplicate rows are removed by default: this dataset has
    many repeats, and the same row in train AND test inflates accuracy (leakage)."""
    df = df.copy()
    if drop_duplicates:
        df = df.drop_duplicates()
    df = df.dropna()
    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame):
    return df.drop(columns=TARGET), df[TARGET]


def split_data(X, y, test_size: float = 0.2):
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )


def prepare_data(drop_duplicates: bool = DROP_DUPLICATES, use_feature_engineering: bool = True):
    """One call: load -> clean -> features -> split.
    Returns X_train, X_test, y_train, y_test (used by train/evaluate/explain)."""
    from src.feature_engineering import add_features

    df = clean_data(load_data(), drop_duplicates=drop_duplicates)
    if use_feature_engineering:
        df = add_features(df)
    X, y = split_features_target(df)
    return split_data(X, y)
