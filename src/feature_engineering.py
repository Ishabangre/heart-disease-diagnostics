"""Optional derived features. Same function is used in training AND prediction."""
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # how close the person got to their age-predicted max heart rate
    df["hr_ratio"] = df["thalach"] / (220 - df["age"]).clip(lower=1)
    # blood pressure x cholesterol load (scaled down)
    df["bp_chol_index"] = (df["trestbps"] * df["chol"]) / 1000
    # exercise-induced stress signal
    df["stress_index"] = df["oldpeak"] * (1 + df["exang"])
    # age group buckets: 0 <40, 1 40-54, 2 55-64, 3 65+
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 39, 54, 64, 200], labels=[0, 1, 2, 3]
    ).astype(int)
    return df
