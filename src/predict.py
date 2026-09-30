"""Load the saved model and predict for one patient."""
import json
from pathlib import Path

import joblib
import pandas as pd

from src.feature_engineering import add_features
from src.preprocessing import RAW_FEATURES

ROOT = Path(__file__).resolve().parents[1]
_model = None
_columns = None
_threshold = 0.5


def load_artifacts():
    global _model, _columns, _threshold
    if _model is None:
        _model = joblib.load(ROOT / "models" / "heart_model.pkl")
        _columns = joblib.load(ROOT / "models" / "feature_columns.pkl")
        mpath = ROOT / "models" / "metrics.json"
        if mpath.exists():
            _threshold = float(json.loads(mpath.read_text()).get("threshold", 0.5))
    return _model, _columns


def build_input_frame(patient) -> pd.DataFrame:
    """patient: dict with the 13 raw features, or a tuple/list in order."""
    if not isinstance(patient, dict):
        patient = dict(zip(RAW_FEATURES, patient))
    missing = [c for c in RAW_FEATURES if c not in patient]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    _, columns = load_artifacts()
    df = add_features(pd.DataFrame([patient], columns=RAW_FEATURES))
    return df[columns]


def predict(patient) -> dict:
    model, _ = load_artifacts()
    X = build_input_frame(patient)
    proba = float(model.predict_proba(X)[0, 1])   # calibrated probability
    label = int(proba >= _threshold)
    return {
        "prediction": label,
        "probability": proba,
        "threshold": _threshold,
        "message": "Person HAS heart disease" if label
                   else "Person does NOT have heart disease",
    }


if __name__ == "__main__":
    sample = (71, 0, 0, 112, 149, 0, 1, 125, 0, 1.6, 1, 0, 2)
    print(predict(sample))