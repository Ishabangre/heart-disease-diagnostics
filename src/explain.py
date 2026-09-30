"""Explainability:
  * permutation feature importance (global)
  * SHAP  : per-patient waterfall + global summary (beeswarm)
  * PDP   : partial dependence (average) and ICE (one patient) risk curves
Works with ANY saved model (linear, trees, ensembles, calibrated)."""
import io
import json
import threading
from pathlib import Path
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from src.feature_engineering import add_features
from src.preprocessing import RANDOM_STATE, RAW_FEATURES

ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = ROOT / "outputs" / "graphs"
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]


# ------------------------------------------------------------------ helpers
def load_background(n: int = 100) -> pd.DataFrame:
    """Sample of training rows (all model columns). Saved once to models/background.pkl,
    so the app does not need heart.csv."""
    path = ROOT / "models" / "background.pkl"
    if path.exists():
        return joblib.load(path)
    from src.preprocessing import prepare_data
    cfg = json.loads((ROOT / "models" / "metrics.json").read_text())
    cols = joblib.load(ROOT / "models" / "feature_columns.pkl")
    X_train, _, _, _ = prepare_data(cfg["drop_duplicates"], cfg["feature_engineering"])
    bg = X_train[cols].sample(min(n, len(X_train)), random_state=RANDOM_STATE)
    joblib.dump(bg, path)
    return bg


# ------------------------------------------------------------------ importance
def global_importance(model, X_test, y_test, n_repeats=15) -> pd.DataFrame:
    """Permutation importance (drop in ROC-AUC when a feature is shuffled)."""
    r = permutation_importance(model, X_test, y_test, n_repeats=n_repeats,
                               random_state=RANDOM_STATE, scoring="roc_auc")
    return (pd.DataFrame({"feature": X_test.columns, "importance": r.importances_mean})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def save_importance_plot(imp: pd.DataFrame, name="feature_importance.png"):
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["importance"], color="#c0392b")
    ax.set_xlabel("Drop in ROC-AUC when feature is shuffled")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    path = GRAPH_DIR / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ SHAP
def build_shap_explainer(model, background: pd.DataFrame, n_background: int = 40):
    """Model-agnostic SHAP on the calibrated RISK PROBABILITY (values are in % points)."""
    cols = list(background.columns)
    bg = background.sample(min(n_background, len(background)), random_state=RANDOM_STATE)

    def risk(X):
        return model.predict_proba(pd.DataFrame(X, columns=cols))[:, 1]

    return shap.Explainer(risk, shap.maskers.Independent(bg, max_samples=n_background),
                          algorithm="permutation")


def shap_explain(explainer, X: pd.DataFrame):
    """X: DataFrame (1 patient or many). Returns a shap.Explanation."""
    return explainer(X, max_evals=4 * X.shape[1] + 1, silent=True)


def plot_shap_waterfall(explanation_row, max_display: int = 10):
    """Why THIS patient got THIS risk. Red bars push risk up, blue push it down."""
    plt.figure(figsize=(7, 4.5))
    shap.plots.waterfall(explanation_row, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_shap_summary(explanation, max_display: int = 12):
    """Global view: which features matter most and in which direction."""
    plt.figure(figsize=(7, 5))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig

_PLOT_LOCK = threading.Lock()   # pyplot is not thread-safe (Streamlit runs sessions in threads)


def fig_to_png(fig, dpi: int = 150) -> bytes:
    """Figure -> PNG bytes, then close it. Bytes are safe to cache and show with st.image
    (a cached Figure can turn blank, because st.pyplot may clear it after first display)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def shap_summary_png(explanation, max_display: int = 12) -> bytes:
    with _PLOT_LOCK:
        return fig_to_png(plot_shap_summary(explanation, max_display))


def shap_waterfall_png(explanation_row, max_display: int = 10) -> bytes:
    with _PLOT_LOCK:
        return fig_to_png(plot_shap_waterfall(explanation_row, max_display))


def save_shap_plots(model, X, name="shap_summary.png"):
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    explainer = build_shap_explainer(model, load_background())
    exp = shap_explain(explainer, X)
    fig = plot_shap_summary(exp)
    path = GRAPH_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------ PDP / ICE / what-if
def default_grid(raw_ref: pd.DataFrame, feature: str, n_points: int = 25):
    vals = np.sort(raw_ref[feature].unique())
    if len(vals) <= 10:                       # categorical-like feature
        return vals
    return np.linspace(raw_ref[feature].min(), raw_ref[feature].max(), n_points)


def risk_curve(model, cols, base_rows: pd.DataFrame, feature: str, grid) -> pd.DataFrame:
    """Vary ONE raw feature over `grid`, keep the rest fixed, recompute engineered
    features and return the mean risk.
      base_rows = many patients -> Partial Dependence (average effect)
      base_rows = one patient   -> ICE curve (effect for that patient)"""
    base = base_rows[RAW_FEATURES]
    risks = []
    for v in grid:
        d = base.copy()
        d[feature] = v
        risks.append(model.predict_proba(add_features(d)[cols])[:, 1].mean())
    return pd.DataFrame({feature: grid, "risk": risks})


def save_pdp_plots(model, cols, raw_ref, features=NUMERIC_FEATURES, name="partial_dependence.png"):
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    for ax, f in zip(axes.ravel(), features):
        c = risk_curve(model, cols, raw_ref, f, default_grid(raw_ref, f))
        ax.plot(c[f], c["risk"], color="#c0392b", lw=2)
        ax.set_title(f)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(features):]:
        ax.axis("off")
    axes[0, 0].set_ylabel("Average predicted risk")
    fig.suptitle("Partial Dependence: how risk changes with each feature")
    fig.tight_layout()
    path = GRAPH_DIR / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ old linear-only helper (kept)
def local_contributions(model, row: pd.DataFrame):
    m = model
    if hasattr(m, "calibrated_classifiers_"):
        m = m.calibrated_classifiers_[0].estimator
    if not hasattr(m, "named_steps") or not hasattr(m.named_steps.get("model"), "coef_"):
        return None
    scaled = m.named_steps["scaler"].transform(row)[0]
    contrib = m.named_steps["model"].coef_[0] * scaled
    return (pd.DataFrame({"feature": row.columns, "contribution": contrib})
            .sort_values("contribution", key=abs, ascending=False).reset_index(drop=True))


if __name__ == "__main__":
    from src.preprocessing import prepare_data

    cfg = json.loads((ROOT / "models" / "metrics.json").read_text())
    model = joblib.load(ROOT / "models" / "heart_model.pkl")
    cols = joblib.load(ROOT / "models" / "feature_columns.pkl")
    _, X_test, _, y_test = prepare_data(cfg["drop_duplicates"], cfg["feature_engineering"])
    X_test = X_test[cols]

    imp = global_importance(model, X_test, y_test)
    print(imp.head(10))
    print("Saved:", save_importance_plot(imp))
    print("Saved:", save_shap_plots(model, X_test))
    print("Saved:", save_pdp_plots(model, cols, load_background()))