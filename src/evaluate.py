"""Metrics + plots (confusion matrix, ROC, calibration, text report).
All functions accept a decision `threshold` (default 0.5)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             brier_score_loss, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
CM_DIR = ROOT / "outputs" / "confusion_matrix"
GRAPH_DIR = ROOT / "outputs" / "graphs"
REPORT_DIR = ROOT / "outputs" / "reports"


def _predict(model, X, threshold):
    return (model.predict_proba(X)[:, 1] >= threshold).astype(int)


def compute_metrics(model, X, y, threshold: float = 0.5) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "brier": brier_score_loss(y, proba),
        "threshold": threshold,
    }


def save_confusion_matrix(model, X, y, threshold=0.5, name="confusion_matrix.png"):
    CM_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y, _predict(model, X, threshold))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No disease", "Disease"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix (test, threshold={threshold:.2f})")
    fig.tight_layout()
    path = CM_DIR / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_roc_curve(model, X, y, name="roc_curve.png"):
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X, y, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC Curve (test set)")
    fig.tight_layout()
    path = GRAPH_DIR / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_calibration_plot(uncalibrated, calibrated, X, y, name="calibration_curve.png"):
    """Reliability diagram: closer to the diagonal = probabilities are trustworthy."""
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    for m, label in [(uncalibrated, "Before calibration"), (calibrated, "After calibration")]:
        p = m.predict_proba(X)[:, 1]
        frac, mean_pred = calibration_curve(y, p, n_bins=5, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", label=f"{label} (Brier {brier_score_loss(y, p):.3f})")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Actual fraction with disease")
    ax.set_title("Calibration curve (test set)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = GRAPH_DIR / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_classification_report(model, X, y, threshold=0.5, name="classification_report.txt"):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    text = classification_report(y, _predict(model, X, threshold),
                                 target_names=["No disease", "Disease"])
    path = REPORT_DIR / name
    path.write_text(f"Threshold = {threshold:.2f}\n\n{text}")
    return path


def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """Run full evaluation, save all outputs, return metrics."""
    metrics = compute_metrics(model, X_test, y_test, threshold)
    save_confusion_matrix(model, X_test, y_test, threshold)
    save_roc_curve(model, X_test, y_test)
    save_classification_report(model, X_test, y_test, threshold)
    return metrics


if __name__ == "__main__":
    import joblib
    from src.preprocessing import prepare_data

    cfg = json.loads((ROOT / "models" / "metrics.json").read_text())
    model = joblib.load(ROOT / "models" / "heart_model.pkl")
    cols = joblib.load(ROOT / "models" / "feature_columns.pkl")
    _, X_test, _, y_test = prepare_data(cfg["drop_duplicates"], cfg["feature_engineering"])
    print(json.dumps(evaluate_model(model, X_test[cols], y_test, cfg["threshold"]), indent=2))
