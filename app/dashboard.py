"""Dashboard: model comparison, CV results, calibration, threshold, graphs."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]


def _img(col, path, caption):
    if path.exists():
        col.image(str(path), caption=caption)


def render_dashboard():
    st.header("📊 Model Dashboard")

    mpath = ROOT / "models" / "metrics.json"
    if not mpath.exists():
        st.warning("No metrics found. Run `python -m src.train` first.")
        return
    m = json.loads(mpath.read_text())
    t = m["test"]

    st.subheader(f"Best model: `{m['best_model']}`")
    st.caption(f"Train rows: {m['n_train']} | Test rows: {m['n_test']} | "
               f"Duplicates removed: {m['drop_duplicates']}")

    c = st.columns(5)
    c[0].metric("Accuracy", f"{t['accuracy']:.1%}")
    c[1].metric("Recall", f"{t['recall']:.1%}")
    c[2].metric("Precision", f"{t['precision']:.1%}")
    c[3].metric("F1", f"{t['f1']:.1%}")
    c[4].metric("ROC-AUC", f"{t['roc_auc']:.3f}")

    # ---- threshold ----
    st.subheader("🎯 Decision threshold")
    d = m["test_at_0_5"]
    st.write(f"Tuned threshold: **{m['threshold']:.2f}** (recall-focused, F{m['threshold_beta']:.0f}) "
             f"instead of the default 0.50")
    st.dataframe(pd.DataFrame({
        "Threshold 0.50": [d["accuracy"], d["recall"], d["specificity"], d["precision"]],
        f"Tuned {m['threshold']:.2f}": [t["accuracy"], t["recall"], t["specificity"], t["precision"]],
    }, index=["Accuracy", "Recall", "Specificity", "Precision"]).style.format("{:.1%}"))

    # ---- calibration ----
    st.subheader("📐 Probability calibration")
    cal = m["calibration"]
    c1, c2 = st.columns(2)
    c1.metric("Brier before", f"{cal['brier_before']:.3f}")
    c2.metric("Brier after", f"{cal['brier_after']:.3f}",
              delta=f"{cal['brier_after'] - cal['brier_before']:.3f}", delta_color="inverse")
    st.caption("Brier score: lower = predicted probabilities are more trustworthy.")

    # ---- comparison table ----
    st.subheader("🏆 Model comparison (5-fold x2 cross-validation, mean ± std)")
    df = pd.DataFrame(m["comparison"])
    df["CV accuracy"] = df.apply(lambda r: f"{r.cv_accuracy:.3f} ± {r.cv_accuracy_std:.3f}", axis=1)
    df["CV recall"] = df.apply(lambda r: f"{r.cv_recall:.3f} ± {r.cv_recall_std:.3f}", axis=1)
    df["CV ROC-AUC"] = df.apply(lambda r: f"{r.cv_roc_auc:.3f} ± {r.cv_roc_auc_std:.3f}", axis=1)
    show = (df.sort_values("cv_roc_auc", ascending=False)
              [["model", "stage", "CV accuracy", "CV recall", "CV ROC-AUC", "test_accuracy"]]
              .rename(columns={"test_accuracy": "Test acc (info only)"}))
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.caption("Best model is chosen by CV ROC-AUC, not by test accuracy. "
               "`tuned` scores are slightly optimistic because tuning used the same folds.")

    chart = df[df["stage"] != "baseline"].set_index("model")[["cv_roc_auc"]]
    st.bar_chart(chart)

    # ---- graphs ----
    st.subheader("Graphs")
    left, right = st.columns(2)
    _img(left, ROOT / "outputs" / "confusion_matrix" / "confusion_matrix.png", "Confusion matrix")
    _img(right, ROOT / "outputs" / "graphs" / "roc_curve.png", "ROC curve")
    left2, right2 = st.columns(2)
    _img(left2, ROOT / "outputs" / "graphs" / "calibration_curve.png", "Calibration curve")
    _img(right2, ROOT / "outputs" / "graphs" / "feature_importance.png", "Feature importance")
