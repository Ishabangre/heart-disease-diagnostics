"""Full training pipeline.

  baseline CV (all models) -> hyper-parameter tuning -> ensembles
  -> pick best by CV ROC-AUC -> probability calibration -> threshold tuning
  -> evaluate on the untouched test set -> save everything.

Run from project root:
    python -m src.train                 # full run
    python -m src.train --n-iter 5      # faster (fewer tuning trials)
    python -m src.train --keep-duplicates
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, fbeta_score
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict, cross_validate)

from src.evaluate import compute_metrics, evaluate_model, save_calibration_plot
from src.model import (AVAILABLE_MODELS, PARAM_DISTRIBUTIONS, build_model,
                       build_stacking, build_voting)
from src.preprocessing import RANDOM_STATE, prepare_data

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "outputs" / "reports"

SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
           "f1": "f1", "roc_auc": "roc_auc"}
BETA = 2.0  # F2 score: recall counts 2x more than precision (missing a sick patient is worse)


def cv_summary(model, X, y) -> dict:
    """Repeated stratified 5-fold CV -> mean and std of every metric."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
    res = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    out = {}
    for k in SCORING:
        out[f"cv_{k}"] = float(res[f"test_{k}"].mean())
        out[f"cv_{k}_std"] = float(res[f"test_{k}"].std())
    return out


def tune(name, X, y, n_iter):
    search = RandomizedSearchCV(
        build_model(name), PARAM_DISTRIBUTIONS[name], n_iter=n_iter,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=-1, refit=False)
    search.fit(X, y)
    return search.best_params_


def tune_threshold(model, X, y, beta=BETA) -> float:
    """Pick the decision threshold on OUT-OF-FOLD train predictions (no test leakage)."""
    oof = cross_val_predict(
        CalibratedClassifierCV(clone(model), method="sigmoid", cv=5), X, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        method="predict_proba")[:, 1]
    grid = np.arange(0.10, 0.91, 0.01)
    scores = np.array([fbeta_score(y, (oof >= t).astype(int), beta=beta) for t in grid])
    best = grid[scores >= scores.max() - 1e-9]
    return float(np.round(np.median(best), 2))


def main(drop_duplicates=True, use_feature_engineering=True, n_iter=20):
    X_train, X_test, y_train, y_test = prepare_data(drop_duplicates, use_feature_engineering)
    print(f"rows: train={len(X_train)} test={len(X_test)} | drop_duplicates={drop_duplicates}")

    rows, candidates = [], {}

    # 1) baseline + 2) tuned
    for name in AVAILABLE_MODELS:
        print(f"\n[{name}]")
        base = build_model(name)
        rows.append({"model": name, "stage": "baseline", **cv_summary(base, X_train, y_train),
                     "test_accuracy": float(clone(base).fit(X_train, y_train).score(X_test, y_test))})
        params = tune(name, X_train, y_train, n_iter)
        tuned = build_model(name, **params)
        r = {"model": name, "stage": "tuned", **cv_summary(tuned, X_train, y_train),
             "test_accuracy": float(clone(tuned).fit(X_train, y_train).score(X_test, y_test))}
        rows.append(r)
        candidates[name] = (tuned, params)
        print(f"  baseline CV AUC {rows[-2]['cv_roc_auc']:.3f} -> tuned {r['cv_roc_auc']:.3f}")

    # 3) ensembles built from the top-3 tuned models
    top3 = sorted(candidates, key=lambda n: next(
        x["cv_roc_auc"] for x in rows if x["model"] == n and x["stage"] == "tuned"), reverse=True)[:3]
    members = {n: clone(candidates[n][0]) for n in top3}
    for ens_name, ens in [("voting_ensemble", build_voting(members)),
                          ("stacking_ensemble", build_stacking(members))]:
        rows.append({"model": ens_name, "stage": "ensemble", **cv_summary(ens, X_train, y_train),
                     "test_accuracy": float(clone(ens).fit(X_train, y_train).score(X_test, y_test))})
        candidates[ens_name] = (ens, {"members": top3})
        print(f"\n[{ens_name}] members={top3} CV AUC {rows[-1]['cv_roc_auc']:.3f}")

    # 4) choose best by CV ROC-AUC (test set is NOT used for selection)
    comp = pd.DataFrame(rows)
    comp = comp[comp["stage"] != "baseline"].sort_values("cv_roc_auc", ascending=False)
    best_name = comp.iloc[0]["model"]
    best_model, best_params = candidates[best_name]
    print(f"\nBEST MODEL (by CV ROC-AUC): {best_name}")

    # 5) calibration
    uncal = clone(best_model).fit(X_train, y_train)
    cal = CalibratedClassifierCV(clone(best_model), method="sigmoid", cv=5).fit(X_train, y_train)
    brier_before = float(brier_score_loss(y_test, uncal.predict_proba(X_test)[:, 1]))
    brier_after = float(brier_score_loss(y_test, cal.predict_proba(X_test)[:, 1]))

    # 6) threshold tuning (recall-focused)
    threshold = tune_threshold(best_model, X_train, y_train)
    print(f"Tuned threshold (F{BETA:.0f}): {threshold}")

    # 7) final evaluation on the test set
    test_default = compute_metrics(cal, X_test, y_test, 0.5)
    test_tuned = evaluate_model(cal, X_test, y_test, threshold)
    save_calibration_plot(uncal, cal, X_test, y_test)
    train_acc = float(((cal.predict_proba(X_train)[:, 1] >= threshold) == y_train).mean())

    # 8) save
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(cal, MODEL_DIR / "heart_model.pkl")
    joblib.dump(list(X_train.columns), MODEL_DIR / "feature_columns.pkl")
    joblib.dump(X_train.sample(min(100, len(X_train)), random_state=RANDOM_STATE),
                MODEL_DIR / "background.pkl")
    pd.DataFrame(rows).to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    metrics = {
        "best_model": best_name,
        "best_params": clean({k.replace("model__", ""): v for k, v in best_params.items()}),
        "drop_duplicates": drop_duplicates,
        "feature_engineering": use_feature_engineering,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "threshold": threshold, "threshold_beta": BETA,
        "train_accuracy": train_acc,
        "test": clean(test_tuned),
        "test_at_0_5": clean(test_default),
        "calibration": {"brier_before": brier_before, "brier_after": brier_after},
        "comparison": clean(rows),
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    t, d = test_tuned, test_default
    print(f"\nTrain accuracy        : {train_acc:.3f}")
    print(f"Test @0.50            : acc {d['accuracy']:.3f} | recall {d['recall']:.3f} | AUC {d['roc_auc']:.3f}")
    print(f"Test @{threshold:.2f} (tuned)    : acc {t['accuracy']:.3f} | recall {t['recall']:.3f} | AUC {t['roc_auc']:.3f}")
    print(f"Brier before/after cal: {brier_before:.3f} -> {brier_after:.3f} (lower = better)")
    print(f"Saved to {MODEL_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--no-fe", action="store_true", help="disable feature engineering")
    ap.add_argument("--n-iter", type=int, default=20, help="tuning trials per model")
    a = ap.parse_args()
    main(drop_duplicates=not a.keep_duplicates,
         use_feature_engineering=not a.no_fe, n_iter=a.n_iter)

    
    