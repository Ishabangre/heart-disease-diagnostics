"""Model zoo, hyper-parameter search spaces and ensembles.
Every model is a Pipeline (StandardScaler + estimator)."""
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.preprocessing import RANDOM_STATE

try:
    from xgboost import XGBClassifier
except ImportError:  # optional
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:  # optional
    LGBMClassifier = None

_FACTORIES = {
    "logistic_regression": lambda: LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "random_forest": lambda: RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
    "svm": lambda: SVC(probability=True, random_state=RANDOM_STATE),
    "knn": lambda: KNeighborsClassifier(),
}
if XGBClassifier is not None:
    _FACTORIES["xgboost"] = lambda: XGBClassifier(
        eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0)
if LGBMClassifier is not None:
    _FACTORIES["lightgbm"] = lambda: LGBMClassifier(
        random_state=RANDOM_STATE, verbose=-1)

AVAILABLE_MODELS = list(_FACTORIES)

# search spaces for RandomizedSearchCV ("model__" = the estimator inside the Pipeline)
PARAM_DISTRIBUTIONS = {
    "logistic_regression": {"model__C": loguniform(1e-3, 1e2)},
    "random_forest": {
        "model__n_estimators": randint(100, 400),
        "model__max_depth": randint(2, 10),
        "model__min_samples_leaf": randint(1, 10),
        "model__max_features": ["sqrt", "log2", 0.5],
    },
    "gradient_boosting": {
        "model__n_estimators": randint(50, 300),
        "model__learning_rate": loguniform(0.01, 0.3),
        "model__max_depth": randint(1, 4),
        "model__subsample": uniform(0.6, 0.4),
    },
    "svm": {
        "model__C": loguniform(0.1, 50),
        "model__gamma": loguniform(1e-3, 1),
    },
    "knn": {
        "model__n_neighbors": randint(3, 25),
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
    "xgboost": {
        "model__n_estimators": randint(50, 300),
        "model__max_depth": randint(2, 6),
        "model__learning_rate": loguniform(0.01, 0.3),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
    },
    "lightgbm": {
        "model__n_estimators": randint(50, 300),
        "model__num_leaves": randint(4, 31),
        "model__learning_rate": loguniform(0.01, 0.3),
        "model__min_child_samples": randint(5, 30),
        "model__colsample_bytree": uniform(0.6, 0.4),
    },
}


def build_model(name: str = "logistic_regression", **params) -> Pipeline:
    if name not in _FACTORIES:
        raise ValueError(f"Unknown model '{name}'. Available: {AVAILABLE_MODELS}")
    pipe = Pipeline([("scaler", StandardScaler()), ("model", _FACTORIES[name]())])
    return pipe.set_params(**params) if params else pipe


def build_voting(models: dict) -> VotingClassifier:
    """Soft-voting ensemble. models = {name: unfitted pipeline}"""
    return VotingClassifier(list(models.items()), voting="soft")


def build_stacking(models: dict) -> StackingClassifier:
    """Stacking ensemble with a Logistic Regression meta-model."""
    return StackingClassifier(
        list(models.items()),
        final_estimator=LogisticRegression(max_iter=2000),
        cv=5,
    )
