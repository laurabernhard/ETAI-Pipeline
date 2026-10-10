"""Model construction."""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessor


_MODELS = {
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
}


def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params", {})

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    return _MODELS[model_type](**params)


def build_pipeline(preprocessing_config: dict, model_config: dict) -> Pipeline:
    """
    The estimator that is cross-validated, tuned and refit: preprocessing and model as one
    Pipeline, so every learned step is re-fit on the training part of each fold. Step names
    address hyperparameters from outside (`model__max_depth`, `prep__numeric__impute__strategy`).
    Any new learned step (e.g. feature selection) belongs here, between "prep" and "model".
    """
    return Pipeline([
        ("prep", build_preprocessor(preprocessing_config)),
        ("model", build_model(model_config)),
    ])
