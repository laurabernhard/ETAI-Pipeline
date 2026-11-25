"""
How often do the most unusual people reoffend?

Scores every development row with each anomaly detector and compares the reoffending rate of the
top 5% (the most unusual people) with everyone else. This only describes the data: the detectors
are fitted on all development rows, so these scores are not used to train or evaluate a model.

Run with:
    python top5.py
"""
import numpy as np
import yaml

from src.data import load_data
from src.preprocessing import (
    AnomalyScore, add_engineered_features, build_preprocessor, clean_dataset,
    drop_duplicate_rows, split_dev_test, split_features_target,
)

TOP_SHARE = 0.05                              # the "most unusual" people: the top 5% by score
METHODS = ["isolation_forest", "lof"]


def load_development_set(config: dict):
    """Same steps as main.py up to the locked test set: returns the development rows and their target."""
    df = clean_dataset(load_data(config["data"]["path"]), config["diagnostics"])
    df = drop_duplicate_rows(df, config["diagnostics"].get("id_column"))
    if config.get("feature_engineering", {}).get("enabled", False):
        df = add_engineered_features(df, config)

    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df, config["data"], mnar_sources)
    X_dev, _, y_dev, _, _, _ = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )
    return X_dev, np.asarray(y_dev)


def main(config_path: str = "config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    X_dev, y_dev = load_development_set(config)
    Z = build_preprocessor(config["preprocessing"]).fit_transform(X_dev, y_dev)    # what the detectors see

    print(f"everyone: {y_dev.mean():.3f} reoffend ({len(y_dev)} people)")
    for method in METHODS:
        detector = AnomalyScore(method=method, n_neighbors=20, random_state=42)
        score = detector.fit_transform(Z)[:, -1]                                   # last column = anomaly score
        top = score >= np.quantile(score, 1 - TOP_SHARE)
        print(f"{method}: top {TOP_SHARE:.0%} ({top.sum()} people) {y_dev[top].mean():.3f} reoffend"
              f"  ·  the other {1 - TOP_SHARE:.0%}: {y_dev[~top].mean():.3f}")


if __name__ == "__main__":
    main()
