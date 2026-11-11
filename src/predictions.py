"""
Predicting new people (week 9, option A).

The final pipeline is refit on all development rows at the end of main.py; this module applies it to a file
that has NO target column (e.g. data/adult_new.csv) and saves one prediction per row.
Everything comes from the config's `predict` section, so a config without that section (COMPAS) skips this step.
"""
import os
import pandas as pd

from src.data import load_data
from src.preprocessing import clean_dataset, add_engineered_features, split_features_target


def predict_new_data(pipeline, config: dict) -> str:
    """
    Cleans the new file exactly like the training data (but never drops rows: every row needs a prediction),
    builds the same features, and saves `id`, `probability` (P(class 1)) and `prediction` (0/1) to a CSV.
    Returns the path of the CSV. `pipeline` must already be fitted.
    """
    predict_config = config["predict"]
    threshold = predict_config.get("threshold", 0.5)
    id_column = config["diagnostics"].get("id_column", "id")

    df_raw = load_data(predict_config["data_path"])
    df = clean_dataset(df_raw, config["diagnostics"])                  # row-preserving; no drop_duplicate_rows()
    if config.get("feature_engineering", {}).get("enabled", False):
        df = add_engineered_features(df, config)

    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X_new, _, _ = split_features_target(df, config["data"], mnar_sources)       # no target column -> y is None

    probability = pipeline.predict_proba(X_new)[:, 1]
    out = pd.DataFrame({
        id_column: df[id_column].values,
        "probability": probability,
        "prediction": (probability >= threshold).astype(int),
    })

    if len(out) != len(df_raw):
        raise ValueError(f"{len(df_raw)} rows in, {len(out)} predictions out: no row may be dropped.")
    if out[id_column].duplicated().any() or out["probability"].isna().any():
        raise ValueError("Predictions have repeated ids or missing probabilities.")

    path = predict_config["output_path"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Predictions: {len(out)} rows, {out['prediction'].mean():.1%} predicted as class 1 (threshold {threshold})")
    return path
