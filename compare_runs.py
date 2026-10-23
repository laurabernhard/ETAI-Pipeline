"""
Compare the models in the scoreboard with the corrected paired t-test.

Everything comes from config.yaml (cv.selection_metric, cv.n_splits, the scoreboard path),
so changing the metric or the number of folds there is enough -- no code changes.

Usage:
    python compare_runs.py                   # best model vs all the others
    python compare_runs.py --best adaboost   # choose the reference model yourself
"""
import argparse
import pandas as pd
import yaml

from src.evaluate import corrected_paired_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--best", default=None, help="reference model (default: highest mean score)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    metric = config["cv"]["selection_metric"]
    k = config["cv"]["n_splits"]
    output = config.get("output", {})
    path = output.get("scoreboard") or output.get("score_board") or "results/runs.csv"

    runs = pd.read_csv(path, dtype={"run": str})
    runs = runs[runs["metric"] == metric]                    # only runs scored on this metric
    runs = runs.sort_values("run").drop_duplicates("model", keep="last")   # latest run of each model

    fold_cols = [f"fold_{i}" for i in range(1, k + 1)]
    scores = runs.set_index("model")[fold_cols]
    means = scores.mean(axis=1)
    best = args.best or means.idxmax()
    ratio = 1 / (k - 1)                                      # validation rows / training rows

    rows = []
    for model in means.sort_values(ascending=False).index:
        if model == best:
            continue
        r = corrected_paired_test(scores.loc[best], scores.loc[model], ratio)
        rows.append({
            "model": model,
            f"mean {metric}": f"{means[model]:.3f}",
            "diff (ref - model)": f"{r['mean_diff']:+.4f}",
            "folds won by ref": r["wins"],
            "t": f"{r['t']:.2f}",
            "p-value": "<0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.3f}",
            "p < 0.05": "yes" if r["p_value"] < 0.05 else "no",
        })

    print(f"Corrected paired t-test | metric: {metric} | {k}-fold CV")
    print(f"Reference model: {best} (mean {metric} = {means[best]:.3f})\n")
    print(pd.DataFrame(rows).to_string(index=False, justify="center"))


if __name__ == "__main__":
    main()