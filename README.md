# Laura Bernhard, 20231606 - Baseline Predictive Pipeline -- ETAI

## Overview
This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks go on.

## Project structure
```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress
| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | EDA + preprocessing -- diagnose the data, then fix it | `src/data_diagnostics.py` (missingness-mechanism test via chi-square + Cramér's V, domain-rule invalid-value detection, two-way duplicate check) and `src/preprocessing.py` (leak-safe category cleanup, mechanism-matched imputation with `_was_missing` indicators for MNAR columns, a deployable `ColumnTransformer`, **and** the train/test split itself, all in the one file rather than split across two) replace the old naive `dropna()`/`pd.get_dummies()` preprocessing; encoder/scaler pair (target encoding + standard scaling) chosen by an empirical grid over 15 repeated splits, checked against the runner-up with a paired comparison so the win isn't just noise; three redundant columns (found via correlation + VIF) dropped; `config.yaml` gains `diagnostics` and `preprocessing` sections -- see "Preprocessing decisions" below. Threshold-independent metrics (ROC-AUC/PR-AUC) and a calibration check are deliberately **not** added yet -- not yet |
| 4 | Preprocessing inside the pipeline + cross-validation -- evaluating a model honestly | A **locked final test set** (20%, stratified, seed 42) is set aside by `split_dev_test()` (replaces `split_train_test()`) and never scored; models are now judged by **stratified 5-fold cross-validation** of the whole pipeline (preprocessing + model) on the development set, reported per fold with mean ± std and the train-validation gap; the classification report and fairness check now use out-of-fold predictions; target encoding switched to scikit-learn's cross-fitting `TargetEncoder` (a row's own label never leaks into its own encoding), encoder/scaler set by hand in `config.yaml` (target encoding + robust scaling, reasons in the comments); **two fixes** in `clean_dataset()`: genuine `NaN`s in categorical columns were being turned into the string `"nan"` (a fake category), so 229 `c_charge_degree` gaps were never imputed or flagged -- fixed in `config.yaml` alone: `"nan"` added to `diagnostics.placeholder_tokens` (the category cleanup's last step turns listed tokens into `NaN`, after its text conversion); and it no longer drops rows -- de-duplication moved to a separate, training-only `drop_duplicate_rows()` (run before the dev/test split), so the same cleaning can run on new data where every row needs a prediction; `src/data_diagnostics.py` removed -- its one cleaning function (`flag_invalid_values`) moved into `preprocessing.py`, and the EDA-only checks (missingness test, duplicate counts) live in the EDA notebooks, not in every pipeline run; `dummy` (majority-class) model added as the floor to beat, and `random_forest` registered (sensible defaults, untuned); the final model is refit on the whole development set after CV; `config.yaml` gains `test_set` and `cv` sections -- see "Model evaluation" below |
| 5 | Hyperparameter tuning without leakage | New `src/tuning.py`: hyperparameters are tuned with **Optuna** (seeded TPE sampler) on the development set only -- every candidate is scored by stratified 5-fold CV of the **whole** pipeline (preprocessing re-fit in every fold of every trial), on the same folds for every candidate; the tuning procedure is estimated honestly by **nested cross-validation** (the tuning repeated inside each outer CV fold, scored on the outer fold it never saw), whose out-of-fold predictions now feed the classification report and fairness check; the reported "best trial" score is shown next to the nested one as an optimism check; the final model is the tuned pipeline refit on all development rows; `build_pipeline()` in `src/model.py` now builds the preprocessing + model `Pipeline` (one place to add a learned step later); `config.yaml` gains a `tuning` section (on/off, trials, inner folds, seed, one search space per model type) and the model becomes the decision tree; `optuna` added to `requirements.txt`. The locked test set is still untouched -- see "Hyperparameter tuning" below |
| 6 | Ensembles in depth: random forest, oblique random forest, AdaBoost | Two new model types in `src/model.py`: `oblique_random_forest` (a forest of oblique trees built from scikit-learn parts -- every tree splits on its own random sparse linear combinations of the features, via `SparseRandomProjection` inside `BaggingClassifier`) and `adaboost` (AdaBoost over decision trees of a chosen depth; stumps by default), both registered in `_MODELS`; the random forest is now the model `main.py` runs (200 trees, all CPU cores); `config.yaml` gains search spaces for the three ensembles -- the forests tune tree size (`max_depth`, `min_samples_leaf`) and decorrelation (`max_features` / number of combinations) but not the number of trees, AdaBoost tunes its base trees' depth, its number of rounds and its learning rate; `corrected_paired_test()` in `src/evaluate.py` compares two models scored on the same folds (Nadeau & Bengio's corrected paired t-test), so a "winner" is only called when the difference is bigger than fold-to-fold noise; `scipy` listed in `requirements.txt` (it already came with scikit-learn, so nothing new to install) -- see "Ensembles" below |
| 7 |	Gradient boosting + evaluating with probabilities	| Three new model types in src/model.py: gradient_boosting (scikit-learn's GradientBoostingClassifier: each shallow tree fits the gradient of the log-loss left by the trees before it), hist_gradient_boosting (the same idea on binned features: much faster on large data, and it handles missing values itself) and naive_bayes (GaussianNB, a fast probabilistic baseline); gradient boosting is now the model main.py runs, each with its search space in config.yaml; evaluation now reports several metrics (accuracy, balanced accuracy, precision, recall, F1, F-beta, ROC-AUC, PR-AUC) and one selection metric (cv.selection_metric, ROC-AUC) that tuning maximises and runs are ranked by; out-of-fold probabilities (not just labels) feed the reports, with a confusion matrix; class_weight: balanced in the commented examples of the models that support it; new save_scores() in src/results.py: every run adds a row to results/runs.csv (the model and its settings, the score on each fold, the no-model floor and the untuned score on the same folds), so any two runs on the same data and folds can be compared with corrected_paired_test(); the printed output was reorganised to be easier to read, with the full tables in the results file (optional: keep your own format if you prefer) -- see "Model evaluation" below |

## Preprocessing decisions
| Column(s) | Issue found | Mechanism | What was done |
|---|---|---|---|
| `age` | 2.0% missing | MCAR | median impute, no indicator needed |
| `juv_fel_count` | 3.0% missing | MCAR | median impute, no indicator needed |
| `priors_count` | ~7% missing (incl. placeholder tokens) | MNAR -- tied to `age_cat` | median impute + `priors_count_was_missing` flag |
| `c_charge_degree` | 3.2% missing | MNAR -- tied to `age_cat` | mode impute + `c_charge_degree_was_missing` flag |
| `race` | ~1% missing (placeholder tokens) | MCAR | mode impute, no indicator (excluded from model features anyway) |
| `sex` | ~1.5% missing (incl. placeholder tokens) | MCAR | mode impute, no indicator needed |
| `age`, `decile_score`, `juv_fel_count`, `priors_count` | invalid values (out-of-range or negative) | domain rule | converted to `NaN` before imputation |
| `sex` / `race` / `c_charge_degree` / `score_text` | inconsistent category spelling (casing, whitespace, abbreviations) | data entry | canonicalized to one spelling per category |
| whole rows | 72 exact-duplicate rows, all sharing a repeated `id` | data entry | dropped, kept first occurrence |
| `prior_offenses`, `age_in_months`, `juvenile_total` | redundant with other columns (correlation r=1.00, or -- for `juvenile_total` -- an exact sum caught only by VIF) | multicollinearity | dropped |

**Encoder/scaler pair:** chosen by hand in `config.yaml` -- **target encoding** (compact, informative) and **robust scaling** (median/IQR, so the few extreme counts don't set the scale). The alternatives (`onehot`/`ordinal`/`count`, `none`/`standard`/`minmax`) are one config change away.

## Model evaluation
| Model | Holdout accuracy | CV accuracy (mean ± std) | CV train–val gap | Nested CV accuracy (mean ± std) | Nested CV train–val gap |
|---|---|---|---|---|---|
| Dummy | — | 0.549 ± 0.000 | 0.000 ± 0.000 | — | — |
| Logistic regression | 0.680 | 0.672 ± 0.013 | +0.003 ± 0.016 | 0.671 ± 0.015 | +0.003 ± 0.016 |
| Decision tree | 0.659 | 0.646 ± 0.009 | +0.045 ± 0.019 | 0.672 ± 0.017 | +0.013 ± 0.023 |
| Random forest | — | 0.676 ± 0.014 | +0.032 ± 0.018 | 0.681 ± 0.016 | +0.009 ± 0.019 |

## Best model
| Week(s) | Dummy | Logistic Regression | Decision Tree | Random Forest | Comparison between models |
|---|---|---|---|---|---|
| 2/3 | — | Train: 0.679 <br> Test: 0.680 <br> Gap: -0.001 | Train: 0.705 <br> Test: 0.659 <br> Gap: +0.046 | — | Logistic Regression is the best model because it achieved the highest test accuracy while showing almost no train-test gap. |
| 4 | Train: 0.549 ± 0.000 <br> Validation: 0.549 ± 0.000 <br> Gap: -0.000 ± 0.000 | Train: 0.675 ± 0.003 <br> Validation: 0.672 ± 0.013 <br> Gap: +0.003 ± 0.016 | Train: 0.691 ± 0.014 <br> Validation: 0.646 ± 0.009 <br> Gap: +0.045 ± 0.019 | Train: 0.708 ± 0.005 <br> Validation: 0.676 ± 0.014 <br> Gap: +0.032 ± 0.018 | Random Forest achieved the highest mean cross-validation, but Logistic Regression achieved very similar predictive performance with substantially less overfitting (smaller train-val gap). Therefore, Logistic Regression is considered the best model so far. |
| 5 | — | Train: 0.676 ± 0.003 <br> Validation: 0.671 ± 0.015 <br> Gap: +0.005 ± 0.017 | Train: 0.685 ± 0.007 <br> Validation: 0.672 ± 0.017 <br> Gap: +0.013 ± 0.023 | Train: 0.691 ± 0.004 <br> Validation: 0.681 ± 0.016 <br> Gap: +0.009 ± 0.019 | After hyperparameter tuning, Random Forest achieved the highest nested cross-validation accuracy (0.681 ± 0.016) while maintaining a very small train-validation gap (+0.009). Decision Tree benefited substantially from tuning, while Logistic Regression showed almost no improvement. Therefore, the tuned Random Forest is selected as the final best model. |

### Hyperparameter tuning
| Model | Default: CV mean ± std | Tuned: nested CV mean ± std | Tuning score (best trial) | Optimism | Chosen hyperparameters |
|---|---|---|---|---|---|
| Decision tree | 0.610 ± 0.018 | 0.672 ± 0.017 | 0.679 | +0.002 | max_depth=6, min_samples_leaf=101, criterion=gini |
| Logistic regression | 0.672 ± 0.013 | 0.671 ± 0.015 | 0.677 | +0.004 | C=0.075, penalty=l2 |
| Random forest | 0.647 ± 0.016 | 0.681 ± 0.016 | 0.684 | -0.000 | n_estimators=58, max_depth=7, min_samples_leaf=1, criterion=entropy |

Hyperparameter tuning benefited the tree-based models the most. The decision tree improved from 0.610 ± 0.018 to 0.672 ± 0.017 (+0.062), and the random forest improved from 0.647 ± 0.016 to 0.681 ± 0.016 (+0.034). In both cases, the improvement was larger than the fold-to-fold standard deviation, suggesting a meaningful gain. In contrast, logistic regression changed from 0.672 ± 0.013 to 0.671 ± 0.015, indicating that tuning provided no practical benefit.

Tuning changed which model was best. Before tuning, logistic regression had the highest mean CV accuracy (0.672), whereas after tuning, the random forest achieved the highest nested CV accuracy (0.681). For the final report, I would report 0.681 ± 0.016, the random forest's nested cross-validation score, because it is the honest performance estimate from nested CV and is not affected by the optimistic bias of selecting the best hyperparameters on the same data used for evaluation.

### Ensembles
| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic regression | 0.672 ± 0.013 (+0.003) | — | — | — |
| Decision tree | 0.610 ± 0.018 (+0.085) | 0.672 ± 0.017 (+0.013) | 0.679 | max_depth = 6, min_samples_leaf = 101, criterion = gini |
| Random forest | 0.647 ± 0.016 | 0.679 ± 0.013 (+0.011) | 0.682 | max_depth = 11, min_samples_leaf = 47, max_features = 0.5 |
| Oblique random forest | 0.645 ± 0.013 (+0.114) | 0.682 ± 0.014 (+0.008) | 0.682 | tree max_depth = 5, tree min_samples_leaf = 3, project n_components = 9 |
| AdaBoost | 0.668 ± 0.013 (+0.003) | 0.682 ± 0.017 (+0.004) | 0.684 | estimator max_depth = 4, n_estimators = 220, learning_rate = 0.0251 |

Since the differences between the tuned models (0.671 to 0.682) are smaller than the fold-to-fold std (about 0.013 to 0.017), the lead of any single model is not meaningful, but I would choose AdaBoost: it ties the oblique random forest at 0.682 and has the best untuned score (0.668), a small gap (+0.004) and zero optimism, so its estimate is the most trustworthy. The chosen hyperparameters fit how each model works: the single decision tree needs heavy regularisation (depth 6, 101 samples per leaf) because it has high variance, the random forest can afford deeper trees (depth 11) since averaging reduces variance, the oblique forest uses shallow trees (depth 5) because each split is already a linear combination of features, and AdaBoost uses shallow weak learners (depth 4) with a low learning rate (0.025) to avoid fitting noise. I would report AdaBoost's nested CV score of 0.682 ± 0.017 rather than the 0.684 tuning score, since the latter is the best of 30 trials and therefore optimistic. Once the model is chosen, the locked test set should be evaluated once for the final number.


## Environment setup
You only need to do this once per machine.

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
