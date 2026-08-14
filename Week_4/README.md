#  Week 4 — Titanic Classification (Supervised Learning)

## Overview

This module builds an end-to-end supervised classification pipeline on the Titanic dataset using Scikit-learn, Pandas, Seaborn, and Matplotlib, applying classification algorithms and model-evaluation techniques to predict passenger survival and reason about which model and decision threshold are right once the real-world cost of a wrong prediction is taken into account.

## Dataset

Titanic dataset from Seaborn, containing 891 passengers with attributes such as class, sex, age, family size, fare, and port of embarkation.

Source: `sns.load_dataset('titanic')`

## Structure

- **Stage 1** — Data preparation and EDA: missing-value audit, justified handling of `deck`/`embarked`/`age` (grouped-median imputation), removal of leaking/redundant columns (`alive`, `class`, `embark_town`, `who`, `adult_male`, `alone`), survival-rate visualisations by sex and class, a class-balance check, and categorical encoding.
- **Stage 2** — Logistic Regression in a `Pipeline` (`StandardScaler` + `LogisticRegression`) on an 80/20 stratified split, evaluated with a classification report, confusion matrix heatmap, and standardised coefficients converted to odds ratios.
- **Stage 3** — A K for K in `[1, 3, 5, 7, 15, 25]` sweep and a `max_depth` for `[2, 4, 6, None]` decision-tree sweep to diagnose the bias-variance tradeoff and overfitting, a visualised best tree, and a Logistic Regression vs KNN vs Decision Tree comparison table.
- **Stage 4** — ROC curves for all three models on one figure, a business-framed model recommendation under a false-negative-costly rescue-prioritisation scenario, and three plain-language insights for a non-technical museum-exhibit audience.
- **Bonus 1** — Precision-recall threshold tuning on the Logistic Regression model: the F1-optimal threshold, the lowest threshold achieving 85% recall, and a three-way confusion-matrix comparison.
- **Bonus 2** — A `ColumnTransformer` rebuild treating `pclass`, `sex`, and `embarked` as one-hot categorical features, compared against the Stage 2 numeric-`pclass` model.
- **Bonus 3** — 5-fold `StratifiedKFold` cross-validation (F1 and ROC-AUC) for the three Stage 3 models, with mean/std stability metrics.

## Key Insights

1. **Sex is the strongest single predictor** — women survived at 74.0% versus 18.9% for men (odds ratio 3.50 in the Logistic Regression model), reflecting the "women and children first" evacuation pattern.
2. **Class and sex interact rather than acting independently** — a 1st/2nd-class woman had a 92–97% survival chance, but a 3rd-class woman's chances fell to 50%, while men stayed low (14–37%) across every class.
3. **Logistic Regression edges out KNN (best K=15) and a Decision Tree (best depth=4) on the single 80/20 split** (accuracy 0.826, F1 0.756, AUC 0.859), producing the fewest false negatives (20 of 68 actual survivors missed) — the failure mode that matters most under the rescue-prioritisation cost framing.
4. **Threshold tuning nearly halves missed cases** — moving the Logistic Regression decision threshold from 0.5 to the F1-optimal / 85%-recall point of 0.312 cuts false negatives from 20 to 10, at the cost of 16 more false positives.
5. **5-fold stratified cross-validation reverses the single-split ranking** — the Decision Tree leads on CV F1 (0.756) and ties KNN for the lead on CV AUC (0.860 vs 0.861), a reminder that a single 178-row test split is not enough to declare a definitive winner.

## Files

- `titanic_classification.ipynb` — full Jupyter Notebook with runnable code, outputs, and markdown narrative.
