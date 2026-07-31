#  Week 3 — Diamond Price Prediction (Regression)

## Overview

This module builds an end-to-end diamond price prediction pipeline using Scikit-learn, Pandas, Seaborn, and Matplotlib, applying machine learning fundamentals and regression analysis to model what drives the market price of a diamond and to quantify how well price can be predicted from its physical and quality attributes.

## Dataset

Diamonds Dataset from Seaborn, containing 53,940 diamonds with attributes such as carat, cut, color, clarity, depth, table, price, and the three physical dimensions x, y, z (mm).

Source: `sns.load_dataset('diamonds')`

## Structure

- **Stage 1** — Cleaning and EDA: missing-value audit, removal of physically impossible rows (zero dimensions) and extreme dimension outliers, a correlation heatmap, the price distribution vs `log(price)`, and a carat-vs-price scatter coloured by cut.
- **Stage 2** — Linear regression with a Scikit-learn `Pipeline` (`StandardScaler` + `LinearRegression`) on the six numeric features, split 80/20 (`random_state=42`) **before** fitting to prevent leakage, evaluated with MAE/RMSE/R² on both train and test, plus residual and coefficient analysis.
- **Stage 3** — Polynomial regression of price on `carat` for degrees 1, 2, 3, and 5, with a train/test comparison table and fitted curves overlaid on a 500-point sample.
- **Stage 4** — Model comparison table and three business-facing insights written for a non-technical decision-maker.
- **Bonus 1** — Log-transformed target pipeline (predict `log(price)`, back-transform with `np.exp()`) and RMSE/MAE comparison against the original model.
- **Bonus 2** — `ColumnTransformer` combining `StandardScaler` on numeric features and `OneHotEncoder` on cut/color/clarity inside a single pipeline, with the percentage RMSE improvement.
- **Bonus 3** — 5-fold cross-validation (`cross_val_score` with shuffled `KFold`) and a `learning_curve` (train/validation R² vs training size) to diagnose bias vs variance and the value of more data.

## Key Insights

1. **Size dominates price** — `carat` alone correlates with price at r = 0.92, and a degree-5 polynomial on carat by itself (test R² = 0.873) slightly outperforms the full six-feature linear model (0.866), confirming carat is the single strongest price driver.
2. **The price-carat relationship is non-linear** — price accelerates with carat, so a 2-carat diamond is worth far more than two 1-carat stones; polynomial terms are needed to capture the upward curve.
3. **Quality grades are worth real money** — adding cut, color, and clarity via one-hot encoding cuts test RMSE by 25% (\$1,454 → \$1,090) and lifts R² from 0.866 to 0.925, the single biggest improvement in the notebook.
4. **Log-transforming the target sharpens typical-case accuracy** — predicting `log(price)` improves MAE by ~9% (\$874 → \$792) by optimising relative rather than absolute error, appropriate for a positive, right-skewed target.
5. **The numeric model is stable but bias-limited** — 5-fold CV gives R² = 0.861 ± 0.004, and the learning curve shows train and validation R² converging at ~0.86 with a near-zero gap, meaning more data would not help; richer features (the quality grades) are the path forward.

## Files

- `GDGOC_Week_3.ipynb` — full Jupyter Notebook with runnable code, outputs, and markdown narrative.
