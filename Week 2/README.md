#  Week 2 — Vehicle Fuel Efficiency (MPG) Analysis

## Overview

This module performs a comprehensive analysis of the vehicle fuel efficiency (MPG) dataset using NumPy, Pandas, Seaborn, and Matplotlib, applying statistics, linear algebra, EDA, and visualisation concepts to answer concrete engineering questions about the factors that influence fuel efficiency.

## Dataset

MPG Dataset from Seaborn, containing 398 vehicles with attributes such as mpg, cylinders, displacement, horsepower, weight, acceleration, model year, origin, and name.

Source: `sns.load_dataset('mpg')`

## Structure

- **Stage 1** — Inspection and cleaning: shape/dtypes/missing value audit, justified handling of missing `horsepower` values, and a manual NumPy summary table (mean, median, std, IQR, IQR×1.5 outlier count).
- **Stage 2** — Statistical analysis with NumPy: vectorized z-score standardisation of the five core numeric features, correlation matrix analysis, and boolean-masking test of the horsepower–weight relationship.
- **Stage 3** — Visualisation: MPG distribution (histogram + KDE with mean/median lines), origin comparison (violin plot), weight-vs-mpg scatter with a manual `np.polyfit()` trend line, and a correlation heatmap.
- **Stage 4** — Contextual interpretation: identifying the strongest predictor of fuel efficiency, and a mean-mpg-per-decade trend analysis.
- **Bonus 1** — Linear Regression via the Normal Equation implemented from scratch (`np.linalg.inv`), validated against `np.polyfit()` and scored with RMSE.
- **Bonus 2** — 2D PCA implemented from scratch (mean-centring, covariance matrix, `np.linalg.eigh()`, projection, explained variance ratio).
- **Bonus 3** — Reusable `DatasetProfiler` class (`summary_stats`, `correlation_report`, `plot_dashboard`, `generate_report`) demonstrated on the MPG dataset.

## Key Insights

1. **Weight is the dominant predictor of fuel efficiency** — it has the strongest correlation with mpg (r = -0.83) of any core feature, and cars with above-average horsepower weigh ~838 lbs more than the dataset average, tying engine power back to mass as the real driver of fuel consumption.
2. **Displacement, horsepower, and weight form a multicollinear cluster** — all three are strongly correlated with each other (r ≥ 0.83) and with mpg, meaning they carry largely redundant information for modelling purposes.
3. **Fuel efficiency rose steadily from the 1970s onward**, consistent with the 1973/1979 oil crises and the introduction of CAFE fuel-economy standards in 1978, which pushed manufacturers toward lighter, more efficient vehicles.
4. **The Normal Equation exactly reproduces `np.polyfit()`** (slope ≈ -0.00765, intercept ≈ 46.22, RMSE ≈ 4.32 mpg), confirming the from-scratch linear regression implementation is correct.
5. **PCA on mean-centred (unscaled) features is dominated by `weight`'s raw scale** (PC1 ≈ 99.8% explained variance), but still cleanly separates USA cars from Japan/Europe cars along that axis.

## Files

- `GDGOC_Week_2.ipynb` — full Jupyter Notebook with runnable code, outputs, and markdown narrative.
