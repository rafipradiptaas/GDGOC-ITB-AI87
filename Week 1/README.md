#  Week 1 — Exploratory Data Analysis (EDA)

## Overview

This module performs a complete exploratory data analysis on the Spotify Songs Dataset (TidyTuesday 2020) using NumPy and Pandas, while integrating AI coding tools as part of the workflow.

## Dataset

Spotify Songs Dataset from TidyTuesday 2020, containing audio features for 32,833 songs across six playlist genres (edm, latin, pop, r&b, rap, rock).

Source: `https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv`

## Structure

- **Stage 1** — Setup, initial inspection, missing value handling, deduplication, and manual IQR summary statistics.
- **Stage 2** — Genre and popularity analysis, top artist analysis with Pearson correlation, and hit-potential filtering.
- **Stage 3** — Pure NumPy analysis: vectorized min-max normalization, correlation matrix, and boolean masking.
- **Stage 4** — Docstring generation via AI tooling (Antigravity) and non-technical key insights.
- **Bonus 1** — Artist audio fingerprint and cosine similarity implemented from its mathematical definition.
- **Bonus 2** — Genre cluster profile using centroid distance matrix computed via NumPy broadcasting.
- **Bonus 3** — Reusable `SpotifyAnalyzer` class with lazy evaluation and JSON-safe report generation.

## Key Insights

1. Releasing more songs does not mean writing more hits — prolific artists tend to have lower average popularity per track.
2. Latin and pop dominate the hit-potential profile — nearly 60% of tracks matching a "hit" audio signature come from these two genres.
3. The most intense songs are not the most popular ones — the top 17% energy tracks average 6 popularity points below the overall mean.

## Files

- `GDGOC_Week_1.ipynb` — full Jupyter Notebook with runnable code, outputs, and markdown narrative.
- `GDGOC_Week_1.pdf` — exported PDF report of the notebook.
