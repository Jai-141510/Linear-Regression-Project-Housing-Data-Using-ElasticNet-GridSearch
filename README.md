<div align="center">

# 🏡 Ames Housing — Sale Price Predictor

**A regularized regression model that predicts residential property sale prices using the Ames, Iowa housing dataset — covering 2,930 homes across 80+ features.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ElasticNet-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)

</div>

---

## 📋 Table of Contents
- [Dataset Overview](#-dataset-overview)
- [Data Cleaning Pipeline](#-data-cleaning-pipeline)
- [Model Choice & Rationale](#-model-choice--rationale)
- [Results & Interpretation](#-results--interpretation)

---

## 📦 Dataset Overview

| Property | Detail |
|---|---|
| **Source** | Ames, Iowa Housing Dataset |
| **Rows** | 2,930 residential sales |
| **Raw Features** | 81 columns (numeric + categorical) |
| **Target Variable** | `SalePrice` |
| **Price Range** | $12,789 — $755,000 |
| **Mean Sale Price** | $180,811 |

---

## 🧹 Data Cleaning Pipeline

The raw data went through a structured 3-stage cleaning process before modelling.

### Stage 1 — Outlier Removal
Extreme property values that would distort model training were identified and removed. Properties with unusually large `Gr Liv Area` relative to their sale price were dropped, bringing the dataset to a cleaner distribution.

### Stage 2 — Handling Missing Data
Several columns had severe sparsity, some over 90% missing. These were dropped entirely rather than imputed, as they would introduce noise rather than signal.

| Dropped Column | % Missing |
|---|---|
| `Pool QC` | 99.6% |
| `Misc Feature` | 96.4% |
| `Alley` | 93.2% |
| `Fence` | 80.5% |
| `Fireplace Qu` | 48.5% |

> After this stage: **2,928 rows × 76 columns**

### Stage 3 — Feature Engineering & Encoding
All remaining categorical variables were one-hot encoded, transforming the dataset from 76 raw columns to **276 model-ready features** — capturing neighbourhood effects, zoning types, sale conditions, garage types, and more without assuming ordinal relationships.

---

## 🤖 Model Choice & Rationale

### Why ElasticNet?

With 276 features and multicollinearity between many structural variables (e.g. `Total Bsmt SF` and `1st Flr SF`), standard Linear Regression overfits badly. ElasticNet was chosen because it:

- **Combines L1 (Lasso) + L2 (Ridge)** regularisation in a single model
- **L1 component** drives irrelevant features to exactly zero thus acting as automatic feature selection across 276 columns
- **L2 component** handles correlated features gracefully without arbitrarily eliminating one
- Produces a **sparse, interpretable** model suited to this high-dimensional setting

### Hyperparameter Tuning via Grid Search

A 5-fold `GridSearchCV` was used to find optimal values:

```python
param_grid = {
    'alpha'    : [0.1, 1, 5, 10, 100],   # overall regularisation strength
    'l1_ratio' : [0.1, 0.7, 0.99]        # mix between L1 and L2
}
```

> Train/Test split: **90% / 10%** — a larger training set was used intentionally to give Grid Search more data per fold.

---

## 📊 Results & Interpretation

| Metric | Value |
|---|---|
| **Mean Absolute Error (MAE)** | ~$16,249 |
| **Root Mean Squared Error (RMSE)** | ~$23,698 |
| **Dataset Mean Sale Price** | $180,811 |
| **Approximate Error Rate** | ~10% |

### What This Means

The model predicts house sale prices **within ~$16K on average**, which is roughly a **10% margin** relative to the mean price. Given the natural noise in real estate transactions (timing, buyer negotiation, local market conditions), this is a solid result from a linear model.

The RMSE being higher than the MAE indicates the model does make occasional larger errors, likely on luxury or atypical properties at the tail of the distribution, which are inherently harder to price with a linear approach.

> **Room for Improvement:** Further tuning of `alpha` and `l1_ratio`, or incorporating non-linear models (e.g. Gradient Boosting), could push error below 8%.

---

<div align="center">

*Built with Python · scikit-learn · pandas · matplotlib · seaborn*

</div>
