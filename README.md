# Semiconductor Manufacturing Yield Prediction Using Machine Learning

Machine learning project for predicting semiconductor manufacturing failures using sensor measurements from the UCI SECOM dataset.

## Objective

Predict whether a manufactured unit will Pass or Fail quality testing, and investigate whether a smaller subset of sensors can maintain useful predictive performance.

## Dataset

[UCI SECOM Dataset](https://archive.ics.uci.edu/dataset/179/secom)
- 1,567 production records
- 591 sensor measurements
- 6.64% failure rate (highly imbalanced)

The notebook downloads the dataset directly from the UCI repository, so no local data file is needed to reproduce results.

## Approach

1. Data cleaning — missing value analysis, dropping high-missing columns, removing zero-variance (constant) sensors
2. Exploratory data analysis — univariate, bivariate, and multivariate analysis
3. Leakage-safe preprocessing — stratified train-test split, imputation and scaling fit on training data only
4. Class imbalance handling — SMOTE applied inside cross-validation folds (via `imblearn.pipeline.Pipeline`) to avoid leakage
5. Feature selection — reduced from 446 to 100 sensors using SelectKBest
6. Model comparison and hyperparameter tuning via GridSearchCV

## Models Compared

- Logistic Regression (baseline)
- Random Forest
- Support Vector Machine (SVM)
- Gaussian Naive Bayes

## Results

| Model | Fail Recall | Fail Precision | Fail F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.19 | 0.11 | 0.14 |
| Random Forest (446 features) | 0.04 | 0.20 | 0.06 |
| Random Forest (100 features, pipeline SMOTE) | 0.12 | 0.27 | 0.16 |
| SVM (100 features, pipeline SMOTE) | 0.35 | 0.13 | 0.19 |
| **Gaussian Naive Bayes (100 features)** | **0.19** | **0.24** | **0.21** |

**Best model:** Gaussian Naive Bayes, selected on F1-score to balance failure detection against false positives. Hyperparameter tuning for the other models optimized for recall, since missing a real manufacturing failure is costly.

## Key Finding

The highly imbalanced and high-dimensional nature of this dataset makes failure detection genuinely difficult. Feature selection and a leakage-safe SMOTE pipeline improved results somewhat, but the models still struggled to detect a large share of actual failures — reflecting the difficulty of predicting rare events from hundreds of noisy sensor readings, rather than a flaw in the pipeline itself.

## Future Improvements

- Advanced resampling (ADASYN, SMOTE-Tomek)
- PCA-based dimensionality reduction
- Ensemble/boosting methods (XGBoost, LightGBM)
- Anomaly detection framing (One-Class SVM)
- Threshold tuning on predicted probabilities
- Cost-sensitive learning

## Repository Structure

```
semiconductor-manufacturing-yield-prediction/
│
├── README.md
├── semiconductor_yield_prediction.ipynb
│
└── models/
    ├── best_model_naive_bayes.pkl
    ├── scaler.pkl
    └── feature_selector.pkl
```

## Tech Stack

Python · pandas · scikit-learn · imbalanced-learn · seaborn · matplotlib

