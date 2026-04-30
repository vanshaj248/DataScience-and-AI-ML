# HousingPrice

A regression project using the California housing dataset to compare Linear Regression and Random Forest models.

## Overview

This project demonstrates:
- Exploratory data analysis with correlation heatmaps and target distribution
- Feature scaling using `StandardScaler`
- Linear Regression and Random Forest model training
- Evaluation using RMSE and R²
- Visualization of feature importance, residuals, and actual vs predicted values

## Files

- `test.py` — main script that runs data loading, preprocessing, model training, evaluation, and visualization
- `correlation_heatmap.png` — heatmap showing feature correlations
- `price_distribution.png` — histogram of median house values
- `feature_importance.png` — Random Forest feature importance bar chart
- `actual_vs_predicted.png` — scatter plot of predicted vs actual values
- `residuals.png` — residual distribution histogram

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Run

```bash
python test.py
```

This script will produce plots in the current directory and print model performance metrics for both Linear Regression and Random Forest.

## Notes

- The Random Forest model is trained on raw features, while the Linear Regression model is trained on scaled features.
- The dataset is loaded from `sklearn.datasets.fetch_california_housing`.
