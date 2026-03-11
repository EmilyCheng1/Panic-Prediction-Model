# Panic Prediction Model

A machine learning classification model to predict panic events from alert data using the **Classification Modeling Skill** workflow.

## Overview

This project builds a binary classifier to predict panic events (label: 0 = no panic, 1 = panic) using features extracted from alert data. The workflow compares multiple ML algorithms and selects the best performing model.

## Files

| File | Description |
|------|-------------|
| `Panic_Classification.ipynb` | Main notebook with full ML pipeline |
| `classifier_xgboost_*.joblib` | Trained XGBoost model (best performer) |
| `scaler_*.joblib` | StandardScaler for feature normalization |
| `feature_names_*.txt` | List of feature names used by the model |
| `model_comparison_*.csv` | Performance metrics for all trained models |

## ML Pipeline

1. **Data Preparation** - Load and clean data, handle missing values
2. **Feature Engineering** - Encode categorical variables, scale features
3. **Train/Test Split** - 80/20 stratified split
4. **Class Imbalance** - SMOTE oversampling (if needed)
5. **Model Training** - Train 8 classifiers:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - K-Nearest Neighbors
   - Naive Bayes
   - AdaBoost
   - XGBoost
6. **Evaluation** - Accuracy, Precision, Recall, F1-Score, ROC-AUC
7. **Hyperparameter Tuning** - RandomizedSearchCV on best model
8. **Model Persistence** - Save model artifacts for deployment

## Results

The best model (XGBoost) achieved:
- See `model_comparison_*.csv` for detailed metrics

## Usage

### Making Predictions

```python
import joblib
import pandas as pd

# Load artifacts
model = joblib.load('classifier_xgboost_20260311_151845.joblib')
scaler = joblib.load('scaler_20260311_151845.joblib')

# Load new data
new_data = pd.read_csv('new_alerts.csv')

# Scale and predict
X_scaled = scaler.transform(new_data)
predictions = model.predict(X_scaled)
```

## Requirements

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
joblib
```

## License

MIT
