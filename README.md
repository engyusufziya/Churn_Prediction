# Banking Customer Churn Prediction

A comprehensive machine learning solution for predicting customer churn in banking/credit card services, featuring advanced feature engineering, model comparison, hyperparameter optimization, and segment-based threshold strategies.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Feature Engineering](#feature-engineering)
- [Evaluation Strategy](#evaluation-strategy)
- [Output](#output)
- [Performance](#performance)
- [Configuration](#configuration)

## Overview

This project implements a complete end-to-end machine learning pipeline for predicting customer churn (T+3 months) using banking transaction and customer data. The solution includes sophisticated feature engineering, multiple model comparison, hyperparameter tuning with Optuna, and business-oriented threshold optimization.

## Features

### Core Functionality
- **Multi-Model Comparison**: XGBoost, LightGBM, Random Forest, Logistic Regression
- **Automated Hyperparameter Tuning**: Optuna-based optimization for XGBoost
- **Advanced Feature Engineering**: 20+ derived features from transaction patterns
- **Imbalanced Data Handling**: SMOTE oversampling for training
- **Segment-Based Thresholds**: Different decision thresholds for high-value vs regular customers
- **Campaign Type Assignment**: Automatic risk-based campaign categorization

### Business Intelligence
- **Hybrid Threshold Strategy**: Optimized thresholds based on customer value segments
- **Campaign Recommendations**: Risk-based customer intervention strategies
- **Performance Metrics**: Comprehensive evaluation including precision, recall, F1, AUC
- **Feature Importance Analysis**: Model interpretability for business insights

## Installation

### Requirements
```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn joblib optuna matplotlib
```

### Google Colab Setup
```python
!pip -q install xgboost lightgbm imbalanced-learn joblib optuna --upgrade
```

## Data Requirements

### Primary Dataset (CHURN_DATA.csv)
Required columns:
- `T+3_CHURN`: Target variable (0/1)
- `KART_GIRIS_TARIHI`: Card registration date (DD.MM.YYYY)
- `LAST_TXN_DT`: Last transaction date (DD.MM.YYYY)
- `FRST_TXN_DT`: First transaction date (DD.MM.YYYY)
- `TXN_ADET_SON_*`: Transaction counts (1 month, 3 months, 6 months, 1 year)
- `TXN_TUTAR_SON_*`: Transaction amounts (1 month, 3 months, 6 months, 1 year)
- `LIMIT`: Credit limit
- `KULLANILABILIR_LIMIT`: Available credit limit
- `KART_STATUSU`: Card status
- `AKTIFLIK_DURUM`: Activity status

### Optional Dataset (BOLGE.csv)
For regional analysis:
- `SUBE_KODU`: Branch code
- `BOLGE`: Region
- `IL_ADI`: City name

## Usage

### Basic Usage
```python
from your_module import BankingChurnPredictor

# Initialize predictor
predictor = BankingChurnPredictor()

# Load and prepare data
df = predictor.load_data("CHURN_DATA.csv")
df_fe = predictor.feature_engineering(df)
X = predictor.prepare_features(df_fe)
y = df_fe['T+3_CHURN'].astype(int).values

# Train models
results = predictor.train_models(X, y, use_smote=True)

# Optimize best model
best_params = predictor.tune_xgb_optuna(X, y, n_trials=50)

# Make predictions
predictions, probabilities = predictor.predict(new_data, "XGBoost")
```

### Advanced Configuration
```python
# Set custom threshold
predictor.set_threshold(0.45)

# Apply hybrid thresholds
high_value_mask = (df['LIMIT'] > 20000) | (df['TXN_TUTAR_SON_6AY'] > 10000)
hybrid_predictions = BankingChurnPredictor.apply_hybrid_thresholds(
    df, 
    score_col="churn_score",
    high_value_mask=high_value_mask,
    t_hv=0.35,  # High-value customer threshold
    t_lv=0.50   # Regular customer threshold
)
```

## Model Architecture

### Supported Models
1. **XGBoost Classifier**
   - Gradient boosting with advanced regularization
   - Hyperparameter optimization via Optuna
   - Best for structured/tabular data

2. **LightGBM Classifier**
   - Fast gradient boosting framework
   - Memory efficient
   - Good baseline performance

3. **Random Forest Classifier**
   - Ensemble of decision trees
   - Robust to overfitting
   - Feature importance insights

4. **Logistic Regression**
   - Linear baseline model
   - Fast training and inference
   - Interpretable coefficients

### Model Selection Criteria
- **Primary Metric**: ROC-AUC score
- **Secondary Metrics**: Precision-Recall AUC, F1-score
- **Cross-Validation**: 5-fold stratified CV

## Feature Engineering

### Temporal Features
- `CARD_AGE_MONTHS`: Age of card in months
- `DAYS_SINCE_LAST_TXN`: Days since last transaction

### Credit Utilization
- `LIMIT_UTILIZATION`: Credit utilization ratio
- `KULLANILABILIR_LIMIT`: Available credit analysis

### Transaction Trends
- `TXN_TREND_3M_6M`: 3-month vs 6-month transaction trend
- `TXN_TREND_1M_3M`: 1-month vs 3-month transaction trend
- `AMOUNT_TREND_*`: Amount-based trend indicators

### Behavioral Patterns
- `AVG_TXN_AMOUNT_*`: Average transaction amounts
- `TXN_DIVERSITY`: Transaction type diversity score
- `DECLINING_ACTIVITY_RISK`: Risk score based on declining activity

### Categorical Encoding
- Label encoding for categorical variables
- Missing value indicators
- Regional information integration

## Evaluation Strategy

### Metrics
- **ROC-AUC**: Primary model selection metric
- **Precision-Recall AUC**: Imbalanced data performance
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification results
- **False Positive/Negative Counts**: Business impact analysis

### Cross-Validation
- 5-fold Stratified Cross-Validation
- Maintains class distribution across folds
- Robust performance estimation

### Threshold Optimization
- Global threshold optimization
- Segment-based threshold strategy
- Business-oriented decision boundaries

## Output

### Prediction Files
1. **churn_scored_segmented.csv**
   - Customer IDs and card numbers
   - Churn probability scores
   - Global and segmented predictions
   - Campaign type recommendations

2. **churn_scored_hybrid.csv**
   - Alternative output format
   - Hybrid threshold predictions
   - Risk-based campaign assignments

### Campaign Types
- **"Güçlü Teşvik"**: High-risk customers (score > 0.60)
- **"Düşük Maliyetli"**: Medium-risk customers (0.38 ≤ score ≤ 0.60)
- **"Yok"**: Low-risk customers (score < 0.38)

## Performance

### Typical Results
- **ROC-AUC**: 0.70-0.75 on test set
- **Precision**: 0.45-0.55 (depending on threshold)
- **Recall**: 0.50-0.60
- **F1-Score**: 0.47-0.57

### Model Comparison
Based on typical performance:
1. **Random Forest**: Best overall AUC performance
2. **XGBoost**: Strong performance, highly tunable
3. **LightGBM**: Fast training, good baseline
4. **Logistic Regression**: Simple baseline, interpretable

## Configuration

### Hyperparameter Ranges (Optuna)
```python
{
    "n_estimators": [200, 800],
    "max_depth": [3, 10],
    "learning_rate": [0.01, 0.2],
    "subsample": [0.6, 1.0],
    "colsample_bytree": [0.6, 1.0],
    "min_child_weight": [1.0, 10.0],
    "reg_lambda": [0.0, 10.0],
    "reg_alpha": [0.0, 5.0]
}
```

### Threshold Configuration
```python
# Segment-based thresholds
HIGH_VALUE_THRESHOLD = 0.35  # More sensitive for valuable customers
REGULAR_THRESHOLD = 0.50     # Conservative for regular customers

# High-value customer definition
high_value_criteria = (
    (customer_limit > 20000) | 
    (transaction_amount_6m > 10000)
)
```

### Business Rules
- **High-Value Customers**: Lower threshold (0.35) for early intervention
- **Regular Customers**: Higher threshold (0.50) to reduce false positives
- **Campaign Assignment**: Automatic based on risk scores
- **Feature Selection**: Domain-knowledge driven feature engineering

## Contributing

To extend this project:
1. Add new feature engineering functions to `feature_engineering()`
2. Implement additional models in `train_models()`
3. Extend threshold strategies in `apply_hybrid_thresholds()`
4. Add new evaluation metrics to `metrics_from_preds()`

## License

This project is designed for banking and financial services customer retention analysis. Ensure compliance with data privacy regulations and internal policies when handling customer data.
