# Churn-Prediction [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

It is a machine learning model to predict customer churn in Indian fast e-commerce. 
Identifies at-risk customers for targeted retention campaigns.

## Problem Statement
Fast e-commerce platforms lose ~27% of repeat customers. 
This model predicts which customers will churn in 60+ days, enabling cost-effective retention offers.

## Simulated Dataset
- **Records:** 4,500 customers
- **Features:** 19 (demographics, purchase behavior, engagement)
- **Target:** Churn (binary: churned/active)
- **Churn Rate:** 27.49%

## Project Structure
```
src/
├── preprocessing.py  - Data cleaning
├── models.py        - Model training
└── evaluation.py    - Metrics & business impact
```

## Key Findings
1. Days since purchase is strongest churn predictor (+0.27 correlation)
2. Subscription status protects against churn (-0.20 correlation)

### Tech Stack
- Python 3.x
- Pandas, NumPy (data manipulation)
- Scikit-learn (machine learning)
- Matplotlib, Seaborn (visualization)
