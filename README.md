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

## Key Findings
1. Days since purchase is strongest churn predictor (+0.27 correlation)
2. Subscription status protects against churn (-0.20 correlation)

### Tech Stack
- Python 3.x
- Pandas, NumPy (data manipulation)
- Scikit-learn (machine learning)
- Matplotlib, Seaborn (visualization)

## 💬 Chat Interface
Ask questions about a customer in plain English and get their churn probability, e.g.

> *What is the probability of churn if days since purchase is 123, purchase frequency per month is 2.21, return frequency is 0.26 and subscription status is regular?*

The bot replies with the probability, a risk level, which details are driving it, and which features were assumed.
Follow-ups like *"what if they had no subscription?"* keep the earlier details; say *"new customer"* to start over.

### Model used by the app
`python churn_model.py` compares the notebook's models with improved candidates using 5-fold
cross-validation and writes [`MODEL_REPORT.md`](MODEL_REPORT.md). The selected model is a
**Logistic Regression with extra recency features** (inactive 90+ days, and inactivity combined
with having no subscription). These fix the notebook models' under-prediction for long-inactive customers.
- **Calibrated:** predicted churn rates match actual rates in every recency × subscription segment.
- **Tuned for recall:** customers at or above a **12%** churn probability are flagged for retention.
  This threshold maximises F2 (recall weighted 2×). On held-out data it catches **163 of 192 churners
  (85% recall, 29 missed)**, compared with 49 caught (143 missed) by the notebook's Random Forest at 0.5.
  The trade-off is lower precision (29%), so more customers get offers. The campaign ROI is still
  positive (+16%) under the notebook's cost assumptions.

| File | Purpose |
|------|---------|
| `churn_model.py` | Feature engineering, model comparison, threshold tuning, training and prediction |
| `question_parser.py` | Extracts customer features from a free-text question (no API key needed) |
| `app.py` | Gradio chat UI |
| `MODEL_REPORT.md` | Model comparison, calibration table and test-set results |

### Run locally
```bash
pip install -r requirements.txt
python app.py          # open http://localhost:7860
```
The model is trained from `ecommerce_churn_dataset.csv` on first start (a few seconds) and cached in `models/`.

### Deploy for free on Hugging Face Spaces
1. Create a new Space at https://huggingface.co/new-space and pick the **Gradio** SDK.
2. Upload `app.py`, `churn_model.py`, `question_parser.py`, `requirements.txt` and `ecommerce_churn_dataset.csv` (or push this repo to the Space's git remote).
3. The Space builds and gives you a public URL for the chat.

It also runs on any host that runs Python (Render, Railway, a VM): start command `python app.py`; the `PORT` environment variable is respected.
