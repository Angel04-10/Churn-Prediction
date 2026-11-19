"""Training and inference for the churn model built in Churn_Prediction.ipynb.

The preprocessing mirrors the notebook's "Baseline Models" section exactly:
same engineered features, same one-hot encoding, same 80/20 stratified split
and the same Random Forest / Logistic Regression hyper-parameters.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "ecommerce_churn_dataset.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"

# Raw (pre-encoding) inputs a user can describe in a question.
RAW_FEATURES = [
    "Age",
    "Gender",
    "Metro",
    "Account_Age_Months",
    "Days_Since_Purchase",
    "Purchase_Frequency_Per_Month",
    "Avg_Purchase_Amount_INR",
    "Return_Frequency",
    "Review_Rating",
    "Customer_Service_Interactions",
    "Subscription_Status",
    "Discount_Sensitivity",
    "Days_Since_Last_Discount",
    "Mobile_App_User",
]

DROP_COLS = [
    "CustomerID",
    "Last_Purchase_Date",
    "Churn",
    "Location",
    "Category_Preference",
    "Purchase_Recency_Group",
]


def _engineer(df: pd.DataFrame, return_median: float) -> pd.DataFrame:
    """Apply the notebook's feature engineering + one-hot encoding."""
    out = df.copy()
    out["Has_Subscription"] = (out["Subscription_Status"] != "None").astype(int)
    out["Is_Recently_Active"] = (out["Days_Since_Purchase"] < 30).astype(int)
    out["High_Recency_Gap"] = (out["Days_Since_Purchase"] > 60).astype(int)
    out["High_Return_Rate"] = (out["Return_Frequency"] > return_median).astype(int)
    # Explicit dummies so a single-row frame encodes identically to training
    # (pd.get_dummies(drop_first=True) drops Gender=Female and Subscription=None).
    out["Gender_Male"] = (out["Gender"] == "Male").astype(int)
    out["Subscription_Status_Premium"] = (out["Subscription_Status"] == "Premium").astype(int)
    out["Subscription_Status_Regular"] = (out["Subscription_Status"] == "Regular").astype(int)
    out = out.drop(columns=["Gender", "Subscription_Status"])
    return out.drop(columns=[c for c in DROP_COLS if c in out.columns])


def train(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> dict:
    df = pd.read_csv(data_path, keep_default_na=False, na_values=[""])
    return_median = float(df["Return_Frequency"].median())

    X = _engineer(df, return_median)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    lr = LogisticRegression(random_state=42, max_iter=1000).fit(X_train_scaled, y_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10).fit(
        X_train, y_train
    )

    metrics = {
        "Random Forest": roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]),
        "Logistic Regression": roc_auc_score(
            y_test, lr.predict_proba(scaler.transform(X_test))[:, 1]
        ),
    }

    # Values used for any feature the user does not mention in their question.
    defaults = {}
    for col in RAW_FEATURES:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s) or s.nunique() <= 2:
            m = s.mode().iloc[0]
            defaults[col] = m.item() if hasattr(m, "item") else str(m)
        else:
            defaults[col] = float(s.median())
    for col in ["Age", "Account_Age_Months", "Days_Since_Purchase",
                "Customer_Service_Interactions", "Days_Since_Last_Discount"]:
        defaults[col] = int(round(defaults[col]))

    ranges = {
        col: (float(df[col].min()), float(df[col].max()))
        for col in RAW_FEATURES
        if pd.api.types.is_numeric_dtype(df[col])
    }

    bundle = {
        "rf": rf,
        "lr": lr,
        "scaler": scaler,
        "columns": list(X.columns),
        "return_median": return_median,
        "defaults": defaults,
        "ranges": ranges,
        "metrics": metrics,
        "base_churn_rate": float(y.mean()),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


def load(model_path: Path = MODEL_PATH) -> dict:
    """Load the trained bundle, training it first if it doesn't exist yet."""
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception:
            pass  # e.g. pickled with a different scikit-learn version: retrain
    return train(model_path=model_path)


def predict(bundle: dict, customer: dict, model: str = "Random Forest") -> float:
    """Return the churn probability (0-1) for one customer.

    `customer` holds any subset of RAW_FEATURES; missing ones use dataset defaults.
    """
    row = {**bundle["defaults"], **customer}
    X = _engineer(pd.DataFrame([row]), bundle["return_median"])[bundle["columns"]]
    if model == "Logistic Regression":
        return float(bundle["lr"].predict_proba(bundle["scaler"].transform(X))[0, 1])
    return float(bundle["rf"].predict_proba(X)[0, 1])


if __name__ == "__main__":
    b = train()
    print(f"Saved model to {MODEL_PATH}")
    for name, auc in b["metrics"].items():
        print(f"  {name:20s} test ROC-AUC = {auc:.4f}")
    print("Defaults for unspecified features:", b["defaults"])
