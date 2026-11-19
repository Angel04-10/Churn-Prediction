"""Training, model selection and inference for the churn model.

Starts from the preprocessing in Churn_Prediction.ipynb ("Baseline Models") and
improves it in three ways:

1. Extra recency features. Churn jumps at 60 *and* again at 90 days without a
   purchase, and a subscription halves that risk. The notebook only flagged
   "> 60 days", so every model under-predicted long-inactive customers.
2. Model selection by 5-fold cross-validation on the training split, scored by
   average precision (how well churners are ranked to the top) and Brier score
   (how honest the probabilities are). The test split is only used to report.
3. A decision threshold tuned for recall (maximum F2 score on out-of-fold
   predictions), so the model flags most customers who will actually churn,
   while the probability shown to the user stays calibrated.

Run `python churn_model.py` to re-run the comparison and write MODEL_REPORT.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "ecommerce_churn_dataset.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
REPORT_PATH = ROOT / "MODEL_REPORT.md"

# Winner of the cross-validated comparison (see MODEL_REPORT.md). The app trains this
# directly so it starts quickly; `python churn_model.py` re-checks the choice.
BEST_MODEL = "Logistic Regression"

# Business assumptions from the notebook's ROI section.
RETENTION_COST = 500
CUSTOMER_CLV = 10000
RECOVERY_RATE = 0.20

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


def engineer(df: pd.DataFrame, return_median: float, extended: bool = True) -> pd.DataFrame:
    """Notebook feature engineering (+ the extra recency features when `extended`)."""
    out = df.copy()
    days = out["Days_Since_Purchase"]
    out["Has_Subscription"] = (out["Subscription_Status"] != "None").astype(int)
    out["Is_Recently_Active"] = (days < 30).astype(int)
    out["High_Recency_Gap"] = (days > 60).astype(int)
    out["High_Return_Rate"] = (out["Return_Frequency"] > return_median).astype(int)
    if extended:
        no_sub = 1 - out["Has_Subscription"]
        out["Very_High_Recency_Gap"] = (days > 90).astype(int)
        out["Recency_Bucket"] = np.select([days > 90, days > 60], [2, 1], 0)
        out["Gap60_No_Subscription"] = out["High_Recency_Gap"] * no_sub
        out["Gap90_No_Subscription"] = out["Very_High_Recency_Gap"] * no_sub
        out["Returns_No_Subscription"] = out["Return_Frequency"] * no_sub
        out["High_Rating"] = (out["Review_Rating"] >= 4.5).astype(int)
        out["Low_Rating"] = (out["Review_Rating"] < 2.5).astype(int)
    # Explicit dummies so a single-row frame encodes identically to training
    # (pd.get_dummies(drop_first=True) drops Gender=Female and Subscription=None).
    out["Gender_Male"] = (out["Gender"] == "Male").astype(int)
    out["Subscription_Status_Premium"] = (out["Subscription_Status"] == "Premium").astype(int)
    out["Subscription_Status_Regular"] = (out["Subscription_Status"] == "Regular").astype(int)
    out = out.drop(columns=["Gender", "Subscription_Status"])
    return out.drop(columns=[c for c in DROP_COLS if c in out.columns])


def candidates() -> dict:
    """name -> (estimator, uses extended features)."""
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=10, random_state=42, n_jobs=-1)
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.03, max_iter=300, max_leaf_nodes=8, min_samples_leaf=50, random_state=42
    )
    return {
        # The notebook's original models, for reference.
        "Notebook Random Forest": (
            RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42), False),
        "Notebook Logistic Regression": (
            make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=1000)), False),
        # Improved candidates.
        "Logistic Regression": (lr, True),
        "Random Forest": (rf, True),
        "Random Forest + calibration": (CalibratedClassifierCV(rf, method="sigmoid", cv=5), True),
        "Gradient Boosting": (hgb, True),
        "Gradient Boosting + calibration": (CalibratedClassifierCV(hgb, method="sigmoid", cv=5), True),
    }


def _load_data(data_path: Path = DATA_PATH):
    df = pd.read_csv(data_path, keep_default_na=False, na_values=[""])
    return df, float(df["Return_Frequency"].median())


def _split(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def _f2_threshold(y, proba) -> float:
    grid = np.round(np.arange(0.05, 0.80, 0.01), 2)
    scores = [fbeta_score(y, proba >= t, beta=2) for t in grid]
    return float(grid[int(np.argmax(scores))])


def _classification_stats(y, proba, threshold) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, proba >= threshold).ravel()
    targeted = tp + fp
    cost = targeted * RETENTION_COST
    recovered = tp * CUSTOMER_CLV * RECOVERY_RATE
    return {
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "precision": tp / targeted if targeted else 0.0,
        "roi": (recovered - cost) / cost if cost else 0.0,
    }


def _segment_calibration(X, y, proba) -> list[dict]:
    """Predicted vs. actual churn rate by recency x subscription segment."""
    rows = []
    recency = pd.cut(X["Days_Since_Purchase"], [-1, 30, 60, 90, 10_000],
                     labels=["0-30 days", "31-60 days", "61-90 days", "90+ days"])
    for rec in recency.cat.categories:
        for sub in (1, 0):
            mask = (recency == rec).to_numpy() & (X["Has_Subscription"] == sub).to_numpy()
            if mask.sum():
                rows.append({
                    "segment": f"{rec}, {'subscribed' if sub else 'no subscription'}",
                    "n": int(mask.sum()),
                    "predicted": float(np.mean(proba[mask])),
                    "actual": float(np.mean(np.asarray(y)[mask])),
                })
    return rows


def compare(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Cross-validate every candidate on the training split (test split is reported only)."""
    df, ret_med = _load_data(data_path)
    y = df["Churn"]
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    rows = []
    for name, (est, extended) in candidates().items():
        X_tr, X_te, y_tr, y_te = _split(engineer(df, ret_med, extended), y)
        oof = cross_val_predict(clone(est), X_tr, y_tr, cv=skf, method="predict_proba")[:, 1]
        threshold = _f2_threshold(y_tr, oof)
        test_p = clone(est).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        long_inactive = (X_tr["Days_Since_Purchase"] > 90).to_numpy()
        stats = _classification_stats(y_te, test_p, threshold)
        at_half = _classification_stats(y_te, test_p, 0.5)
        rows.append({
            "Model": name,
            "CV avg precision": average_precision_score(y_tr, oof),
            "CV Brier": brier_score_loss(y_tr, oof),
            "CV 90+ days pred/actual": f"{oof[long_inactive].mean():.0%} / {y_tr[long_inactive].mean():.0%}",
            "Test ROC-AUC": roc_auc_score(y_te, test_p),
            "Threshold": threshold,
            "Test recall": stats["recall"],
            "Test precision": stats["precision"],
            "_at_0.5": at_half,
        })
    return pd.DataFrame(rows)


def train(model_name: str = BEST_MODEL, data_path: Path = DATA_PATH,
          model_path: Path = MODEL_PATH) -> dict:
    est, extended = candidates()[model_name]
    df, ret_med = _load_data(data_path)
    X = engineer(df, ret_med, extended)
    y = df["Churn"]
    X_tr, X_te, y_tr, y_te = _split(X, y)

    # Threshold from out-of-fold predictions on the training split only.
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = cross_val_predict(clone(est), X_tr, y_tr, cv=skf, method="predict_proba")[:, 1]
    threshold = _f2_threshold(y_tr, oof)

    # Honest held-out evaluation.
    test_p = clone(est).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    evaluation = {
        "roc_auc": roc_auc_score(y_te, test_p),
        "avg_precision": average_precision_score(y_te, test_p),
        "brier": brier_score_loss(y_te, test_p),
        "at_threshold": _classification_stats(y_te, test_p, threshold),
        "at_0.5": _classification_stats(y_te, test_p, 0.5),
        "segments_cv": _segment_calibration(X_tr, y_tr, oof),
    }

    # Deploy a model trained on all 4,500 customers.
    final = clone(est).fit(X, y)

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

    bundle = {
        "model": final,
        "model_name": model_name,
        "extended": extended,
        "columns": list(X.columns),
        "threshold": threshold,
        "evaluation": evaluation,
        "return_median": ret_med,
        "defaults": defaults,
        "ranges": {c: (float(df[c].min()), float(df[c].max()))
                   for c in RAW_FEATURES if pd.api.types.is_numeric_dtype(df[c])},
        "base_churn_rate": float(y.mean()),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


def load(model_path: Path = MODEL_PATH) -> dict:
    """Load the trained bundle, (re)training it if missing or out of date."""
    if model_path.exists():
        try:
            bundle = joblib.load(model_path)
            if bundle.get("model_name") == BEST_MODEL:
                return bundle
        except Exception:
            pass  # e.g. pickled with a different scikit-learn version: retrain
    return train(model_path=model_path)


def predict(bundle: dict, customer: dict) -> float:
    """Calibrated churn probability (0-1) for one customer.

    `customer` holds any subset of RAW_FEATURES; missing ones use dataset defaults.
    """
    row = {**bundle["defaults"], **customer}
    X = engineer(pd.DataFrame([row]), bundle["return_median"], bundle["extended"])
    return float(bundle["model"].predict_proba(X[bundle["columns"]])[0, 1])


def _write_report(table: pd.DataFrame, bundle: dict) -> str:
    ev = bundle["evaluation"]
    t, base = ev["at_threshold"], ev["at_0.5"]
    nb = table.set_index("Model").loc["Notebook Random Forest", "_at_0.5"]
    fmt = table.drop(columns=["_at_0.5"])
    for c in ["CV avg precision", "Test ROC-AUC"]:
        fmt[c] = fmt[c].map("{:.3f}".format)
    fmt["CV Brier"] = fmt["CV Brier"].map("{:.4f}".format)
    for c in ["Test recall", "Test precision"]:
        fmt[c] = fmt[c].map("{:.0%}".format)
    fmt["Threshold"] = fmt["Threshold"].map("{:.2f}".format)

    seg = "\n".join(
        f"| {s['segment']} | {s['n']} | {s['predicted']:.0%} | {s['actual']:.0%} |"
        for s in ev["segments_cv"]
    )
    md = f"""# Churn model report

Generated by `python churn_model.py`. All selection decisions use 5-fold cross-validation
on the 3,600-customer training split; the 900-customer test split is only used for the
final numbers below.

## Selected model: {bundle['model_name']} (decision threshold {bundle['threshold']:.2f})

| Test set (900 customers, {t['tp'] + t['fn']} churners) | Notebook Random Forest @ 0.50 | Selected model @ {t['threshold']:.2f} |
|---|---|---|
| Churners caught (true positives) | {nb['tp']} | **{t['tp']}** |
| Churners missed (false negatives) | {nb['fn']} | **{t['fn']}** |
| Recall | {nb['recall']:.0%} | **{t['recall']:.0%}** |
| Precision | {nb['precision']:.0%} | {t['precision']:.0%} |
| Customers targeted | {nb['tp'] + nb['fp']} | {t['tp'] + t['fp']} |
| Campaign ROI (₹{RETENTION_COST} offer, ₹{CUSTOMER_CLV:,} CLV, {RECOVERY_RATE:.0%} recovery) | {nb['roi']:+.0%} | {t['roi']:+.0%} |

The selected model at the default 0.50 threshold would catch only {base['tp']} churners
(recall {base['recall']:.0%}); lowering the threshold is what buys the extra true positives.
The threshold maximises the F2 score (recall weighted twice as much as precision) on
out-of-fold predictions.

Held-out ROC-AUC {ev['roc_auc']:.3f}, average precision {ev['avg_precision']:.3f}, Brier score {ev['brier']:.4f}.

## Calibration: predicted vs. actual churn rate (out-of-fold, training split)

| Segment | Customers | Predicted | Actual |
|---|---|---|---|
{seg}

## Candidate comparison

Ranked by cross-validated average precision (ranking churners first), then Brier score
(probability accuracy). "90+ days pred/actual" shows whether long-inactive customers are
under-predicted.

{fmt.to_markdown(index=False)}
"""
    REPORT_PATH.write_text(md)
    return md


if __name__ == "__main__":
    print("Comparing candidate models with 5-fold cross-validation...")
    table = compare()
    improved = table[~table["Model"].str.startswith("Notebook")]
    best = improved.sort_values(["CV avg precision", "CV Brier"], ascending=[False, True]).iloc[0]["Model"]
    print(table.drop(columns=["_at_0.5"]).to_string(index=False))
    if best != BEST_MODEL:
        print(f"\nNOTE: best model is now '{best}'; update BEST_MODEL in churn_model.py.", file=sys.stderr)
    bundle = train(best)
    _write_report(table, bundle)
    print(f"\nSelected: {best} (threshold {bundle['threshold']:.2f})")
    print(f"Saved model to {MODEL_PATH} and report to {REPORT_PATH}")
