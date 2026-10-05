"""HR Analytics & Employee Attrition — Task 3

Run from any working directory:
    pip install -r source_code/requirements.txt
    python source_code/hr_analytics_task3.py

The script reads the cleaned workbook from the project root and writes
metrics, held-out predictions, and diagnostic/EDA charts to project folders.
"""
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "HR_Analytics_Cleaned.xlsx"
CHARTS = ROOT / "charts"
CHARTS.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_excel(DATA_PATH, sheet_name="Cleaned_Data")
required = {"Attrition", "OverTime", "Department", "TenureGroup", "MonthlyIncome"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

# Binary target: Yes = employee attrition, No = retained.
y = (df["Attrition"].astype(str).str.strip().str.lower() == "yes").astype(int)
drop_cols = ["Attrition", "EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

# Feature engineering: average of the available employee experience ratings.
sat_cols = [c for c in [
    "EnvironmentSatisfaction", "JobInvolvement", "JobSatisfaction",
    "RelationshipSatisfaction", "WorkLifeBalance"
] if c in X.columns]
if sat_cols:
    X["SatisfactionScore"] = X[sat_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
if "OverTime" in X.columns:
    X["OverTimeFlag"] = (X["OverTime"].astype(str).str.strip().str.lower() == "yes").astype(int)

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]
preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
}
results, fitted, preds = [], {}, {}
for name, model in models.items():
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, proba),
    })
    fitted[name] = pipe
    preds[name] = (pred, proba)

metrics = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
metrics.to_csv(ROOT / "model_metrics.csv", index=False)
best_name = metrics.iloc[0]["Model"]
best_pred, best_proba = preds[best_name]
joblib.dump(fitted[best_name], ROOT / "attrition_model.joblib")
pd.DataFrame({
    "test_row_index": X_test.index,
    "actual_attrition": y_test.to_numpy(),
    "predicted_attrition": best_pred,
    "predicted_attrition_probability": best_proba,
}).to_csv(ROOT / "test_predictions.csv", index=False)

# EDA charts
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    def save_bar(series, title, xlabel, ylabel, filename, rotation=0):
        fig, ax = plt.subplots(figsize=(8, 5))
        series.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=rotation)
        fig.tight_layout()
        fig.savefig(CHARTS / filename, dpi=160)
        plt.close(fig)

    df.assign(_attrition=y).groupby("Department", observed=False)["_attrition"].mean().mul(100).sort_values(ascending=False).pipe(
        lambda s: save_bar(s, "Attrition Rate by Department", "Department", "Attrition rate (%)", "attrition_by_department.png", 15)
    )
    df.assign(_attrition=y).groupby("OverTime", observed=False)["_attrition"].mean().mul(100).pipe(
        lambda s: save_bar(s, "Attrition Rate by Overtime", "Overtime", "Attrition rate (%)", "attrition_by_overtime.png")
    )
    df.assign(_attrition=y).groupby("TenureGroup", observed=False)["_attrition"].mean().mul(100).pipe(
        lambda s: save_bar(s, "Attrition Rate by Tenure Group", "Tenure group", "Attrition rate (%)", "attrition_by_tenure.png", 15)
    )
    df.groupby("Department", observed=False)["MonthlyIncome"].mean().sort_values(ascending=False).pipe(
        lambda s: save_bar(s, "Average Monthly Income by Department", "Department", "Average monthly income", "income_by_department.png", 15)
    )

    cm = confusion_matrix(y_test, best_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Left"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(CHARTS / "confusion_matrix.png", dpi=160)
    plt.close(fig)

print(f"Dataset rows: {len(df)} | Attrition cases: {int(y.sum())} | Rate: {y.mean():.2%}")
print("Model metrics (sorted by ROC-AUC):")
print(metrics.round(3).to_string(index=False))
print(f"Selected by test ROC-AUC: {best_name}")
print(f"Saved attrition_model.joblib, model_metrics.csv, test_predictions.csv, and charts/ under {ROOT}")
