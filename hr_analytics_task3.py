"""HR Analytics & Employee Attrition — Task 3.

Place this file in the project root beside HR_Analytics_Cleaned.xlsx (preferred)
or HR Analytics Dataset.xlsx, then run:
    pip install -r requirements.txt
    python hr_analytics_task3.py

Outputs: model_metrics.csv, test_predictions.csv, attrition_model.joblib, charts/.
"""
from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)


def load_dataset():
    """Find the workbook and select the sheet containing the required fields."""
    candidates = [
        ROOT / "HR_Analytics_Cleaned.xlsx",
        ROOT / "HR Analytics Dataset.xlsx",
        ROOT / "HR_Analytics_Dataset.xlsx",
    ]
    workbook = next((p for p in candidates if p.exists()), None)
    if workbook is None:
        workbooks = sorted(p for p in ROOT.glob("*.xlsx") if not p.name.startswith("~$"))
        if len(workbooks) == 1:
            workbook = workbooks[0]
        else:
            names = ", ".join(p.name for p in workbooks) or "no .xlsx files found"
            raise FileNotFoundError(
                "Could not locate the HR dataset. Put 'HR_Analytics_Cleaned.xlsx' "
                f"or 'HR Analytics Dataset.xlsx' beside this script. Found: {names}"
            )

    required = {"Attrition", "OverTime", "Department", "MonthlyIncome"}
    excel = pd.ExcelFile(workbook)
    for sheet in excel.sheet_names:
        candidate = pd.read_excel(workbook, sheet_name=sheet)
        normalized = {str(c).strip(): c for c in candidate.columns}
        if required.issubset(normalized):
            candidate.columns = [str(c).strip() for c in candidate.columns]
            print(f"Loaded dataset: {workbook.name} | sheet: {sheet}")
            return candidate
    raise ValueError(
        f"No sheet in {workbook.name} contains all required columns: "
        f"{', '.join(sorted(required))}. Available sheets: {', '.join(excel.sheet_names)}"
    )


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


def main():
    df = load_dataset()
    required = {"Attrition", "OverTime", "Department", "MonthlyIncome"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df = df.dropna(subset=["Attrition"]).copy()
    if df.empty:
        raise ValueError("Dataset has no rows with a valid Attrition value.")

    # Create a tenure band if the source workbook does not already contain one.
    if "TenureGroup" not in df.columns:
        if "YearsAtCompany" in df.columns:
            df["TenureGroup"] = pd.cut(
                pd.to_numeric(df["YearsAtCompany"], errors="coerce"),
                bins=[-np.inf, 1, 5, 10, np.inf],
                labels=["0-1 years", "2-5 years", "6-10 years", "10+ years"],
            ).astype("object")
        else:
            df["TenureGroup"] = "Unknown"

    # Target: 1 = employee left, 0 = employee stayed.
    y = (df["Attrition"].astype(str).str.strip().str.lower() == "yes").astype(int)
    if y.nunique() < 2:
        raise ValueError("Attrition must contain both 'Yes' and 'No' examples to train a classifier.")

    drop_cols = ["Attrition", "EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    # Feature engineering: combine available satisfaction measures and encode overtime.
    satisfaction_cols = [c for c in [
        "EnvironmentSatisfaction", "JobInvolvement", "JobSatisfaction",
        "RelationshipSatisfaction", "WorkLifeBalance",
    ] if c in X.columns]
    if satisfaction_cols:
        X["SatisfactionScore"] = X[satisfaction_cols].apply(
            lambda col: pd.to_numeric(col, errors="coerce")
        ).mean(axis=1)
    if "OverTime" in X.columns:
        X["OverTimeFlag"] = (
            X["OverTime"].astype(str).str.strip().str.lower() == "yes"
        ).astype(int)

    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols))
    preprocess = ColumnTransformer(transformers, remainder="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
    }
    results, fitted, predictions = [], {}, {}
    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        probability = pipe.predict_proba(X_test)[:, 1]
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, probability),
        })
        fitted[name] = pipe
        predictions[name] = (pred, probability)

    metrics = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
    metrics.to_csv(ROOT / "model_metrics.csv", index=False)
    best_name = str(metrics.iloc[0]["Model"])
    best_prediction, best_probability = predictions[best_name]
    joblib.dump(fitted[best_name], ROOT / "attrition_model.joblib")
    pd.DataFrame({
        "test_row_index": X_test.index,
        "actual_attrition": y_test.to_numpy(),
        "predicted_attrition": best_prediction,
        "predicted_attrition_probability": best_probability,
    }).to_csv(ROOT / "test_predictions.csv", index=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data_with_target = df.assign(_attrition=y)
        save_bar(
            data_with_target.groupby("Department", observed=False)["_attrition"].mean().mul(100).sort_values(ascending=False),
            "Attrition Rate by Department", "Department", "Attrition rate (%)",
            "attrition_by_department.png", 15,
        )
        save_bar(
            data_with_target.groupby("OverTime", observed=False)["_attrition"].mean().mul(100),
            "Attrition Rate by Overtime", "Overtime", "Attrition rate (%)",
            "attrition_by_overtime.png",
        )
        save_bar(
            data_with_target.groupby("TenureGroup", observed=False)["_attrition"].mean().mul(100),
            "Attrition Rate by Tenure Group", "Tenure group", "Attrition rate (%)",
            "attrition_by_tenure.png", 15,
        )
        income = df.assign(_income=pd.to_numeric(df["MonthlyIncome"], errors="coerce"))
        save_bar(
            income.groupby("Department", observed=False)["_income"].mean().sort_values(ascending=False),
            "Average Monthly Income by Department", "Department", "Average monthly income",
            "income_by_department.png", 15,
        )
        matrix = confusion_matrix(y_test, best_prediction, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(matrix, display_labels=["Stayed", "Left"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"Confusion Matrix — {best_name}")
        fig.tight_layout()
        fig.savefig(CHARTS / "confusion_matrix.png", dpi=160)
        plt.close(fig)

    print(f"Dataset rows: {len(df)} | Attrition cases: {int(y.sum())} | Attrition rate: {y.mean():.2%}")
    print("Model metrics (sorted by ROC-AUC):")
    print(metrics.round(3).to_string(index=False))
    print(f"Selected model: {best_name}")
    print(f"Outputs saved in: {ROOT} (charts in {CHARTS.name}/)")


if __name__ == "__main__":
    main()
