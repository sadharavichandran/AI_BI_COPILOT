# modules/ml_engine.py
# Auto-detects problem type and trains appropriate ML model

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, r2_score,
    mean_squared_error, mean_absolute_error,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


def detect_problem_type(series: pd.Series) -> str:
    """
    Auto-detect if target column is classification or regression.
    Heuristic: ≤ 10 unique values OR dtype is object/bool → classification
    """
    if series.dtype == object or series.dtype == bool:
        return "classification"
    unique_ratio = series.nunique() / len(series)
    if series.nunique() <= 10 or unique_ratio < 0.05:
        return "classification"
    return "regression"


def prepare_features(df: pd.DataFrame, target_col: str):
    """
    Encode categorical features and return X, y arrays ready for sklearn.
    Returns: X (array), y (array), feature_names (list), label_encoder (if classification)
    """
    df = df.copy().dropna()
    y_raw = df[target_col]
    X_df = df.drop(columns=[target_col])

    # Encode categorical features
    for col in X_df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))

    # Encode target if classification
    label_encoder = None
    problem_type = detect_problem_type(y_raw)
    if problem_type == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
    else:
        y = y_raw.values

    # Keep only numeric columns
    X_df = X_df.select_dtypes(include=[np.number])
    feature_names = list(X_df.columns)

    return X_df.values, y, feature_names, label_encoder


def train_model(df: pd.DataFrame, target_col: str, test_size: float = 0.2):
    """
    Full pipeline: detect problem, train model, evaluate metrics.

    Returns a result dict with:
        problem_type, model, metrics, feature_names,
        label_encoder, X_test, y_test, y_pred
    """
    X, y, feature_names, label_encoder = prepare_features(df, target_col)

    if len(X) < 10:
        raise ValueError("Not enough rows to train a model (need ≥ 10 after cleaning).")

    problem_type = detect_problem_type(df[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Build pipeline with scaling
    if problem_type == "classification":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "F1 Score (weighted)": round(f1_score(y_test, y_pred, average="weighted"), 4),
            "Train Size": len(X_train),
            "Test Size": len(X_test),
        }
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=[str(c) for c in (label_encoder.classes_ if label_encoder else range(len(np.unique(y))))]
        )

    else:  # regression
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        metrics = {
            "R² Score": round(r2_score(y_test, y_pred), 4),
            "MSE": round(mse, 4),
            "RMSE": round(np.sqrt(mse), 4),
            "MAE": round(mean_absolute_error(y_test, y_pred), 4),
            "Train Size": len(X_train),
            "Test Size": len(X_test),
        }
        cm = None
        report = None

    # Feature importance (coefficients for linear models)
    model_obj = pipeline.named_steps["model"]
    if hasattr(model_obj, "coef_"):
        coefs = model_obj.coef_
        if coefs.ndim > 1:
            coefs = np.abs(coefs).mean(axis=0)
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": np.abs(coefs)
        }).sort_values("Importance", ascending=False)
    else:
        importance = pd.DataFrame()

    return {
        "problem_type": problem_type,
        "model": pipeline,
        "metrics": metrics,
        "feature_names": feature_names,
        "label_encoder": label_encoder,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_importance": importance
    }


def sample_data(df: pd.DataFrame, fraction: float = 0.25, random_state: int = 42) -> pd.DataFrame:
    """Return a random sample of the dataframe."""
    n = max(1, int(len(df) * fraction))
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)
