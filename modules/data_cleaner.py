# modules/data_cleaner.py
# Handles all data cleaning operations: missing values, invalid rows, type inference

import pandas as pd
import numpy as np


def detect_column_types(df: pd.DataFrame) -> dict:
    """
    Automatically detect and categorize columns into numeric, categorical, datetime.
    Returns a dict: {col_name: 'numeric' | 'categorical' | 'datetime'}
    """
    col_types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_types[col] = "datetime"
        else:
            # Try parsing as datetime
            try:
                pd.to_datetime(df[col], infer_datetime_format=True)
                col_types[col] = "datetime"
            except Exception:
                col_types[col] = "categorical"
    return col_types


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame showing missing counts and percentages per column."""
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    summary = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing %": missing_pct
    })
    return summary[summary["Missing Count"] > 0].sort_values("Missing %", ascending=False)


def clean_data(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Clean the dataframe:
    - Fill numeric NaNs with mean or median based on strategy
    - Fill categorical NaNs with mode
    - Drop fully duplicate rows
    - Drop rows where ALL values are NaN

    Parameters:
        df       : Input DataFrame
        strategy : 'mean' or 'median' for numeric columns
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()

    # Drop fully duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    dupes_removed = before - len(df)

    # Drop rows where every column is NaN
    df.dropna(how="all", inplace=True)

    col_types = detect_column_types(df)

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue

        if col_types.get(col) == "numeric":
            if strategy == "median":
                fill_val = df[col].median()
            else:
                fill_val = df[col].mean()
            df[col].fillna(fill_val, inplace=True)

        elif col_types.get(col) == "categorical":
            mode_vals = df[col].mode()
            if len(mode_vals) > 0:
                df[col].fillna(mode_vals[0], inplace=True)
            else:
                df[col].fillna("Unknown", inplace=True)

        elif col_types.get(col) == "datetime":
            # Forward fill datetime columns
            df[col].fillna(method="ffill", inplace=True)

    return df, dupes_removed


def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Remove rows with outliers using IQR method for specified numeric columns.
    """
    df = df.copy()
    before = len(df)
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    removed = before - len(df)
    return df, removed
