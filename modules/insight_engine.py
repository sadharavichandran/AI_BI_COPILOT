# modules/insight_engine.py
# Automatically generates business insights from dataset analysis results

import pandas as pd
import numpy as np
from modules.data_analyzer import get_correlation_matrix, get_top_correlations


def generate_insights(df: pd.DataFrame, domain: str = "General") -> list:
    """
    Auto-generate textual business insights from the dataset.
    Returns a list of insight strings.
    """
    insights = []
    numeric_df = df.select_dtypes(include=[np.number])
    cat_df = df.select_dtypes(include=["object", "category"])

    # ── Basic dataset insights ──────────────────────────────────────────────
    n_rows, n_cols = df.shape
    insights.append(
        f"📊 **Dataset Overview:** Your dataset contains **{n_rows:,} records** and **{n_cols} features** "
        f"({numeric_df.shape[1]} numeric, {cat_df.shape[1]} categorical)."
    )

    missing_pct = df.isnull().mean().mean() * 100
    if missing_pct > 0:
        insights.append(
            f"⚠️ **Data Quality:** About **{missing_pct:.1f}%** of data values are missing across all columns. "
            f"Imputation has been applied to maintain data integrity."
        )
    else:
        insights.append("✅ **Data Quality:** No missing values detected — dataset is complete.")

    # ── Numeric column insights ──────────────────────────────────────────────
    if not numeric_df.empty:
        for col in numeric_df.columns[:5]:  # Limit to 5 to avoid noise
            skew = numeric_df[col].skew()
            mean_val = numeric_df[col].mean()
            std_val = numeric_df[col].std()

            if abs(skew) > 1:
                direction = "right (positive)" if skew > 0 else "left (negative)"
                insights.append(
                    f"📈 **{col}** is **heavily skewed {direction}** (skewness={skew:.2f}). "
                    f"Mean={mean_val:.2f}, Std={std_val:.2f}. "
                    f"Consider log transformation for modeling."
                )
            else:
                insights.append(
                    f"📉 **{col}** follows a roughly **normal distribution** (skewness={skew:.2f}). "
                    f"Mean={mean_val:.2f}, Std={std_val:.2f}."
                )

        # Outlier detection using IQR
        for col in numeric_df.columns[:5]:
            Q1, Q3 = numeric_df[col].quantile(0.25), numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            n_outliers = ((numeric_df[col] < Q1 - 1.5 * IQR) | (numeric_df[col] > Q3 + 1.5 * IQR)).sum()
            if n_outliers > 0:
                pct = n_outliers / len(df) * 100
                insights.append(
                    f"🔍 **Outliers in {col}:** {n_outliers} outliers detected ({pct:.1f}% of data). "
                    f"These may affect model performance."
                )

    # ── Correlation insights ──────────────────────────────────────────────────
    top_corr = get_top_correlations(df, top_n=5)
    if not top_corr.empty:
        for _, row in top_corr.iterrows():
            v1, v2, corr = row["Variable 1"], row["Variable 2"], row["Correlation"]
            if abs(corr) >= 0.7:
                strength = "strong"
                direction = "positive" if corr > 0 else "negative"
                insights.append(
                    f"🔗 **High Correlation:** **{v1}** and **{v2}** have a **{strength} {direction} correlation** "
                    f"(r={corr:.2f}). {'As one increases, so does the other.' if corr > 0 else 'They move in opposite directions.'}"
                )
            elif abs(corr) >= 0.4:
                insights.append(
                    f"📎 **Moderate Correlation:** **{v1}** and **{v2}** are moderately correlated (r={corr:.2f})."
                )

    # ── Categorical insights ──────────────────────────────────────────────────
    for col in cat_df.columns[:3]:
        top_val = df[col].value_counts().idxmax()
        top_pct = df[col].value_counts(normalize=True).max() * 100
        n_unique = df[col].nunique()
        insights.append(
            f"🏷️ **{col}:** Has **{n_unique} unique categories**. "
            f"Most common: **'{top_val}'** ({top_pct:.1f}% of records)."
        )

    # ── Domain-specific insights ──────────────────────────────────────────────
    domain_insights = _get_domain_insights(df, domain)
    insights.extend(domain_insights)

    return insights


def _get_domain_insights(df: pd.DataFrame, domain: str) -> list:
    """Generate domain-specific insights based on column name heuristics."""
    insights = []
    cols_lower = {c.lower(): c for c in df.columns}

    if domain == "Student Analysis":
        if "grade" in cols_lower or "score" in cols_lower or "marks" in cols_lower:
            col = cols_lower.get("grade") or cols_lower.get("score") or cols_lower.get("marks")
            mean_g = df[col].mean()
            insights.append(
                f"🎓 **Student Performance:** Average grade/score is **{mean_g:.2f}**. "
                f"{'Students are performing above average.' if mean_g > 60 else 'Performance improvement is recommended.'}"
            )
        math_col = next((v for k, v in cols_lower.items() if "math" in k), None)
        sci_col = next((v for k, v in cols_lower.items() if "science" in k or "sci" in k), None)
        if math_col and sci_col and pd.api.types.is_numeric_dtype(df[math_col]):
            corr = df[math_col].corr(df[sci_col])
            if abs(corr) > 0.5:
                insights.append(
                    f"📚 **Math & Science** are **{'highly' if abs(corr) > 0.7 else 'moderately'} correlated** (r={corr:.2f}). "
                    "Students strong in one tend to excel in the other."
                )

    elif domain == "Healthcare Prediction":
        age_col = next((v for k, v in cols_lower.items() if "age" in k), None)
        if age_col and pd.api.types.is_numeric_dtype(df[age_col]):
            mean_age = df[age_col].mean()
            insights.append(
                f"🏥 **Patient Demographics:** Average patient age is **{mean_age:.1f} years**. "
                f"{'Older patient population — focus on chronic disease management.' if mean_age > 50 else 'Younger population — preventive care is key.'}"
            )
        diag_col = next((v for k, v in cols_lower.items() if "diagnos" in k or "condition" in k), None)
        if diag_col:
            top = df[diag_col].value_counts().idxmax()
            insights.append(f"💊 **Most Common Diagnosis:** '{top}' — prioritize resources for this condition.")

    elif domain == "Retail Business Analytics":
        sales_col = next((v for k, v in cols_lower.items() if "sales" in k or "revenue" in k or "amount" in k), None)
        if sales_col and pd.api.types.is_numeric_dtype(df[sales_col]):
            total = df[sales_col].sum()
            mean_s = df[sales_col].mean()
            insights.append(
                f"🛒 **Revenue Summary:** Total sales = **${total:,.2f}**, average transaction = **${mean_s:,.2f}**."
            )
        prod_col = next((v for k, v in cols_lower.items() if "product" in k or "item" in k or "category" in k), None)
        if prod_col and sales_col and pd.api.types.is_numeric_dtype(df[sales_col]):
            top_prod = df.groupby(prod_col)[sales_col].sum().idxmax()
            insights.append(
                f"🏆 **Top Performing Product/Category:** **'{top_prod}'** generates the highest revenue. "
                "Consider expanding inventory or promotions for this item."
            )

    elif domain == "Financial Modeling":
        ret_col = next((v for k, v in cols_lower.items() if "return" in k or "profit" in k or "gain" in k), None)
        if ret_col and pd.api.types.is_numeric_dtype(df[ret_col]):
            sharpe = df[ret_col].mean() / df[ret_col].std() if df[ret_col].std() != 0 else 0
            insights.append(
                f"💰 **Risk-Adjusted Performance:** Estimated Sharpe-like ratio = **{sharpe:.4f}**. "
                f"{'Good risk-adjusted returns.' if sharpe > 1 else 'Consider rebalancing for better risk-return tradeoff.'}"
            )

    return insights


def generate_ml_insights(result: dict, target_col: str) -> list:
    """Generate insights specific to model training results."""
    insights = []
    pt = result["problem_type"]
    metrics = result["metrics"]

    if pt == "classification":
        acc = metrics.get("Accuracy", 0)
        f1 = metrics.get("F1 Score (weighted)", 0)
        quality = "excellent" if acc >= 0.9 else "good" if acc >= 0.75 else "moderate" if acc >= 0.6 else "low"
        insights.append(
            f"🤖 **Classification Model:** Logistic Regression trained on **'{target_col}'**. "
            f"Accuracy = **{acc:.2%}** ({quality}), F1 = **{f1:.4f}**."
        )
        if acc < 0.7:
            insights.append(
                "💡 **Improvement Tip:** Accuracy below 70%. Consider feature engineering, "
                "removing low-importance features, or trying a non-linear model."
            )
    else:
        r2 = metrics.get("R² Score", 0)
        rmse = metrics.get("RMSE", 0)
        quality = "excellent" if r2 >= 0.9 else "good" if r2 >= 0.7 else "moderate" if r2 >= 0.5 else "weak"
        insights.append(
            f"📊 **Regression Model:** Linear Regression trained on **'{target_col}'**. "
            f"R² = **{r2:.4f}** ({quality} fit), RMSE = **{rmse:.4f}**."
        )

    # Feature importance insight
    fi = result.get("feature_importance")
    if fi is not None and not fi.empty:
        top_feat = fi.iloc[0]["Feature"]
        insights.append(
            f"🔑 **Most Important Feature:** **'{top_feat}'** has the highest influence on predicting '{target_col}'."
        )

    return insights
