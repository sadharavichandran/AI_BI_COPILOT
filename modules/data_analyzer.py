# modules/data_analyzer.py
# Generates summary statistics, correlation matrix, trend analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO


def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return extended descriptive statistics for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()

    stats = numeric_df.describe().T
    stats["skewness"] = numeric_df.skew()
    stats["kurtosis"] = numeric_df.kurt()
    stats["median"] = numeric_df.median()
    stats["variance"] = numeric_df.var()
    return stats.round(4)


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr()


def plot_correlation_heatmap(df: pd.DataFrame):
    """Generate a seaborn heatmap of the correlation matrix. Returns a matplotlib figure."""
    corr = get_correlation_matrix(df)
    if corr.empty:
        return None

    fig, ax = plt.subplots(figsize=(min(14, max(6, len(corr.columns))),
                                    min(12, max(5, len(corr.columns)))))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_distribution(df: pd.DataFrame, column: str):
    """Plot histogram + KDE for a numeric column using plotly."""
    if column not in df.columns:
        return None
    fig = px.histogram(
        df, x=column, marginal="box", nbins=40,
        title=f"Distribution of {column}",
        template="plotly_white",
        color_discrete_sequence=["#636EFA"]
    )
    return fig


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None):
    """Scatter plot between two numeric columns."""
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col,
        title=f"{x_col} vs {y_col}",
        template="plotly_white",
        trendline="ols" if color_col is None else None
    )
    return fig


def plot_bar_chart(df: pd.DataFrame, column: str, top_n: int = 15):
    """Bar chart for categorical column value counts."""
    counts = df[column].value_counts().head(top_n).reset_index()
    counts.columns = [column, "count"]
    fig = px.bar(
        counts, x=column, y="count",
        title=f"Top {top_n} Values in '{column}'",
        template="plotly_white",
        color="count",
        color_continuous_scale="Blues"
    )
    return fig


def plot_line_trend(df: pd.DataFrame, x_col: str, y_col: str):
    """Line chart for trend visualization."""
    fig = px.line(
        df.sort_values(x_col), x=x_col, y=y_col,
        title=f"Trend: {y_col} over {x_col}",
        template="plotly_white",
        markers=True
    )
    return fig


def plot_box_plots(df: pd.DataFrame, columns: list):
    """Box plots for multiple numeric columns side by side."""
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return None

    fig = go.Figure()
    for col in numeric_cols:
        fig.add_trace(go.Box(y=df[col], name=col, boxmean=True))

    fig.update_layout(
        title="Box Plot Comparison",
        template="plotly_white",
        showlegend=True
    )
    return fig


def get_top_correlations(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the top N most correlated variable pairs (excluding self-correlation)."""
    corr = get_correlation_matrix(df)
    if corr.empty:
        return pd.DataFrame()

    corr_unstacked = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    ).stack().reset_index()
    corr_unstacked.columns = ["Variable 1", "Variable 2", "Correlation"]
    corr_unstacked["Abs Correlation"] = corr_unstacked["Correlation"].abs()
    return corr_unstacked.sort_values("Abs Correlation", ascending=False).head(top_n)
