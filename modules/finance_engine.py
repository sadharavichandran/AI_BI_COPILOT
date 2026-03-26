# modules/finance_engine.py
# Portfolio optimization, risk metrics, and stock simulation using NumPy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# ─── Stock Simulation ────────────────────────────────────────────────────────

def simulate_stock_price(
    initial_price: float = 100.0,
    mu: float = 0.0008,       # daily drift (expected return)
    sigma: float = 0.02,      # daily volatility
    days: int = 252,          # trading days in a year
    n_simulations: int = 50,
    seed: int = 42
) -> dict:
    """
    Monte Carlo simulation of stock prices using Geometric Brownian Motion.

    dS = S * (mu * dt + sigma * sqrt(dt) * Z)
    where Z ~ N(0,1)
    """
    np.random.seed(seed)
    dt = 1  # daily steps

    # Shape: (days, n_simulations)
    random_shocks = np.random.standard_normal((days, n_simulations))
    daily_returns = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks)

    # Price paths
    price_paths = np.zeros((days + 1, n_simulations))
    price_paths[0] = initial_price
    for t in range(1, days + 1):
        price_paths[t] = price_paths[t - 1] * daily_returns[t - 1]

    final_prices = price_paths[-1]
    annualized_return = (np.mean(final_prices) / initial_price - 1) * 100
    prob_profit = np.mean(final_prices > initial_price) * 100
    var_95 = np.percentile(final_prices, 5)  # Value at Risk (5th percentile)

    return {
        "price_paths": price_paths,
        "final_prices": final_prices,
        "initial_price": initial_price,
        "mean_final": round(np.mean(final_prices), 2),
        "median_final": round(np.median(final_prices), 2),
        "std_final": round(np.std(final_prices), 2),
        "annualized_return_pct": round(annualized_return, 2),
        "prob_profit_pct": round(prob_profit, 2),
        "var_95": round(var_95, 2),
        "days": days
    }


def plot_stock_simulation(sim_result: dict):
    """Plot Monte Carlo stock simulation paths."""
    paths = sim_result["price_paths"]
    days = sim_result["days"]
    x = list(range(days + 1))

    fig = go.Figure()

    # Plot individual paths (up to 50)
    n_paths = min(paths.shape[1], 50)
    for i in range(n_paths):
        fig.add_trace(go.Scatter(
            x=x, y=paths[:, i],
            mode="lines",
            line=dict(width=0.8, color="rgba(99, 110, 250, 0.3)"),
            showlegend=False
        ))

    # Mean path
    mean_path = paths.mean(axis=1)
    fig.add_trace(go.Scatter(
        x=x, y=mean_path,
        mode="lines",
        line=dict(width=3, color="red"),
        name="Mean Path"
    ))

    fig.update_layout(
        title=f"Monte Carlo Stock Simulation ({n_paths} paths, {days} days)",
        xaxis_title="Days",
        yaxis_title="Price ($)",
        template="plotly_white"
    )
    return fig


# ─── Portfolio Optimization ───────────────────────────────────────────────────

def generate_random_portfolio(n_assets: int = 4, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic daily return data for n_assets.
    Returns a DataFrame of daily returns.
    """
    np.random.seed(seed)
    dates = pd.date_range(start="2022-01-01", periods=252, freq="B")
    returns = pd.DataFrame(
        np.random.randn(252, n_assets) * 0.015 + 0.0005,
        index=dates,
        columns=[f"Asset_{chr(65 + i)}" for i in range(n_assets)]
    )
    return returns


def optimize_portfolio(returns: pd.DataFrame, n_portfolios: int = 5000, seed: int = 42) -> dict:
    """
    Monte Carlo portfolio optimization.
    Generates n_portfolios with random weights, computes Sharpe ratios,
    and identifies the optimal (max Sharpe) and minimum variance portfolios.
    """
    np.random.seed(seed)
    n_assets = returns.shape[1]
    asset_names = list(returns.columns)

    mean_returns = returns.mean() * 252       # annualized
    cov_matrix = returns.cov() * 252          # annualized

    port_returns = []
    port_volatilities = []
    port_sharpes = []
    port_weights_list = []

    risk_free_rate = 0.04  # 4% annual

    for _ in range(n_portfolios):
        weights = np.random.dirichlet(np.ones(n_assets))  # sum to 1
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(weights @ cov_matrix @ weights)
        sharpe = (ret - risk_free_rate) / vol

        port_returns.append(ret)
        port_volatilities.append(vol)
        port_sharpes.append(sharpe)
        port_weights_list.append(weights)

    port_returns = np.array(port_returns)
    port_volatilities = np.array(port_volatilities)
    port_sharpes = np.array(port_sharpes)

    # Best portfolio: max Sharpe ratio
    max_sharpe_idx = np.argmax(port_sharpes)
    min_vol_idx = np.argmin(port_volatilities)

    optimal_weights = dict(zip(asset_names, port_weights_list[max_sharpe_idx].round(4)))
    min_vol_weights = dict(zip(asset_names, port_weights_list[min_vol_idx].round(4)))

    return {
        "port_returns": port_returns,
        "port_volatilities": port_volatilities,
        "port_sharpes": port_sharpes,
        "max_sharpe_return": round(port_returns[max_sharpe_idx] * 100, 2),
        "max_sharpe_vol": round(port_volatilities[max_sharpe_idx] * 100, 2),
        "max_sharpe_ratio": round(port_sharpes[max_sharpe_idx], 4),
        "optimal_weights": optimal_weights,
        "min_vol_return": round(port_returns[min_vol_idx] * 100, 2),
        "min_vol_vol": round(port_volatilities[min_vol_idx] * 100, 2),
        "min_vol_weights": min_vol_weights,
        "asset_names": asset_names
    }


def plot_efficient_frontier(opt_result: dict):
    """Plot the efficient frontier from portfolio simulation."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=opt_result["port_volatilities"] * 100,
        y=opt_result["port_returns"] * 100,
        mode="markers",
        marker=dict(
            color=opt_result["port_sharpes"],
            colorscale="Viridis",
            size=4,
            colorbar=dict(title="Sharpe Ratio"),
            showscale=True
        ),
        name="Portfolios",
        text=[f"Sharpe: {s:.2f}" for s in opt_result["port_sharpes"]],
        hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<br>%{text}"
    ))

    # Mark optimal portfolio
    fig.add_trace(go.Scatter(
        x=[opt_result["max_sharpe_vol"]],
        y=[opt_result["max_sharpe_return"]],
        mode="markers",
        marker=dict(color="red", size=15, symbol="star"),
        name="Max Sharpe (Optimal)"
    ))

    fig.add_trace(go.Scatter(
        x=[opt_result["min_vol_vol"]],
        y=[opt_result["min_vol_return"]],
        mode="markers",
        marker=dict(color="green", size=15, symbol="diamond"),
        name="Min Volatility"
    ))

    fig.update_layout(
        title="Efficient Frontier (Monte Carlo Portfolio Optimization)",
        xaxis_title="Volatility (Risk) %",
        yaxis_title="Expected Annual Return %",
        template="plotly_white"
    )
    return fig


def calculate_var(returns_series: np.ndarray, confidence: float = 0.95, investment: float = 100000) -> dict:
    """
    Calculate Value at Risk (VaR) using historical simulation.
    """
    sorted_returns = np.sort(returns_series)
    var_pct = np.percentile(sorted_returns, (1 - confidence) * 100)
    cvar_pct = sorted_returns[sorted_returns <= var_pct].mean()

    return {
        "VaR (%)": round(var_pct * 100, 4),
        "CVaR (%)": round(cvar_pct * 100, 4),
        "VaR ($)": round(var_pct * investment, 2),
        "CVaR ($)": round(cvar_pct * investment, 2),
        "Confidence Level": f"{confidence * 100}%",
        "Investment": investment
    }
