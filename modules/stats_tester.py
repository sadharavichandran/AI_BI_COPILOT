# modules/stats_tester.py
# Statistical hypothesis testing: Z-test and T-test implementations

import numpy as np
import pandas as pd
from scipy import stats


def z_test_one_sample(data: np.ndarray, popmean: float, alpha: float = 0.05) -> dict:
    """
    One-sample Z-test: tests whether sample mean differs from a known population mean.

    Parameters:
        data    : 1D numeric array (sample)
        popmean : Hypothesized population mean (H0)
        alpha   : Significance level (default 0.05)

    Returns dict with z_stat, p_value, decision, explanation
    """
    n = len(data)
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    std_error = sample_std / np.sqrt(n)

    z_stat = (sample_mean - popmean) / std_error
    # Two-tailed p-value using normal distribution
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    decision = "Reject H₀" if p_value < alpha else "Fail to Reject H₀"
    is_significant = p_value < alpha

    explanation = (
        f"The sample mean ({sample_mean:.4f}) {'significantly differs from' if is_significant else 'does NOT significantly differ from'} "
        f"the hypothesized population mean ({popmean}).\n\n"
        f"**Z-statistic:** {z_stat:.4f}\n"
        f"**P-value:** {p_value:.4f}\n"
        f"**Alpha (significance level):** {alpha}\n\n"
        f"{'✅ There IS a statistically significant difference.' if is_significant else '❌ There is NO statistically significant difference.'}"
    )

    return {
        "test": "One-Sample Z-Test",
        "z_stat": round(z_stat, 4),
        "p_value": round(p_value, 4),
        "alpha": alpha,
        "sample_mean": round(sample_mean, 4),
        "sample_size": n,
        "decision": decision,
        "is_significant": is_significant,
        "explanation": explanation
    }


def z_test_two_sample(data1: np.ndarray, data2: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Two-sample Z-test: tests whether the means of two independent samples are equal.
    """
    n1, n2 = len(data1), len(data2)
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)

    pooled_se = np.sqrt((std1**2 / n1) + (std2**2 / n2))
    z_stat = (mean1 - mean2) / pooled_se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    decision = "Reject H₀" if p_value < alpha else "Fail to Reject H₀"
    is_significant = p_value < alpha

    explanation = (
        f"Comparing **Group 1** (mean={mean1:.4f}, n={n1}) vs **Group 2** (mean={mean2:.4f}, n={n2}).\n\n"
        f"**Z-statistic:** {z_stat:.4f}\n"
        f"**P-value:** {p_value:.4f}\n"
        f"**Alpha:** {alpha}\n\n"
        f"{'✅ The means are statistically DIFFERENT.' if is_significant else '❌ The means are NOT statistically different.'}"
    )

    return {
        "test": "Two-Sample Z-Test",
        "z_stat": round(z_stat, 4),
        "p_value": round(p_value, 4),
        "alpha": alpha,
        "mean_group1": round(mean1, 4),
        "mean_group2": round(mean2, 4),
        "decision": decision,
        "is_significant": is_significant,
        "explanation": explanation
    }


def chi_square_test(df: pd.DataFrame, col1: str, col2: str, alpha: float = 0.05) -> dict:
    """Chi-square test of independence between two categorical columns."""
    contingency = pd.crosstab(df[col1], df[col2])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    decision = "Reject H₀" if p_value < alpha else "Fail to Reject H₀"
    is_significant = p_value < alpha

    explanation = (
        f"Testing whether **{col1}** and **{col2}** are independent.\n\n"
        f"**Chi² statistic:** {chi2:.4f}\n"
        f"**Degrees of freedom:** {dof}\n"
        f"**P-value:** {p_value:.4f}\n\n"
        f"{'✅ The variables are statistically DEPENDENT (related).' if is_significant else '❌ The variables are statistically INDEPENDENT (no significant relationship).'}"
    )

    return {
        "test": "Chi-Square Test of Independence",
        "chi2_stat": round(chi2, 4),
        "p_value": round(p_value, 4),
        "dof": dof,
        "alpha": alpha,
        "decision": decision,
        "is_significant": is_significant,
        "explanation": explanation,
        "contingency_table": contingency
    }
