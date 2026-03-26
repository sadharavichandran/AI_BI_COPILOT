# modules/__init__.py
# Makes 'modules' a Python package

from modules.data_cleaner import detect_column_types, get_missing_summary, clean_data, remove_outliers_iqr
from modules.data_analyzer import (
    get_summary_statistics, get_correlation_matrix, plot_correlation_heatmap,
    plot_distribution, plot_scatter, plot_bar_chart, plot_line_trend,
    plot_box_plots, get_top_correlations
)
from modules.ml_engine import detect_problem_type, train_model, sample_data
from modules.stats_tester import z_test_one_sample, z_test_two_sample, chi_square_test
from modules.finance_engine import (
    simulate_stock_price, plot_stock_simulation,
    generate_random_portfolio, optimize_portfolio,
    plot_efficient_frontier, calculate_var
)
from modules.insight_engine import generate_insights, generate_ml_insights
from modules.ai_mentor import ask_mentor, build_context_from_session
