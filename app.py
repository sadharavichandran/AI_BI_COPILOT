# app.py
# AI Business Intelligence & Decision Copilot
# Main Streamlit application — entry point

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ── Module imports ────────────────────────────────────────────────────────────
from modules.data_cleaner import (
    detect_column_types, get_missing_summary, clean_data, remove_outliers_iqr
)
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

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Business Intelligence Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .insight-card {
        background: linear-gradient(135deg, #1a2a45 0%, #223a5f 100%);
        color: #eaf2ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #6ea8fe;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 8px;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeeba;
        padding: 1rem;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [
    ("df_raw", None), ("df_clean", None), ("df_sample", None),
    ("ml_result", None), ("chat_history", []), ("domain", "General"),
    ("insights", []), ("target_col", None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("🤖 AI BI Copilot")
    st.divider()

    # Domain Selection
    st.subheader("🌐 Domain Mode")
    domain = st.selectbox(
        "Select your domain:",
        ["General", "Student Analysis", "Healthcare Prediction",
         "Retail Business Analytics", "Financial Modeling"],
        index=0
    )
    st.session_state.domain = domain

    st.divider()

    # Navigation
    st.subheader("📂 Navigation")
    page = st.radio(
        "Go to:",
        ["🏠 Home", "📤 Data Upload", "🧹 Data Cleaning", "📊 Data Analysis",
         "🤖 Machine Learning", "📐 Statistics", "💰 Finance", "💡 AI Insights",
         "💬 AI Mentor Chat"],
        label_visibility="collapsed"
    )

    st.divider()
    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean
        st.success(f"✅ Dataset loaded\n{df.shape[0]:,} rows × {df.shape[1]} cols")
    else:
        st.info("📂 Upload a dataset to begin")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Business Intelligence & Decision Copilot</h1>
        <p style="font-size:1.1rem; opacity:0.9;">
            Upload any dataset → Clean → Analyze → Predict → Gain AI-powered insights
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🧹 Data Cleaning", "Auto", "NaN handling")
    with col2:
        st.metric("📊 Analysis", "Full EDA", "Stats + Charts")
    with col3:
        st.metric("🤖 ML", "Auto-detect", "Class + Regress")
    with col4:
        st.metric("💬 AI Mentor", "Live Chat", "Powered by Claude")

    st.divider()
    st.subheader("🚀 Quick Start Guide")

    cols = st.columns(3)
    steps = [
        ("1️⃣ Upload", "Go to **Data Upload** and upload your CSV file. The system auto-detects columns and types."),
        ("2️⃣ Clean & Analyze", "Visit **Data Cleaning** to fix missing values, then **Data Analysis** for EDA with charts."),
        ("3️⃣ ML & Insights", "Use **Machine Learning** for auto-trained models, then **AI Insights** for smart recommendations."),
    ]
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.info(f"**{title}**\n\n{desc}")

    st.divider()
    st.subheader("✨ Features")
    features = [
        "📤 **CSV Upload** — Auto column detection & type inference",
        "🧹 **Smart Cleaning** — Mean/median imputation, duplicate removal, outlier handling",
        "📊 **Full EDA** — Correlation heatmap, distributions, box plots, trend charts",
        "🎲 **Sampling** — Random 25% sample for quick exploratory analysis",
        "🤖 **Auto ML** — Logistic / Linear Regression with accuracy, F1, R², RMSE metrics",
        "📐 **Statistics** — Z-test (one/two sample), Chi-square hypothesis testing",
        "💰 **Finance** — Portfolio optimization, Monte Carlo simulation, VaR calculation",
        "💡 **AI Insights** — Auto-generated business insights from your data",
        "💬 **AI Mentor** — Chat with Claude for simple explanations & advice",
    ]
    for f in features:
        st.markdown(f"- {f}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATA UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📤 Data Upload":
    st.header("📤 Data Upload Module")
    st.write("Upload your CSV dataset — the system will automatically detect column types.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    col_sep, col_enc = st.columns(2)
    with col_sep:
        separator = st.selectbox("Separator", [",", ";", "\t", "|"], index=0)
    with col_enc:
        encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "iso-8859-1"], index=0)

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=separator, encoding=encoding)
            st.session_state.df_raw = df
            st.session_state.df_clean = df.copy()  # Start with raw, user can clean next
            st.success(f"✅ File uploaded successfully: **{uploaded_file.name}**")

            # ── Dataset overview
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📋 Rows", f"{df.shape[0]:,}")
            c2.metric("📊 Columns", df.shape[1])
            c3.metric("💾 Size", f"{df.memory_usage().sum() / 1024:.1f} KB")
            c4.metric("❓ Missing", f"{df.isnull().sum().sum():,}")

            st.divider()
            tab1, tab2, tab3 = st.tabs(["👁️ Preview", "🏷️ Column Types", "❓ Missing Values"])

            with tab1:
                n_preview = st.slider("Rows to preview", 5, 50, 10)
                st.dataframe(df.head(n_preview), use_container_width=True)

            with tab2:
                col_types = detect_column_types(df)
                type_df = pd.DataFrame({
                    "Column": list(col_types.keys()),
                    "Detected Type": list(col_types.values()),
                    "Pandas Dtype": [str(df[c].dtype) for c in col_types.keys()],
                    "Unique Values": [df[c].nunique() for c in col_types.keys()],
                    "Missing": [df[c].isnull().sum() for c in col_types.keys()]
                })
                st.dataframe(type_df, use_container_width=True)

                # Visual: type distribution pie chart
                type_counts = pd.Series(list(col_types.values())).value_counts()
                fig = px.pie(
                    values=type_counts.values, names=type_counts.index,
                    title="Column Type Distribution", template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab3:
                missing_summary = get_missing_summary(df)
                if missing_summary.empty:
                    st.success("🎉 No missing values in your dataset!")
                else:
                    st.warning(f"Found missing values in {len(missing_summary)} columns:")
                    st.dataframe(missing_summary, use_container_width=True)

                    fig = px.bar(
                        missing_summary.reset_index(), x="index", y="Missing %",
                        title="Missing Value % by Column", template="plotly_white",
                        color="Missing %", color_continuous_scale="Reds"
                    )
                    fig.update_xaxes(title="Column")
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("💡 Try changing the separator or encoding settings above.")

    else:
        st.info("👆 Upload a CSV file to get started.")
        st.markdown("""
        **Supported formats:** CSV with any delimiter  
        **Tip:** Your dataset can be about anything — sales, healthcare, students, finance, etc.  
        **No hardcoded data:** The system fully adapts to your dataset structure.
        """)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧹 Data Cleaning":
    st.header("🧹 Data Cleaning Module")

    if st.session_state.df_raw is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df_raw = st.session_state.df_raw

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("⚙️ Cleaning Options")
        strategy = st.radio("Numeric imputation strategy:", ["mean", "median"])
        remove_outliers = st.checkbox("Remove outliers (IQR method)")

        if remove_outliers:
            numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
            outlier_cols = st.multiselect(
                "Select columns for outlier removal:", numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )

        if st.button("🚀 Apply Cleaning", type="primary"):
            with st.spinner("Cleaning data..."):
                df_clean, dupes_removed = clean_data(df_raw, strategy=strategy)

                outliers_removed = 0
                if remove_outliers and outlier_cols:
                    df_clean, outliers_removed = remove_outliers_iqr(df_clean, outlier_cols)

                st.session_state.df_clean = df_clean
                st.success("✅ Data cleaned successfully!")

                # Summary of changes
                st.markdown(f"""
                **Cleaning Summary:**
                - Original rows: **{len(df_raw):,}**
                - Duplicate rows removed: **{dupes_removed}**
                - Outlier rows removed: **{outliers_removed}**
                - Final rows: **{len(df_clean):,}**
                - Missing values remaining: **{df_clean.isnull().sum().sum()}**
                """)

    with col2:
        st.subheader("📊 Before vs After Comparison")
        if st.session_state.df_clean is not None:
            df_clean = st.session_state.df_clean
            before_missing = df_raw.isnull().sum().sum()
            after_missing = df_clean.isnull().sum().sum()

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Rows Before", f"{len(df_raw):,}")
            mc2.metric("Rows After", f"{len(df_clean):,}", f"-{len(df_raw)-len(df_clean)}")
            mc3.metric("Missing Values", f"{after_missing}", f"-{before_missing-after_missing}")

            st.divider()

            # Show cleaned data preview
            st.subheader("Cleaned Dataset Preview")
            st.dataframe(df_clean.head(15), use_container_width=True)

            # Numeric stats comparison
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Inspect column distribution:", numeric_cols)
                fig = plot_distribution(df_clean, selected_col)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Apply cleaning to see results here.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Data Analysis":
    st.header("📊 Data Analysis Module")

    if st.session_state.df_clean is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = st.session_state.df_clean
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ── Sampling option
    use_sample = st.checkbox("🎲 Use 25% random sample for faster analysis")
    if use_sample:
        df = sample_data(df, fraction=0.25)
        st.info(f"Using random sample: **{len(df):,} rows** (25% of dataset)")
        st.session_state.df_sample = df

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Summary Stats", "🔥 Correlation", "📈 Distributions", "📊 Charts", "📦 Box Plots"
    ])

    # ── Tab 1: Summary Statistics ─────────────────────────────────────────
    with tab1:
        st.subheader("Summary Statistics")
        stats = get_summary_statistics(df)
        if not stats.empty:
            st.dataframe(stats.style.background_gradient(cmap="Blues", subset=["mean", "std"]),
                         use_container_width=True)
        else:
            st.info("No numeric columns found.")

        if cat_cols:
            st.divider()
            st.subheader("Categorical Column Summaries")
            for col in cat_cols[:5]:
                with st.expander(f"📝 {col} — {df[col].nunique()} unique values"):
                    vc = df[col].value_counts().head(10)
                    fig = px.bar(x=vc.index, y=vc.values, title=f"Top values in '{col}'",
                                 labels={"x": col, "y": "Count"},
                                 template="plotly_white",
                                 color=vc.values, color_continuous_scale="Blues")
                    st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Correlation ─────────────────────────────────────────────────
    with tab2:
        st.subheader("Correlation Matrix")
        if len(numeric_cols) >= 2:
            # Heatmap
            fig_heat = plot_correlation_heatmap(df)
            if fig_heat:
                st.pyplot(fig_heat, use_container_width=True)

            st.divider()
            st.subheader("🔝 Top Correlated Pairs")
            top_corr = get_top_correlations(df)
            if not top_corr.empty:
                st.dataframe(
                    top_corr.style.background_gradient(cmap="RdYlGn", subset=["Correlation"]),
                    use_container_width=True
                )

                # Quick scatter for top pair
                top_row = top_corr.iloc[0]
                st.subheader(f"Scatter: {top_row['Variable 1']} vs {top_row['Variable 2']}")
                fig = plot_scatter(df, top_row["Variable 1"], top_row["Variable 2"])
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis.")

    # ── Tab 3: Distributions ───────────────────────────────────────────────
    with tab3:
        st.subheader("Column Distributions")
        if numeric_cols:
            col_to_plot = st.selectbox("Select column:", numeric_cols, key="dist_col")
            fig = plot_distribution(df, col_to_plot)
            st.plotly_chart(fig, use_container_width=True)

            # All distributions in one view
            if st.checkbox("Show all numeric distributions"):
                cols_to_show = st.multiselect(
                    "Columns:", numeric_cols, default=numeric_cols[:4]
                )
                if cols_to_show:
                    ncols = 2
                    for i in range(0, len(cols_to_show), ncols):
                        row_cols = cols_to_show[i:i+ncols]
                        c1, c2 = st.columns(ncols)
                        for col, ui_col in zip(row_cols, [c1, c2]):
                            with ui_col:
                                f = plot_distribution(df, col)
                                if f:
                                    st.plotly_chart(f, use_container_width=True)
        else:
            st.info("No numeric columns available.")

    # ── Tab 4: Charts ──────────────────────────────────────────────────────
    with tab4:
        st.subheader("Interactive Charts")
        chart_type = st.selectbox("Chart Type:",
                                   ["Scatter Plot", "Line Trend", "Bar Chart (Categorical)"])

        if chart_type == "Scatter Plot" and len(numeric_cols) >= 2:
            c1, c2, c3 = st.columns(3)
            x_col = c1.selectbox("X axis:", numeric_cols, key="sc_x")
            y_col = c2.selectbox("Y axis:", numeric_cols, index=min(1, len(numeric_cols)-1), key="sc_y")
            color_col = c3.selectbox("Color by:", ["None"] + cat_cols, key="sc_c")
            color_col = None if color_col == "None" else color_col
            fig = plot_scatter(df, x_col, y_col, color_col)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Line Trend" and len(numeric_cols) >= 2:
            c1, c2 = st.columns(2)
            x_col = c1.selectbox("X axis:", df.columns.tolist(), key="lt_x")
            y_col = c2.selectbox("Y axis:", numeric_cols, key="lt_y")
            fig = plot_line_trend(df, x_col, y_col)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Bar Chart (Categorical)" and cat_cols:
            cat_col = st.selectbox("Categorical column:", cat_cols, key="bar_c")
            top_n = st.slider("Top N values:", 5, 30, 15)
            fig = plot_bar_chart(df, cat_col, top_n)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough columns of the required type for this chart.")

    # ── Tab 5: Box Plots ───────────────────────────────────────────────────
    with tab5:
        st.subheader("Box Plot Comparison")
        if numeric_cols:
            selected_for_box = st.multiselect(
                "Select columns:", numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            if selected_for_box:
                fig = plot_box_plots(df, selected_for_box)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MACHINE LEARNING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Machine Learning":
    st.header("🤖 Machine Learning Module")

    if st.session_state.df_clean is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = st.session_state.df_clean

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Model Configuration")
        target_col = st.selectbox("🎯 Target column (what to predict):", df.columns.tolist())
        test_size = st.slider("Test set size:", 0.1, 0.4, 0.2, 0.05,
                               help="Fraction of data used for testing")

        # Show detected problem type
        if target_col:
            pt = detect_problem_type(df[target_col])
            if pt == "classification":
                st.success(f"🏷️ Detected: **Classification** task\nModel: Logistic Regression")
            else:
                st.info(f"📈 Detected: **Regression** task\nModel: Linear Regression")

        use_sample = st.checkbox("Use 25% sample for training")

        if st.button("🚀 Train Model", type="primary"):
            df_train = sample_data(df, 0.25) if use_sample else df
            with st.spinner("Training model..."):
                try:
                    result = train_model(df_train, target_col, test_size)
                    st.session_state.ml_result = result
                    st.session_state.target_col = target_col
                    st.success("✅ Model trained successfully!")
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")

    with col2:
        if st.session_state.ml_result is not None:
            result = st.session_state.ml_result
            pt = result["problem_type"]

            st.subheader("📊 Model Performance")

            # Metrics display
            metrics = result["metrics"]
            mcols = st.columns(len(metrics))
            for i, (k, v) in enumerate(metrics.items()):
                mcols[i].metric(k, v)

            st.divider()

            tab1, tab2, tab3 = st.tabs(["📈 Predictions", "🔑 Feature Importance", "📋 Details"])

            with tab1:
                y_test = result["y_test"]
                y_pred = result["y_pred"]

                if pt == "classification":
                    # Confusion matrix heatmap
                    cm = result.get("confusion_matrix")
                    if cm is not None:
                        le = result.get("label_encoder")
                        labels = [str(c) for c in le.classes_] if le else None
                        fig = px.imshow(
                            cm, text_auto=True, aspect="auto",
                            title="Confusion Matrix",
                            color_continuous_scale="Blues",
                            x=labels, y=labels
                        )
                        fig.update_layout(template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)

                else:  # regression
                    fig = px.scatter(
                        x=y_test, y=y_pred,
                        labels={"x": "Actual", "y": "Predicted"},
                        title="Actual vs Predicted Values",
                        template="plotly_white"
                    )
                    # Add perfect prediction line
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val], y=[min_val, max_val],
                        mode="lines", name="Perfect Prediction",
                        line=dict(color="red", dash="dash")
                    ))
                    st.plotly_chart(fig, use_container_width=True)

            with tab2:
                fi = result.get("feature_importance")
                if fi is not None and not fi.empty:
                    fig = px.bar(
                        fi.head(15), x="Importance", y="Feature",
                        orientation="h", title="Feature Importance (Absolute Coefficients)",
                        template="plotly_white",
                        color="Importance", color_continuous_scale="Purples"
                    )
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance not available.")

            with tab3:
                if pt == "classification" and result.get("classification_report"):
                    st.text("Classification Report:")
                    st.code(result["classification_report"])
                else:
                    st.write("**Problem Type:**", pt.capitalize())
                    st.write("**Features used:**", result.get("feature_names", []))
                    st.write("**Train size:**", metrics.get("Train Size"))
                    st.write("**Test size:**", metrics.get("Test Size"))

            # Generate ML insights
            ml_insights = generate_ml_insights(result, st.session_state.target_col)
            st.divider()
            st.subheader("💡 Model Insights")
            for insight in ml_insights:
                st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
        else:
            st.info("👈 Configure and train a model to see results here.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📐 Statistics":
    st.header("📐 Statistical Testing Module")

    if st.session_state.df_clean is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = st.session_state.df_clean
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    test_type = st.selectbox(
        "Select test type:",
        ["Z-Test (One Sample)", "Z-Test (Two Sample)", "Chi-Square Test of Independence"]
    )

    st.divider()

    if test_type == "Z-Test (One Sample)":
        st.subheader("🔬 One-Sample Z-Test")
        st.write("Tests whether a sample mean significantly differs from a hypothesized population mean.")

        col1, col2, col3 = st.columns(3)
        with col1:
            col = st.selectbox("Select column:", numeric_cols)
        with col2:
            pop_mean = st.number_input("Hypothesized population mean (H₀):",
                                        value=float(df[col].mean()) if col else 0.0)
        with col3:
            alpha = st.select_slider("Significance level (α):", [0.01, 0.05, 0.10], value=0.05)

        if st.button("Run Z-Test", type="primary"):
            result = z_test_one_sample(df[col].dropna().values, pop_mean, alpha)

            # Display result
            color = "🟢" if not result["is_significant"] else "🔴"
            st.subheader(f"{color} Result: **{result['decision']}**")
            st.markdown(result["explanation"])

            # Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Z-Statistic", result["z_stat"])
            c2.metric("P-Value", result["p_value"])
            c3.metric("Sample Size", result["sample_size"])

            # Visualization: normal curve with rejection regions
            x = np.linspace(-4, 4, 1000)
            from scipy.stats import norm
            y_norm = norm.pdf(x)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y_norm, fill="tozeroy", name="Normal Distribution",
                                      fillcolor="rgba(99,110,250,0.2)", line=dict(color="#636EFA")))

            # Rejection regions
            z_crit = norm.ppf(1 - alpha / 2)
            fig.add_vline(x=result["z_stat"], line_color="red", line_dash="dash",
                          annotation_text=f"Z={result['z_stat']}")
            fig.add_vline(x=z_crit, line_color="orange", line_dash="dot",
                          annotation_text=f"Z_crit=±{z_crit:.2f}")
            fig.add_vline(x=-z_crit, line_color="orange", line_dash="dot")
            fig.update_layout(title="Z-Test Visualization", template="plotly_white",
                              xaxis_title="Z-score", yaxis_title="Probability Density")
            st.plotly_chart(fig, use_container_width=True)

    elif test_type == "Z-Test (Two Sample)":
        st.subheader("🔬 Two-Sample Z-Test")
        st.write("Tests whether two group means are significantly different.")

        c1, c2 = st.columns(2)
        group_col = c1.selectbox("Grouping column (categorical):", cat_cols if cat_cols else df.columns.tolist())
        value_col = c2.selectbox("Value column (numeric):", numeric_cols)
        alpha = st.select_slider("α:", [0.01, 0.05, 0.10], value=0.05)

        if group_col and value_col:
            groups = df[group_col].dropna().unique()
            if len(groups) >= 2:
                c3, c4 = st.columns(2)
                group1 = c3.selectbox("Group 1:", groups, index=0)
                group2 = c4.selectbox("Group 2:", groups, index=min(1, len(groups)-1))

                if st.button("Run Two-Sample Z-Test", type="primary"):
                    d1 = df[df[group_col] == group1][value_col].dropna().values
                    d2 = df[df[group_col] == group2][value_col].dropna().values
                    result = z_test_two_sample(d1, d2, alpha)

                    color = "🟢" if not result["is_significant"] else "🔴"
                    st.subheader(f"{color} Result: **{result['decision']}**")
                    st.markdown(result["explanation"])

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Z-Statistic", result["z_stat"])
                    c2.metric("P-Value", result["p_value"])

                    # Box plot comparison
                    temp = pd.DataFrame({
                        "Group": [str(group1)] * len(d1) + [str(group2)] * len(d2),
                        value_col: list(d1) + list(d2)
                    })
                    fig = px.box(temp, x="Group", y=value_col,
                                 title=f"Distribution Comparison: {group1} vs {group2}",
                                 template="plotly_white",
                                 color="Group")
                    st.plotly_chart(fig, use_container_width=True)

    elif test_type == "Chi-Square Test of Independence":
        st.subheader("🔬 Chi-Square Test of Independence")
        st.write("Tests whether two categorical variables are statistically independent.")

        if len(cat_cols) < 2:
            st.warning("Need at least 2 categorical columns for Chi-Square test.")
        else:
            c1, c2, c3 = st.columns(3)
            col1_sel = c1.selectbox("Variable 1:", cat_cols, key="chi1")
            col2_sel = c2.selectbox("Variable 2:", cat_cols, index=min(1, len(cat_cols)-1), key="chi2")
            alpha = c3.select_slider("α:", [0.01, 0.05, 0.10], value=0.05, key="chi_a")

            if st.button("Run Chi-Square Test", type="primary"):
                result = chi_square_test(df, col1_sel, col2_sel, alpha)

                color = "🟢" if not result["is_significant"] else "🔴"
                st.subheader(f"{color} Result: **{result['decision']}**")
                st.markdown(result["explanation"])

                c1, c2, c3 = st.columns(3)
                c1.metric("Chi² Statistic", result["chi2_stat"])
                c2.metric("P-Value", result["p_value"])
                c3.metric("Degrees of Freedom", result["dof"])

                st.divider()
                st.subheader("Contingency Table")
                ct = result["contingency_table"]
                fig = px.imshow(ct, text_auto=True, aspect="auto",
                                title="Contingency Table Heatmap",
                                color_continuous_scale="Blues")
                st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: FINANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💰 Finance":
    st.header("💰 Finance Module")
    st.write("Portfolio optimization, Monte Carlo stock simulation, and risk metrics.")

    tab1, tab2, tab3 = st.tabs(["📈 Stock Simulation", "📊 Portfolio Optimization", "⚠️ Risk (VaR)"])

    # ── Tab 1: Stock Simulation ───────────────────────────────────────────────
    with tab1:
        st.subheader("Monte Carlo Stock Price Simulation")
        st.write("Simulates future stock prices using **Geometric Brownian Motion**.")

        c1, c2, c3, c4 = st.columns(4)
        init_price = c1.number_input("Initial Price ($)", 10.0, 10000.0, 100.0)
        mu = c2.number_input("Daily Drift (μ)", -0.01, 0.05, 0.0008, 0.0001, format="%.4f")
        sigma = c3.number_input("Daily Volatility (σ)", 0.001, 0.1, 0.02, 0.001, format="%.3f")
        n_sim = c4.slider("# Simulations", 10, 200, 50)

        days = st.slider("Simulation Period (days)", 30, 504, 252)

        if st.button("🎲 Run Simulation", type="primary"):
            with st.spinner("Running Monte Carlo simulation..."):
                sim = simulate_stock_price(init_price, mu, sigma, days, n_sim)

            # Key metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean Final Price", f"${sim['mean_final']}")
            c2.metric("Median Final Price", f"${sim['median_final']}")
            c3.metric("Annualized Return", f"{sim['annualized_return_pct']}%")
            c4.metric("Probability of Profit", f"{sim['prob_profit_pct']}%")

            st.metric("Value at Risk (5th percentile price)", f"${sim['var_95']}",
                      delta=f"{((sim['var_95'] / init_price) - 1) * 100:.2f}% from initial")

            fig = plot_stock_simulation(sim)
            st.plotly_chart(fig, use_container_width=True)

            # Final price distribution
            fig2 = px.histogram(
                x=sim["final_prices"], nbins=50,
                title="Distribution of Final Prices",
                labels={"x": "Final Price ($)"},
                template="plotly_white",
                color_discrete_sequence=["#636EFA"]
            )
            fig2.add_vline(x=sim["mean_final"], line_color="red", line_dash="dash",
                           annotation_text="Mean")
            fig2.add_vline(x=sim["var_95"], line_color="orange", line_dash="dash",
                           annotation_text="VaR 5%")
            st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 2: Portfolio Optimization ─────────────────────────────────────────
    with tab2:
        st.subheader("Portfolio Optimization (Efficient Frontier)")
        st.write("Monte Carlo simulation finds the **optimal portfolio** maximizing the Sharpe Ratio.")

        c1, c2 = st.columns(2)
        n_assets = c1.slider("Number of assets", 2, 8, 4)
        n_portfolios = c2.select_slider("# Random portfolios", [1000, 2000, 5000, 10000], value=5000)

        if st.button("🔍 Optimize Portfolio", type="primary"):
            with st.spinner("Running portfolio optimization..."):
                returns = generate_random_portfolio(n_assets=n_assets)
                opt = optimize_portfolio(returns, n_portfolios=n_portfolios)

            # Results
            st.divider()
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("⭐ Optimal Portfolio (Max Sharpe)")
                st.metric("Expected Annual Return", f"{opt['max_sharpe_return']}%")
                st.metric("Annual Volatility", f"{opt['max_sharpe_vol']}%")
                st.metric("Sharpe Ratio", opt["max_sharpe_ratio"])
                st.write("**Weights:**")
                weights_df = pd.DataFrame.from_dict(
                    opt["optimal_weights"], orient="index", columns=["Weight"]
                )
                fig_pie = px.pie(weights_df, values="Weight", names=weights_df.index,
                                  title="Optimal Portfolio Weights", template="plotly_white")
                st.plotly_chart(fig_pie, use_container_width=True)

            with c2:
                st.subheader("🛡️ Minimum Volatility Portfolio")
                st.metric("Expected Annual Return", f"{opt['min_vol_return']}%")
                st.metric("Annual Volatility", f"{opt['min_vol_vol']}%")
                weights_df2 = pd.DataFrame.from_dict(
                    opt["min_vol_weights"], orient="index", columns=["Weight"]
                )
                fig_pie2 = px.pie(weights_df2, values="Weight", names=weights_df2.index,
                                   title="Min Volatility Weights", template="plotly_white")
                st.plotly_chart(fig_pie2, use_container_width=True)

            st.divider()
            fig_ef = plot_efficient_frontier(opt)
            st.plotly_chart(fig_ef, use_container_width=True)

    # ── Tab 3: VaR ─────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Value at Risk (VaR) Calculator")
        st.write("Estimate the maximum expected loss at a given confidence level.")

        c1, c2, c3 = st.columns(3)
        investment = c1.number_input("Portfolio Value ($)", 1000.0, 10_000_000.0, 100_000.0, step=1000.0)
        confidence = c2.select_slider("Confidence Level", [0.90, 0.95, 0.99], value=0.95)
        vol_assumption = c3.slider("Daily Volatility %", 0.5, 5.0, 1.5, 0.1) / 100

        if st.button("⚠️ Calculate VaR", type="primary"):
            # Generate synthetic daily returns
            np.random.seed(42)
            daily_returns = np.random.normal(0.0005, vol_assumption, 1000)
            var_result = calculate_var(daily_returns, confidence, investment)

            st.subheader("Risk Metrics")
            for k, v in var_result.items():
                st.write(f"**{k}:** {v}")

            # Distribution plot
            fig = px.histogram(
                x=daily_returns * investment, nbins=60,
                title=f"Daily P&L Distribution (${investment:,.0f} portfolio)",
                labels={"x": "Daily P&L ($)"},
                template="plotly_white",
                color_discrete_sequence=["#636EFA"]
            )
            fig.add_vline(
                x=var_result["VaR ($)"],
                line_color="red", line_dash="dash",
                annotation_text=f"VaR ({confidence*100:.0f}%): ${var_result['VaR ($)']:,.0f}"
            )
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: AI INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💡 AI Insights":
    st.header("💡 AI Insight Engine")

    if st.session_state.df_clean is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = st.session_state.df_clean
    domain = st.session_state.domain

    if st.button("🔍 Generate Insights", type="primary"):
        with st.spinner("Analyzing your dataset..."):
            insights = generate_insights(df, domain)
            st.session_state.insights = insights

    if st.session_state.insights:
        st.subheader(f"📊 Insights for {domain} Domain")
        st.write(f"Generated **{len(st.session_state.insights)} insights** from your dataset:")

        for i, insight in enumerate(st.session_state.insights, 1):
            st.markdown(f"""
            <div class="insight-card">
                <strong>#{i}</strong> {insight}
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.ml_result:
            st.divider()
            st.subheader("🤖 ML Model Insights")
            ml_insights = generate_ml_insights(
                st.session_state.ml_result,
                st.session_state.target_col or "Target"
            )
            for insight in ml_insights:
                st.markdown(f"""
                <div class="insight-card">
                    {insight}
                </div>
                """, unsafe_allow_html=True)

        # Quick dataset summary widget
        st.divider()
        st.subheader("📋 Dataset Snapshot")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Features", df.shape[1])
        col3.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
        col4.metric("Categorical Columns", len(df.select_dtypes(include=["object"]).columns))

        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            fig = px.box(
                numeric_df.melt(var_name="Feature", value_name="Value").head(10000),
                x="Feature", y="Value",
                title="Feature Value Distribution Overview",
                template="plotly_white",
                color="Feature"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Click 'Generate Insights' to analyze your dataset.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: AI MENTOR CHAT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💬 AI Mentor Chat":
    st.header("💬 AI Mentor Chat")
    st.write("Ask questions about your data, ML concepts, statistics, or business analytics.")

    mentor_disabled = False
    st.info("🧠 AI Mentor is running in local mode (no external API).")

    # Chat display
    chat_container = st.container()

    with chat_container:
        if not st.session_state.chat_history:
            st.info("👋 Hello! I'm your AI Data Science Mentor. Ask me anything about your data, models, or analytics concepts!")
        else:
            for msg in st.session_state.chat_history:
                role = msg["role"]
                content = msg["content"]
                # Strip context prefix for display
                if role == "user" and "[Dataset Context:" in content:
                    display_content = content.split("User Question: ")[-1]
                else:
                    display_content = content

                with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
                    st.markdown(display_content)

    st.divider()

    # Suggestion chips
    st.write("**💡 Suggested questions:**")
    suggestions = [
        "What is a Z-test and when should I use it?",
        "How do I interpret an R² score of 0.75?",
        "What does a high correlation mean for my analysis?",
        "How can I improve my model accuracy?",
        "Explain the Sharpe ratio in simple terms",
        "What is overfitting and how to prevent it?"
    ]

    # 3-column layout for suggestions
    suggestion_cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with suggestion_cols[i % 3]:
            if st.button(
                suggestion,
                key=f"sug_{i}",
                use_container_width=True,
                disabled=mentor_disabled,
            ):
                # Build context from session
                context = build_context_from_session(st.session_state)
                with st.spinner("Thinking..."):
                    reply = ask_mentor(suggestion, context, st.session_state.chat_history)

                # Append to history (keeping history clean for API)
                full_q = f"[Dataset Context: {context}]\n\nUser Question: {suggestion}" if context else suggestion
                st.session_state.chat_history.append({"role": "user", "content": full_q})
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()

    st.divider()

    # Free text input
    user_input = st.chat_input("Ask your AI Mentor a question...", disabled=mentor_disabled)
    if user_input:
        context = build_context_from_session(st.session_state)
        with st.spinner("🤔 Thinking..."):
            reply = ask_mentor(user_input, context, st.session_state.chat_history)

        full_q = f"[Dataset Context: {context}]\n\nUser Question: {user_input}" if context else user_input
        st.session_state.chat_history.append({"role": "user", "content": full_q})
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem;">
    🤖 <strong>AI Business Intelligence & Decision Copilot</strong> | 
    Built with Streamlit · Pandas · Scikit-learn · Plotly
</div>
""", unsafe_allow_html=True)
