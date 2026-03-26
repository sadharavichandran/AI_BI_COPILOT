# modules/ai_mentor.py
# AI Mentor Chat: local-only guidance (no external API)

SYSTEM_PROMPT = """You are an AI Data Science and Business Intelligence Mentor.
Your role is to:
1. Explain machine learning concepts in simple, clear language
2. Help users understand their data analysis results
3. Provide actionable business recommendations
4. Answer questions about statistics, data science, and business analytics
5. Give step-by-step explanations when asked

Keep responses concise but informative. Use bullet points, emojis, and bold text for clarity.
If asked about a specific dataset result, interpret it in business-friendly terms.
Always be encouraging and educational."""


def has_mentor_api_credentials() -> bool:
    """Compatibility helper: no API credentials required in local mode."""
    return False


def _ask_local(question: str, context: str = "") -> str:
    """Return a lightweight offline response when API is unavailable."""
    q = question.lower()

    if "z-test" in q or "z test" in q:
        return (
            "📌 **Z-test** use pannunga when sample size usually >= 30 and population variance known/assumed. "
            "It checks whether observed mean difference is statistically significant (p-value < 0.05 na significant)."
        )

    if "r²" in q or "r2" in q or "r-squared" in q:
        return (
            "📈 **R²** model explain panna mudiyura variance proportion. Example: R² = 0.75 means target variation-la ~75% model explain panrathu. "
            "Higher is generally better, but overfitting check pannunga."
        )

    if "correlation" in q:
        return (
            "🔗 High correlation means two variables together move agudhu; causal relation nu confirm panna mudiyadhu. "
            "Multicollinearity issue irukka regression-ku check pannunga."
        )

    if "overfitting" in q:
        return (
            "🛡️ **Overfitting**: train data-la super performance, new data-la weak performance. "
            "Fix: cross-validation, regularization, feature selection, simpler model, more data."
        )

    if "accuracy" in q or "improve" in q:
        return (
            "🚀 Accuracy improve panna: data cleaning, feature engineering, class imbalance handling, "
            "hyperparameter tuning, and suitable model selection try pannunga."
        )

    base = (
        "🤖 API illaamal local mentor mode-la irukken. Naan core data science guidance kudukka mudiyum: "
        "EDA interpretation, ML metrics explanation, statistics basics, and next-step recommendations."
    )
    if context:
        return f"{base}\n\n📊 Context detected: {context}"
    return base


def ask_mentor(question: str, context: str = "", chat_history: list = None) -> str:
    """
    Send a question to the AI Mentor and return the response.

    Parameters:
        question    : The user's question
        context     : Optional context about the current dataset/analysis
        chat_history: List of {"role": "user"/"assistant", "content": "..."} dicts
    """
    if chat_history is None:
        chat_history = []

    return _ask_local(question, context)


def build_context_from_session(session_state) -> str:
    """
    Build a brief context string from Streamlit session state to pass to the mentor.
    """
    parts = []
    if hasattr(session_state, "df") and session_state.df is not None:
        df = session_state.df
        parts.append(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        parts.append(f"Columns: {', '.join(df.columns.tolist()[:10])}")

    if hasattr(session_state, "ml_result") and session_state.ml_result:
        r = session_state.ml_result
        parts.append(f"ML Model: {r.get('problem_type', 'unknown')} task")
        parts.append(f"Metrics: {r.get('metrics', {})}")

    if hasattr(session_state, "domain") and session_state.domain:
        parts.append(f"Domain: {session_state.domain}")

    return " | ".join(parts) if parts else ""
