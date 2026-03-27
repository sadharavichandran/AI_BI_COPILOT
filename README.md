# 🤖 AI Business Intelligence & Decision Copilot

A complete end-to-end Machine Learning and Business Intelligence system built with Python and Streamlit.
---

## 📁 SAMPLES 

<img width="1920" height="1008" alt="Screenshot 2026-03-27 161546" src="https://github.com/user-attachments/assets/c981e081-9a10-4349-b782-3164f9d545b3" />

<img width="1920" height="1008" alt="Screenshot 2026-03-27 161654" src="https://github.com/user-attachments/assets/7745b97f-ba6d-440c-b22c-b817a8798bb8" />

<img width="1920" height="1008" alt="Screenshot 2026-03-27 161808" src="https://github.com/user-attachments/assets/ba12b479-345b-4076-b435-68d40c5f072b" />

<img width="1920" height="1008" alt="Screenshot 2026-03-27 161834" src="https://github.com/user-attachments/assets/6d1b8586-057c-437d-b867-0349ba00b129" />

<img width="1920" height="1008" alt="image" src="https://github.com/user-attachments/assets/c5b1a08d-62a8-4052-961c-bc60d160e48c" />





## 📁 Project Structure

```
ai_bi_copilot/
│
├── app.py                      # Main Streamlit application (entry point)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
└── modules/
    ├── __init__.py             # Package exports
    ├── data_cleaner.py         # Data cleaning: NaN handling, deduplication, outliers
    ├── data_analyzer.py        # EDA: stats, correlation, charts, heatmaps
    ├── ml_engine.py            # Auto ML: classification & regression pipelines
    ├── stats_tester.py         # Z-test, Chi-square hypothesis testing
    ├── finance_engine.py       # Portfolio optimization, stock simulation, VaR
    ├── insight_engine.py       # Auto-generated AI business insights
    └── ai_mentor.py            # AI Mentor chatbot (Anthropic API)
```

---

## ✨ Features

| Module              | What it does                                                |
| ------------------- | ----------------------------------------------------------- |
| 📤 Data Upload      | Upload any CSV, auto-detect column types & data quality     |
| 🧹 Data Cleaning    | Handle NaN (mean/median/mode), remove duplicates & outliers |
| 📊 Data Analysis    | Summary stats, correlation heatmap, distributions, charts   |
| 🎲 Sampling         | 25% random sample for fast exploration                      |
| 🤖 Machine Learning | Auto-detect classification/regression, train & evaluate     |
| 📐 Statistics       | Z-test (one/two sample), Chi-Square hypothesis testing      |
| 💰 Finance          | Monte Carlo stock simulation, portfolio optimization, VaR   |
| 💡 AI Insights      | Domain-aware auto-generated insights                        |
| 💬 AI Mentor        | Chat with Claude for explanations & advice                  |

---

## 🚀 Installation & Setup

### 1. Clone / Download the project

```bash
git clone <repo_url>
cd ai_bi_copilot
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 🌐 Domain Modes

Select a domain in the sidebar to get specialized insights:

- **General** — Generic analysis for any dataset
- **Student Analysis** — Education metrics, grade correlation, performance insights
- **Healthcare Prediction** — Patient demographics, diagnosis patterns
- **Retail Business Analytics** — Revenue trends, top products, sales analysis
- **Financial Modeling** — Return analysis, risk metrics, portfolio insights

---

## 📊 Example Datasets to Try

You can use any CSV. Here are some ideas:

| Domain     | Suggested Dataset                              |
| ---------- | ---------------------------------------------- |
| Student    | Kaggle: Student Performance Dataset            |
| Healthcare | Kaggle: Heart Disease UCI                      |
| Retail     | Kaggle: Superstore Sales Dataset               |
| Finance    | Yahoo Finance historical prices (CSV export)   |
| General    | Any CSV with mixed numeric/categorical columns |

---

## 🤖 AI Mentor Chat

The AI Mentor runs in **local mode** and does not require any external API.

### Local setup (Windows PowerShell)

Run the app directly:

```powershell
python -m streamlit run app.py
```

---

## 🔧 Tech Stack

| Tool                        | Purpose                            |
| --------------------------- | ---------------------------------- |
| **Streamlit**               | Dashboard UI                       |
| **Pandas**                  | Data manipulation                  |
| **NumPy**                   | Numerical computation              |
| **Scikit-learn**            | ML models & preprocessing          |
| **Matplotlib / Seaborn**    | Static plots (correlation heatmap) |
| **Plotly**                  | Interactive charts                 |
| **SciPy**                   | Statistical tests                  |
| **Local Rule-based Mentor** | AI Mentor chatbot                  |

---

## 📝 Notes

- **No hardcoded dataset** — the system adapts to any CSV you upload
- **Fully modular** — each feature is in its own file under `modules/`
- **Production-ready structure** — clean separation of concerns
- Minimum recommended dataset: **50+ rows**, **3+ columns**

---

## 🐛 Troubleshooting

| Issue                          | Fix                                                                        |
| ------------------------------ | -------------------------------------------------------------------------- |
| `ModuleNotFoundError`          | Run `pip install -r requirements.txt` again                                |
| `streamlit: command not found` | Add Python Scripts to PATH or use `python -m streamlit run app.py`         |
| File encoding error on upload  | Change encoding to `latin-1` in the upload settings                        |
| Model training fails           | Ensure your dataset has numeric columns and the target column has variance |
