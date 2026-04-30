# ML Stock Portfolio Manager

> An algorithmic trading system that uses machine learning to predict daily stock price movements, simulate trading strategies through walk-forward backtesting, and explain performance in plain English using a large language model.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the App](#running-the-app)
- [How It Works](#how-it-works)
- [Models & Feature Sets](#models--feature-sets)
- [Performance Metrics](#performance-metrics)
- [Team](#team)

---

## Overview

This project was developed as a Software Engineering project. It explores whether machine learning models can outperform a simple buy-and-hold benchmark when applied to daily stock price prediction.

The system ingests historical stock data (CSV format), engineers a set of technical indicators as features, trains one of several classifiers, and then simulates a trading strategy over a held-out test period. A walk-forward methodology is used to ensure the model never trains on data from the test window, preventing lookahead bias. Results are surfaced through an interactive Streamlit dashboard, and an integrated LLM can generate a natural-language explanation of each strategy's performance.

---

## Features

- **Multi-model support** — Logistic Regression, Random Forest, Gradient Boosting, SVM, and a Dummy baseline
- **Walk-forward backtesting** — strict train/test separation by cutoff date; no data leakage
- **Multi-ticker portfolio** — capital split evenly across all tickers that successfully complete backtesting
- **Configurable feature sets** — V1 and V2 feature groups for apples-to-apples model comparison
- **LLM-powered analysis** — pass backtest metrics to an LLM to generate stakeholder-readable summaries
- **Interactive dashboard** — explore equity curves, trade logs, and experiment results via Streamlit

---

## Project Structure

```
ml-portfolio-manager/
├── uml_class_diagram.svg           # UML class diagram
├── prompts.txt                     # LLM prompt templates (plain text copy)
├── pytest.ini                      # pytest configuration
├── requirements.txt
├── README.md
├── app/
│   └── streamlit_app.py            # Main Streamlit dashboard
├── tests/
│   ├── test_walk_forward.py        # Backtesting engine tests
│   ├── test_evaluate.py            # Metrics function tests
│   └── test_experiments.py        # Experiment runner tests
├── models/                         # Serialized model artifacts (.pkl)
│   ├── logreg_model.pkl
│   ├── scaler.pkl
│   └── <TICKER>_logreg_model.pkl   # Per-ticker trained models (AAPL, TSLA, etc.)
├── data/
│   ├── portfolio/
│   │   └── portfolio.json          # Saved portfolio state
│   └── raw/                        # Historical OHLCV CSVs
│       ├── AAPL.csv
│       ├── AMZN.csv
│       ├── GME.csv
│       ├── GOOGL.csv
│       ├── MSFT.csv
│       ├── NASDAQ.csv
│       ├── NDX.csv
│       ├── NVDA.csv
│       ├── SPY.csv
│       └── TSLA.csv
└── src/
    ├── data/
    │   ├── loader.py               # CSV ingestion and validation
    │   └── preprocessing.py        # Data cleaning and normalization
    ├── features/
    │   ├── technical_indicators.py # SMA, RSI, MACD, volatility, lag features
    │   └── feature_builder.py      # Feature set assembly (V1 / V2)
    ├── llm/
    │   ├── explainer.py            # LLM API integration
    │   └── prompts.py              # Prompt construction for performance summaries
    ├── models/
    │   ├── train.py                # Model training and serialization
    │   ├── predict.py              # Next-day signal generation
    │   ├── evaluate.py             # Classification metrics, Sharpe ratio, drawdown
    │   ├── walk_forward.py         # Walk-forward backtesting engine
    │   └── experiments.py          # Multi-model × multi-feature-set runner
    ├── portfolio/
    │   ├── __init__.py
    │   └── portfolio.py            # Portfolio state management
    └── utils/
        ├── config.py               # Paths, constants, global settings
        └── logger.py               # Logging configuration
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip
- ollama
### Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   > If you are on a system where `pip` points to Python 2, use `pip3` instead.

   Install ollama from https://ollama.com
   then run "ollama pull llama3"
   then continue below

### Running the App

```bash
streamlit run app/streamlit_app.py
```

The dashboard will open automatically in your browser. From there you can upload stock CSVs, select a model and feature set, set a cutoff date, and run a backtest.

---

## How It Works

1. **Data loading** — Raw OHLCV CSV files are validated and parsed into a standard DataFrame format.
2. **Feature engineering** — Technical indicators (SMA, RSI, MACD, volatility, volume change, lagged closes) are computed and appended as columns. The binary target label is `1` if the next day's close is higher, `0` otherwise.
3. **Walk-forward split** — All rows before the cutoff date form the training set; all rows from the cutoff date onward form the test set.
4. **Model training** — The selected classifier is fit on the scaled training features only.
5. **Simulation** — For each day in the test period, the model outputs a probability. If the confidence exceeds the threshold, a BUY or SELL order is triggered. Open positions are force-closed at the end of the test window.
6. **Evaluation** — Total return, alpha over buy-and-hold, Sharpe ratio, maximum drawdown, and win rate are computed and displayed.
7. **LLM explanation** — Metrics are passed to a language model which returns a plain-English summary of the strategy's strengths and weaknesses.

---

## Models & Feature Sets

| Model | Notes |
|---|---|
| Logistic Regression | Fast, interpretable baseline |
| Random Forest | 100 trees; handles non-linear patterns |
| Gradient Boosting | Sequential boosting, depth-3 trees |
| SVM (RBF kernel) | Effective in high-dimensional feature spaces |
| Dummy Classifier | Always predicts majority class — sets the floor |

| Feature Set | Included Features |
|---|---|
| Baseline | SMA\_5, SMA\_10, Close\_lag\_1, Close\_lag\_2, Return |
| With RSI | + RSI\_14 |
| With Volume | + Volume\_change, Volatility\_10 |
| Full v1 | All of the above + MACD |
| Full v2 | Extended set from `FEATURE_SETS_V2` in `technical_indicators.py` |

Key parameters that can be tuned from the dashboard:

- `confidence_threshold` — minimum model probability required to execute a trade (default `0.55`)
- `position_size_pct` — fraction of available capital deployed per trade (default `0.95`)
- `starting_cash` — initial portfolio value per ticker (default `$100,000`)

---

## Performance Metrics

| Metric | Description |
|---|---|
| Total Return % | Portfolio gain over the full test period |
| B&H Return % | Equivalent buy-and-hold return for comparison |
| Alpha | Strategy return minus buy-and-hold return |
| Sharpe Ratio | Annualized risk-adjusted return (252 trading days) |
| Max Drawdown | Largest peak-to-trough portfolio decline |
| Win Rate % | Percentage of closed trades that were profitable |
| Train ROC-AUC | In-sample classifier quality (higher = better-fit model) |

---

## Team

| Armin Omidvar and Seth Sorrentino| 


> *Built with Python · scikit-learn · Streamlit · pandas*