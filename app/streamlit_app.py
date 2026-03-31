"""
app/streamlit_app.py

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
import yfinance as yf
from datetime import date

from src.features.technical_indicators import add_technical_indicators
from src.models.train import train_logreg_model
from src.models.evaluate import evaluate_model, backtest_strategy, sharpe_ratio, max_drawdown
from src.llm.explainer import explain_prediction, explain_backtest
from src.utils.config import MODEL_DIR, RAW_DATA_DIR

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Portfolio Manager",
    page_icon="📈",
    layout="wide",
)

st.title("📈 ML Portfolio Manager")
st.caption("Machine-learning driven stock prediction & backtesting dashboard")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    ticker = st.text_input("Stock Ticker", value="TSLA").upper()
    start_date = st.date_input("Start Date", value=date(2020, 1, 1))
    end_date = st.date_input("End Date", value=date.today())

    if st.button("🔄 Download & Retrain", use_container_width=True):
        with st.spinner(f"Downloading {ticker} data..."):
            df_raw = yf.download(ticker, start=start_date, end=end_date)
            df_raw.reset_index(inplace=True)
            os.makedirs(RAW_DATA_DIR, exist_ok=True)
            os.makedirs(MODEL_DIR, exist_ok=True)
            csv_path = f"{RAW_DATA_DIR}/{ticker}.csv"
            df_raw.to_csv(csv_path, index=False)
            st.success(f"Saved {len(df_raw)} rows to {csv_path}")

        with st.spinner("Training model..."):
            train_logreg_model(f"{ticker}.csv")
            st.success("Model trained and saved!")
            st.rerun()

    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. Downloads OHLCV data via yfinance\n"
        "2. Adds SMA, lag & return features\n"
        "3. Trains a Logistic Regression classifier\n"
        "4. Predicts next-day direction\n"
        "5. Backtests a long-only strategy"
    )

# ── Load data & model ────────────────────────────────────────────────────────
model_path = f"{MODEL_DIR}/logreg_model.pkl"
scaler_path = f"{MODEL_DIR}/scaler.pkl"
csv_path = f"{RAW_DATA_DIR}/{ticker}.csv"

model_exists = os.path.exists(model_path) and os.path.exists(scaler_path)
data_exists = os.path.exists(csv_path)

if not data_exists:
    st.info(f"No data found for **{ticker}**. Use the sidebar to download & train.")
    st.stop()

df = pd.read_csv(csv_path)

# Flatten MultiIndex columns from yfinance if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join([c for c in col if c]).strip() for col in df.columns]

# Normalise column names — yfinance sometimes appends ticker
rename_map = {}
for col in df.columns:
    for base in ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']:
        if col.startswith(base):
            rename_map[col] = base
df.rename(columns=rename_map, inplace=True)

df['Date'] = pd.to_datetime(df['Date'])
df = add_technical_indicators(df)

features = ['SMA_5', 'SMA_10', 'Close_lag_1', 'Close_lag_2', 'Return']

if not model_exists:
    st.warning("No trained model found. Click **Download & Retrain** in the sidebar.")
    st.stop()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

X_scaled = scaler.transform(df[features])
df['Prediction'] = model.predict(X_scaled)
df['Pred_Prob'] = model.predict_proba(X_scaled)[:, 1]

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Price & Signals", "🏆 Backtest", "🔍 Model Metrics"])

# ── Tab 1: Price chart + latest prediction ───────────────────────────────────
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"{ticker} Close Price + Signals")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', line=dict(color='#636EFA')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_5'], name='SMA 5', line=dict(dash='dot', color='orange')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_10'], name='SMA 10', line=dict(dash='dot', color='green')))

        buys = df[df['Prediction'] == 1]
        fig.add_trace(go.Scatter(
            x=buys['Date'], y=buys['Close'],
            mode='markers', name='Predict UP',
            marker=dict(symbol='triangle-up', size=6, color='lime')
        ))
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Tomorrow's Prediction")
        latest = df.iloc[-1]
        pred = int(latest['Prediction'])
        prob = float(latest['Pred_Prob'])

        if pred == 1:
            st.success(f"### 📈 UP  ({prob:.0%} confidence)")
        else:
            st.error(f"### 📉 DOWN  ({1 - prob:.0%} confidence)")

        st.metric("Last Close", f"${latest['Close']:.2f}")
        st.metric("Daily Return", f"{latest['Return']*100:.2f}%")
        st.metric("SMA 5", f"${latest['SMA_5']:.2f}")
        st.metric("SMA 10", f"${latest['SMA_10']:.2f}")

        st.divider()
        explanation = explain_prediction(
            ticker, pred, prob if pred == 1 else 1 - prob, latest.to_dict()
        )
        st.markdown(explanation)

# ── Tab 2: Backtest ──────────────────────────────────────────────────────────
with tab2:
    bt = backtest_strategy(df)
    sr = sharpe_ratio(bt['Strategy_Return'])
    md = max_drawdown(bt['Cumulative_Strategy'])
    final_strat = bt['Cumulative_Strategy'].iloc[-1]
    final_market = bt['Cumulative_Market'].iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategy Return", f"{(final_strat - 1)*100:.1f}%")
    c2.metric("Buy & Hold Return", f"{(final_market - 1)*100:.1f}%")
    c3.metric("Sharpe Ratio", str(sr))
    c4.metric("Max Drawdown", f"{md:.1%}")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=bt['Date'], y=bt['Cumulative_Strategy'], name='ML Strategy', line=dict(color='lime')))
    fig2.add_trace(go.Scatter(x=bt['Date'], y=bt['Cumulative_Market'], name='Buy & Hold', line=dict(color='royalblue', dash='dash')))
    fig2.update_layout(title="Cumulative Returns", height=400, yaxis_title="Growth of $1")
    st.plotly_chart(fig2, use_container_width=True)

    st.info(explain_backtest(sr, md, final_strat, final_market))

# ── Tab 3: Model metrics ─────────────────────────────────────────────────────
with tab3:
    metrics = evaluate_model(df['Target'], df['Prediction'], df['Pred_Prob'])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", metrics['accuracy'])
    c2.metric("ROC-AUC", metrics['roc_auc'])
    c3.metric("Precision", metrics['precision'])
    c4.metric("Recall", metrics['recall'])
    c5.metric("F1", metrics['f1'])

    st.subheader("Confusion Matrix")
    cm = metrics['confusion_matrix']
    cm_df = pd.DataFrame(cm, index=['Actual Down', 'Actual Up'], columns=['Pred Down', 'Pred Up'])
    fig3 = px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues',
                     title="Confusion Matrix")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Prediction Probability Distribution")
    fig4 = px.histogram(df, x='Pred_Prob', color='Target',
                        nbins=40, barmode='overlay',
                        labels={'Pred_Prob': 'Predicted Probability (Up)', 'Target': 'Actual'},
                        color_discrete_map={0: 'tomato', 1: 'steelblue'})
    st.plotly_chart(fig4, use_container_width=True)