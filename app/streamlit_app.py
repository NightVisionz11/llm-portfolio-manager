"""
app/streamlit_app.py
Run with:  streamlit run app/streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
import yfinance as yf
from datetime import date, timedelta

from src.features.technical_indicators import add_technical_indicators
from src.models.train import train_logreg_model
from src.models.evaluate import evaluate_model, backtest_strategy, sharpe_ratio, max_drawdown
from src.models.walk_forward import run_walk_forward, run_multi_ticker_walk_forward
from src.llm.explainer import (
    explain_prediction, explain_backtest,
    ollama_available, list_ollama_models,
    _rule_based_prediction, _rule_based_backtest,
)
from src.models.experiments import run_experiments, MODEL_REGISTRY, FEATURE_SETS
from src.utils.config import MODEL_DIR, RAW_DATA_DIR
from src.portfolio.portfolio import (
    load_portfolio, reset_portfolio, execute_signal, get_portfolio_summary,
)

st.set_page_config(page_title="ML Portfolio Manager", page_icon="📈", layout="wide")
st.title("📈 ML Portfolio Manager")
st.caption("Machine-learning driven stock prediction, backtesting & paper trading")

FEATURES = ['SMA_5', 'SMA_10', 'Close_lag_1', 'Close_lag_2', 'Return']


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    rename_map = {col: base for col in df.columns
                  for base in ['Open','High','Low','Close','Volume','Date']
                  if col.startswith(base)}
    df.rename(columns=rename_map, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def load_and_predict(ticker: str):
    csv_path   = f"{RAW_DATA_DIR}/{ticker}.csv"
    model_path = f"{MODEL_DIR}/{ticker}_logreg_model.pkl"
    scaler_path= f"{MODEL_DIR}/{ticker}_scaler.pkl"

    if not os.path.exists(csv_path):
        return None, None, None

    df = normalise_df(pd.read_csv(csv_path))
    df = add_technical_indicators(df)

    df = df.dropna(subset=FEATURES)
    if df.empty:
        print(f"[{ticker}] No usable rows after indicators — skipping.")
        return None, None, None

    if not os.path.exists(model_path):
        return df, None, None

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    X      = scaler.transform(df[FEATURES])
    df['Prediction'] = model.predict(X)
    df['Pred_Prob']  = model.predict_proba(X)[:, 1]
    return df, model, scaler


def get_current_price(ticker: str):
    try:
        data = yf.download(ticker, period="2d", progress=False)
        if not data.empty:
            c = data['Close']
            if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
            return float(c.iloc[-1])
    except Exception:
        pass
    return None


def get_trained_tickers() -> list:
    if not os.path.exists(MODEL_DIR):
        return []
    return [f.replace("_logreg_model.pkl","")
            for f in os.listdir(MODEL_DIR)
            if f.endswith("_logreg_model.pkl")]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    ticker     = st.text_input("Stock Ticker", value="TSLA").upper()
    start_date = st.date_input("Start Date", value=date(2020, 1, 1))
    end_date   = st.date_input("End Date",   value=date.today())

    if st.button("🔄 Download & Retrain", use_container_width=True):
        with st.spinner(f"Downloading {ticker}..."):
            df_raw = yf.download(ticker, start=start_date, end=end_date)
            df_raw.reset_index(inplace=True)
            os.makedirs(RAW_DATA_DIR, exist_ok=True)
            os.makedirs(MODEL_DIR,    exist_ok=True)
            df_raw.to_csv(f"{RAW_DATA_DIR}/{ticker}.csv", index=False)
        with st.spinner("Training model..."):
            train_logreg_model(f"{ticker}.csv", ticker=ticker)
            st.success(f"✅ {ticker} ready!")
            st.rerun()

    st.divider()
    st.header("💼 Portfolio")
    confidence_threshold = st.slider("Min confidence to trade", 0.50, 0.95, 0.55, 0.01)
    position_size        = st.slider("Position size (% of portfolio)", 0.05, 0.30, 0.10, 0.01)

    if st.button("⚡ Run Signals on All Tickers", use_container_width=True):
        st.session_state["run_signals"] = True

    st.divider()
    if st.button("🔁 Reset Portfolio", use_container_width=True):
        reset_portfolio()
        st.success("Portfolio reset to $100,000")
        st.rerun()
    starting_cash = st.number_input("Starting cash ($)", value=100_000, step=5_000)
    if st.button("Set Starting Cash", use_container_width=True):
        reset_portfolio(starting_cash=float(starting_cash))
        st.rerun()

    # ── LLM Explainer ─────────────────────────────────────────────────────────
    st.divider()
    st.header("🤖 LLM Explainer")

    _ollama_running = ollama_available()

    if _ollama_running:
        st.success("✅ Ollama is running")
        _available_models = list_ollama_models()
        ollama_model = st.selectbox(
            "Model",
            options=_available_models,
            help="Select which locally installed Ollama model to use for explanations.",
        )
    else:
        st.warning("⚠️ Ollama not detected")
        st.caption(
            "To enable AI explanations:\n"
            "1. Download [Ollama](https://ollama.com)\n"
            "2. Run: `ollama pull llama3.2`\n"
            "3. Restart the app"
        )
        ollama_model = "llama3.2"  # default — won't be called if Ollama is offline

    use_llm = st.toggle(
        "Use LLM explanations",
        value=_ollama_running,
        disabled=not _ollama_running,
        help="Uncheck to always use the fast rule-based template instead.",
    )


# ── Load data ─────────────────────────────────────────────────────────────────
df, model, scaler = load_and_predict(ticker)

if df is None:
    st.info(f"No data for **{ticker}**. Use the sidebar to Download & Retrain.")
    st.stop()
if model is None:
    st.warning(f"No trained model for **{ticker}**. Click Download & Retrain.")
    st.stop()


# ── Run portfolio signals ─────────────────────────────────────────────────────
if st.session_state.get("run_signals"):
    st.session_state["run_signals"] = False
    portfolio_state = load_portfolio()
    signal_log = []
    trained = get_trained_tickers()
    if not trained:
        st.warning("No trained models found.")
    else:
        for t in trained:
            t_df, t_model, _ = load_and_predict(t)
            if t_df is None or t_model is None:
                continue
            latest = t_df.iloc[-1]
            pred   = int(latest['Prediction'])
            prob   = float(latest['Pred_Prob'])
            price  = get_current_price(t) or float(latest['Close'])
            trade  = execute_signal(portfolio_state, t, pred, prob, price,
                                    confidence_threshold=confidence_threshold,
                                    position_size_pct=position_size)
            signal_log.append({
                "Ticker":     t,
                "Prediction": "UP 📈" if pred == 1 else "DOWN 📉",
                "Confidence": f"{prob:.0%}",
                "Price":      f"${price:.2f}",
                "Action":     trade["action"] if trade else "HOLD",
            })
        st.success(f"✅ Signals run on {len(trained)} ticker(s).")
        st.dataframe(pd.DataFrame(signal_log), use_container_width=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Price & Signals", "🏆 Backtest", "🔍 Model Metrics",
    "💼 Portfolio", "🧪 Walk-Forward Test", "🔬 Experiments"
])

# ── Tab 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'],  name='Close', line=dict(color='#636EFA')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_5'],  name='SMA 5', line=dict(dash='dot', color='orange')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_10'], name='SMA 10',line=dict(dash='dot', color='green')))
        buys = df[df['Prediction'] == 1]
        fig.add_trace(go.Scatter(x=buys['Date'], y=buys['Close'], mode='markers',
                                 name='Predict UP', marker=dict(symbol='triangle-up', size=6, color='lime')))
        fig.update_layout(height=420, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Tomorrow's Prediction")
        latest = df.iloc[-1]
        pred   = int(latest['Prediction'])
        prob   = float(latest['Pred_Prob'])
        if pred == 1:
            st.success(f"### 📈 UP  ({prob:.0%} confidence)")
        else:
            st.error(f"### 📉 DOWN  ({1-prob:.0%} confidence)")
        st.metric("Last Close",   f"${latest['Close']:.2f}")
        st.metric("Daily Return", f"{latest['Return']*100:.2f}%")
        st.metric("SMA 5",        f"${latest['SMA_5']:.2f}")
        st.metric("SMA 10",       f"${latest['SMA_10']:.2f}")
        st.divider()

        st.subheader("🤖 Explanation")

        if st.button("Generate explanation", key=f"explain_{ticker}"):

            with st.spinner("Generating explanation..."):
                if use_llm:
                    explanation = explain_prediction(
                        ticker,
                        pred,
                        prob if pred == 1 else 1 - prob,
                        latest.to_dict(),
                        model=ollama_model,
                    )
                else:
                    explanation = _rule_based_prediction(
                        ticker,
                        pred,
                        prob if pred == 1 else 1 - prob,
                        latest.to_dict(),
                    )

                # store result so it persists across reruns
                st.session_state[f"explanation_{ticker}"] = explanation

        # show cached explanation if it exists
        key = f"explanation_{ticker}"
        if key in st.session_state:
            st.markdown(st.session_state[key])


# ── Tab 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    bt          = backtest_strategy(df)
    sr          = sharpe_ratio(bt['Strategy_Return'])
    md          = max_drawdown(bt['Cumulative_Strategy'])
    final_strat = bt['Cumulative_Strategy'].iloc[-1]
    final_mkt   = bt['Cumulative_Market'].iloc[-1]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Strategy Return",   f"{(final_strat-1)*100:.1f}%")
    c2.metric("Buy & Hold Return", f"{(final_mkt-1)*100:.1f}%")
    c3.metric("Sharpe Ratio",      str(sr))
    c4.metric("Max Drawdown",      f"{md:.1%}")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=bt['Date'], y=bt['Cumulative_Strategy'], name='ML Strategy', line=dict(color='lime')))
    fig2.add_trace(go.Scatter(x=bt['Date'], y=bt['Cumulative_Market'],   name='Buy & Hold',  line=dict(color='royalblue', dash='dash')))
    fig2.update_layout(title="Cumulative Returns", height=400, yaxis_title="Growth of $1")
    st.plotly_chart(fig2, use_container_width=True)

    # ── LLM or rule-based backtest explanation
    total_ret_pct = round((final_strat - 1) * 100, 2)
    bh_ret_pct    = round((final_mkt   - 1) * 100, 2)

    st.divider()
    st.subheader("🤖 Backtest Explanation")

    bt_key = f"bt_explanation_{ticker}_{bt.index[-1]}"

    if st.button("Generate backtest explanation", key=f"bt_btn_{ticker}"):

        with st.spinner("Generating explanation..."):
            if use_llm:
                explanation = explain_backtest(
                    sharpe=sr,
                    max_dd=md,
                    final_strat=final_strat,
                    final_market=final_mkt,
                    ticker=ticker,
                    total_return=total_ret_pct,
                    bh_return=bh_ret_pct,
                    num_trades=int(bt['Prediction'].sum()),
                    win_rate=0.0,
                    model=ollama_model,
                )
            else:
                explanation = _rule_based_backtest(sr, md, final_strat, final_mkt)

            st.session_state[bt_key] = explanation

    if bt_key in st.session_state:
        st.info(st.session_state[bt_key])


# ── Tab 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    metrics = evaluate_model(df['Target'], df['Prediction'], df['Pred_Prob'])
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Accuracy",  metrics['accuracy'])
    c2.metric("ROC-AUC",   metrics['roc_auc'])
    c3.metric("Precision", metrics['precision'])
    c4.metric("Recall",    metrics['recall'])
    c5.metric("F1",        metrics['f1'])

    cm    = metrics['confusion_matrix']
    cm_df = pd.DataFrame(cm, index=['Actual Down','Actual Up'], columns=['Pred Down','Pred Up'])
    st.plotly_chart(px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues',
                              title="Confusion Matrix"), use_container_width=True)
    st.plotly_chart(px.histogram(df, x='Pred_Prob', color='Target', nbins=40, barmode='overlay',
                                 color_discrete_map={0:'tomato',1:'steelblue'},
                                 labels={'Pred_Prob':'Predicted Probability (Up)','Target':'Actual'}),
                    use_container_width=True)


# ── Tab 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    portfolio_state = load_portfolio()
    held            = list(portfolio_state["positions"].keys())
    cur_prices      = {t: (get_current_price(t) or portfolio_state["positions"][t]["avg_cost"]) for t in held}
    summary         = get_portfolio_summary(portfolio_state, cur_prices)

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Total Value",  f"${summary['total_value']:,.2f}")
    m2.metric("Cash",         f"${summary['cash']:,.2f}")
    m3.metric("Equity",       f"${summary['equity']:,.2f}")
    m4.metric("Total Return", f"${summary['total_return']:,.2f}", delta=f"{summary['total_return_pct']:.2f}%")
    m5.metric("Realized P&L", f"${summary['realized_pnl']:,.2f}")
    st.divider()

    st.subheader("📌 Open Positions")
    if summary["positions"]:
        pos_df = pd.DataFrame(summary["positions"])
        pos_df.columns = ["Ticker","Shares","Avg Cost","Current Price","Market Value","Unrealized P&L","P&L %"]
        for c in ["Avg Cost","Current Price"]:
            pos_df[c] = pos_df[c].apply(lambda x: f"${x:.2f}")
        pos_df["Market Value"]   = pos_df["Market Value"].apply(lambda x: f"${x:,.2f}")
        pos_df["Unrealized P&L"] = pos_df["Unrealized P&L"].apply(lambda x: f"${x:,.2f}")
        pos_df["P&L %"]          = pos_df["P&L %"].apply(lambda x: f"{x:.2f}%")
        st.dataframe(pos_df, use_container_width=True, hide_index=True)
    else:
        st.info("No open positions. Click ⚡ Run Signals in the sidebar.")

    if summary["positions"]:
        st.divider()
        alloc_labels = ["Cash"] + [p["ticker"] for p in summary["positions"]]
        alloc_values = [summary["cash"]] + [p["market_value"] for p in summary["positions"]]
        st.plotly_chart(px.pie(values=alloc_values, names=alloc_labels, title="Portfolio Allocation"),
                        use_container_width=True)
    st.divider()

    st.subheader("📋 Trade History")
    trades = portfolio_state.get("trades", [])
    if trades:
        td = pd.DataFrame(trades)
        td["pnl"]         = td["pnl"].apply(lambda x: f"${x:,.2f}" if x is not None else "—")
        td["value"]       = td["value"].apply(lambda x: f"${x:,.2f}")
        td["price"]       = td["price"].apply(lambda x: f"${x:.2f}")
        td["probability"] = td["probability"].apply(lambda x: f"{x:.0%}")
        td.columns        = ["Date","Ticker","Action","Shares","Price","Value","Confidence","P&L"]
        st.dataframe(td.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("No trades yet. Click ⚡ Run Signals on All Tickers in the sidebar.")


# ── Tab 5: Walk-Forward Test ──────────────────────────────────────────────────
with tab5:
    st.subheader("🧪 Walk-Forward Backtest")
    st.markdown(
        "The model trains **only** on data before the cutoff date and then "
        "simulates real trading decisions on the period it never saw. "
        "This gives an honest picture of out-of-sample performance."
    )

    trained_tickers = get_trained_tickers()
    if not trained_tickers:
        st.warning("No trained tickers found. Download & Retrain at least one ticker first.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        selected_tickers = st.multiselect(
            "Tickers to test", trained_tickers, default=trained_tickers
        )
        cutoff = st.date_input(
            "Train/test cutoff date",
            value=date.today() - timedelta(days=365),
            help="Model trains on data BEFORE this date, trades AFTER it."
        )
        wf_model_name   = st.selectbox("Model", list(MODEL_REGISTRY.keys()), key="wf_model")
        wf_feature_name = st.selectbox("Feature set", list(FEATURE_SETS.keys()), key="wf_features")

    with col2:
        wf_cash       = st.number_input("Starting capital ($)", value=100_000, step=10_000)
        wf_confidence = st.slider("Confidence threshold", 0.5, 0.90, 0.55, 0.01, key="wf_conf")
        wf_pos_size   = st.slider("Position size (%)", 0.05, 1.0, 0.20, 0.01, key="wf_pos")
        st.caption(f"**{wf_model_name}** · {wf_feature_name} features: "
                   f"`{', '.join(FEATURE_SETS[wf_feature_name])}`")

    if st.button("▶️ Run Walk-Forward Backtest", use_container_width=True, type="primary"):
        if not selected_tickers:
            st.error("Select at least one ticker.")
        else:
            with st.spinner("Loading data and running backtest..."):
                raw_data = {}
                for t in selected_tickers:
                    csv_path = f"{RAW_DATA_DIR}/{t}.csv"
                    if os.path.exists(csv_path):
                        raw_data[t] = normalise_df(pd.read_csv(csv_path))

                import copy
                results = run_multi_ticker_walk_forward(
                    tickers=selected_tickers,
                    raw_data=raw_data,
                    cutoff_date=str(cutoff),
                    starting_cash=float(wf_cash),
                    confidence_threshold=wf_confidence,
                    position_size_pct=wf_pos_size,
                    model=copy.deepcopy(MODEL_REGISTRY[wf_model_name]),
                    feature_set=FEATURE_SETS[wf_feature_name],
                )

            if not results:
                st.error("Backtest failed — check that your tickers have enough data before the cutoff date.")
                st.stop()

            st.session_state["wf_results"] = results
            st.session_state["wf_cutoff"]  = cutoff

    if "wf_results" in st.session_state:
        results = st.session_state["wf_results"]
        cutoff  = st.session_state["wf_cutoff"]
        summary = results["summary"]

        st.divider()
        st.subheader("📊 Overall Results")

        profitable = summary['total_return_pct'] > 0
        if profitable:
            st.success("✅ Strategy was **profitable** over the test period!")
        else:
            st.error("❌ Strategy was **not profitable** over the test period.")

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Final Portfolio Value", f"${summary['final_value']:,.2f}",
                  delta=f"{summary['total_return_pct']:.2f}%")
        m2.metric("Buy & Hold Return",     f"{summary['bh_return_pct']:.2f}%")
        m3.metric("Alpha vs Buy & Hold",   f"{summary['alpha']:+.2f}%")
        m4.metric("Total Trades",          summary['num_trades'])

        st.subheader("🤖 Walk-Forward Explanation")

        if st.button("Generate walk-forward explanation", key="wf_explain_btn"):

            with st.spinner("🤖 Generating strategy summary..."):

                first_ticker = list(results["per_ticker"].keys())[0]
                wf_m = results["per_ticker"][first_ticker]["metrics"]

                if use_llm:
                    wf_explanation = explain_backtest(
                        sharpe=wf_m["sharpe_ratio"],
                        max_dd=wf_m["max_drawdown"],
                        final_strat=wf_m["final_value"] / wf_m["starting_cash"],
                        final_market=wf_m["bh_final_value"] / wf_m["starting_cash"],
                        ticker=", ".join(summary["tickers"]),
                        total_return=summary["total_return_pct"],
                        bh_return=summary["bh_return_pct"],
                        num_trades=summary["num_trades"],
                        win_rate=wf_m.get("win_rate_pct", 0),
                        model=ollama_model,
                    )
                else:
                    wf_explanation = _rule_based_backtest(
                        wf_m["sharpe_ratio"],
                        wf_m["max_drawdown"],
                        wf_m["final_value"] / wf_m["starting_cash"],
                        wf_m["bh_final_value"] / wf_m["starting_cash"],
                    )

                st.session_state["wf_explanation"] = wf_explanation

        # persist output
        if "wf_explanation" in st.session_state:
            st.info(st.session_state["wf_explanation"])

        # ── Combined equity curve
        combined  = results["combined_equity"]
        cutoff_ts = pd.to_datetime(cutoff)
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=combined['Date'], y=combined['Total_Strategy'],
                                    name='ML Strategy', line=dict(color='lime', width=2)))
        fig_eq.add_trace(go.Scatter(x=combined['Date'], y=combined['Total_BuyHold'],
                                    name='Buy & Hold',  line=dict(color='royalblue', dash='dash', width=2)))
        fig_eq.add_shape(type="line", x0=cutoff_ts, x1=cutoff_ts, y0=0, y1=1,
                         xref="x", yref="paper", line=dict(color="orange", dash="dot"))
        fig_eq.add_annotation(x=cutoff_ts, y=1, xref="x", yref="paper",
                              text="Train/Test Split", showarrow=False,
                              xanchor="left", yanchor="bottom")
        fig_eq.update_layout(title="Combined Portfolio Value — Out-of-Sample Period",
                             height=420, yaxis_title="Portfolio Value ($)")
        st.plotly_chart(fig_eq, use_container_width=True)

        # ── Per-ticker breakdown
        st.divider()
        st.subheader("📈 Per-Ticker Results")
        ticker_rows = []
        for t, r in results["per_ticker"].items():
            m = r["metrics"]
            ticker_rows.append({
                "Ticker":        t,
                "Return %":      f"{m['total_return_pct']:+.2f}%",
                "B&H Return %":  f"{m['bh_return_pct']:+.2f}%",
                "Alpha":         f"{m['alpha']:+.2f}%",
                "Sharpe":        m['sharpe_ratio'],
                "Max Drawdown":  f"{m['max_drawdown']:.1%}",
                "# Trades":      m['num_trades'],
                "Win Rate":      f"{m['win_rate_pct']:.0f}%",
                "Realized P&L":  f"${m['realized_pnl']:,.2f}",
                "Train ROC-AUC": m['train_roc_auc'],
                "Test Days":     m['test_days'],
            })
        st.dataframe(pd.DataFrame(ticker_rows), use_container_width=True, hide_index=True)

        # ── Per-ticker equity curves
        fig_tickers = go.Figure()
        colors = px.colors.qualitative.Plotly
        for i, (t, r) in enumerate(results["per_ticker"].items()):
            eq = r["equity_curve"]
            fig_tickers.add_trace(go.Scatter(
                x=eq['Date'], y=eq['Portfolio_Value'],
                name=f"{t} Strategy", line=dict(color=colors[i % len(colors)])
            ))
            fig_tickers.add_trace(go.Scatter(
                x=eq['Date'], y=eq['BuyHold_Value'],
                name=f"{t} B&H", line=dict(color=colors[i % len(colors)], dash='dot'),
                opacity=0.5
            ))
        fig_tickers.update_layout(title="Individual Ticker Equity Curves",
                                  height=420, yaxis_title="Value ($)")
        st.plotly_chart(fig_tickers, use_container_width=True)

        # ── Trade log
        st.divider()
        st.subheader("📋 All Trades During Test Period")
        all_trades = results["all_trades"]
        if all_trades:
            tdf = pd.DataFrame(all_trades)
            tdf["pnl"]        = tdf["pnl"].apply(lambda x: f"${x:,.2f}" if x is not None else "—")
            tdf["value"]      = tdf["value"].apply(lambda x: f"${x:,.2f}")
            tdf["price"]      = tdf["price"].apply(lambda x: f"${x:.2f}")
            tdf["confidence"] = tdf["confidence"].apply(
                lambda x: f"{x:.0%}" if x is not None else "—"
            )
            tdf.columns = ["Date","Ticker","Action","Shares","Price","Value","Confidence","P&L"]
            st.dataframe(tdf, use_container_width=True, hide_index=True)
        else:
            st.info("No trades were executed. Try lowering the confidence threshold.")


# ── Tab 6: Experiments ────────────────────────────────────────────────────────
with tab6:
    st.subheader("🔬 Model & Feature Experiments")
    st.markdown(
        "Runs every combination of model and feature set through the same "
        "walk-forward engine so results are directly comparable."
    )

    trained_tickers = get_trained_tickers()
    if not trained_tickers:
        st.warning("No trained tickers found.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        exp_ticker = st.selectbox("Ticker", trained_tickers, key="exp_ticker")
        exp_cutoff = st.date_input(
            "Train/test cutoff",
            value=date.today() - timedelta(days=365),
            key="exp_cutoff"
        )
    with col2:
        exp_cash      = st.number_input("Starting capital ($)", value=100_000, key="exp_cash")
        exp_threshold = st.slider("Confidence threshold", 0.50, 0.95, 0.55, 0.01, key="exp_thresh")
        exp_pos_size  = st.slider("Position size (%)", 0.10, 1.0, 0.95, 0.01, key="exp_pos")

    if st.button("▶️ Run All Experiments", use_container_width=True, type="primary"):
        csv_path = f"{RAW_DATA_DIR}/{exp_ticker}.csv"
        if not os.path.exists(csv_path):
            st.error(f"No data found for {exp_ticker}.")
            st.stop()

        df_raw = normalise_df(pd.read_csv(csv_path))

        with st.spinner(f"Running {len(MODEL_REGISTRY) * len(FEATURE_SETS)} combinations..."):
            results_df = run_experiments(
                df_raw=df_raw,
                ticker=ticker,
                cutoff_date=str(exp_cutoff),
                starting_cash=float(exp_cash),
                confidence_threshold=exp_threshold,
                position_size_pct=exp_pos_size,
            )

        st.session_state["exp_results"] = results_df

    if "exp_results" in st.session_state:
        results_df = st.session_state["exp_results"]

        if not results_df.empty and "Return %" in results_df.columns:
            best = results_df.loc[results_df["Return %"].idxmax()]
        else:
            st.error(f"No valid results returned for {ticker}. Check console for details.")
            best = pd.Series()  # or handle as needed

        st.success(
            f"🏆 Best: **{best['Model']}** with **{best['Features']}** features — "
            f"{best['Return %']:+.2f}% return, {best['Alpha']:+.2f}% alpha"
        )

        # ── LLM experiment summary
        st.divider()
        st.subheader("🤖 Experiment Explanation")

        exp_key = f"exp_explanation_{exp_ticker}_{best['Model']}_{best['Features']}"

        if st.button("Generate experiment summary", key="exp_btn"):

            with st.spinner("Generating summary..."):
                if use_llm:
                    explanation = explain_backtest(
                        sharpe=best["Sharpe"],
                        max_dd=best["Max Drawdown"],
                        final_strat=1 + best["Return %"] / 100,
                        final_market=1 + best["B&H Return %"] / 100,
                        ticker=exp_ticker,
                        total_return=best["Return %"],
                        bh_return=best["B&H Return %"],
                        num_trades=int(best["# Trades"]),
                        win_rate=best["Win Rate %"],
                        model=ollama_model,
                    )
                else:
                    explanation = "Rule-based mode: LLM disabled."

                st.session_state[exp_key] = explanation

        if exp_key in st.session_state:
            st.info(st.session_state[exp_key])

        st.subheader("📊 All Results")
        display_df = results_df.copy()
        display_df["Return %"]     = display_df["Return %"].apply(lambda x: f"{x:+.2f}%")
        display_df["B&H Return %"] = display_df["B&H Return %"].apply(lambda x: f"{x:+.2f}%")
        display_df["Alpha"]        = display_df["Alpha"].apply(lambda x: f"{x:+.2f}%")
        display_df["Max Drawdown"] = display_df["Max Drawdown"].apply(lambda x: f"{x:.1%}")
        display_df["Win Rate %"]   = display_df["Win Rate %"].apply(lambda x: f"{x:.0f}%")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        fig = px.bar(results_df, x="Model", y="Return %", color="Features",
                     barmode="group", title="Return % by Model and Feature Set")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(results_df, x="Model", y="Sharpe", color="Features",
                      barmode="group", title="Sharpe Ratio by Model and Feature Set")
        st.plotly_chart(fig2, use_container_width=True)