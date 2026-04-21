"""
src/models/walk_forward.py

Walk-forward backtesting engine.

Workflow:
  1. Load full price history for a ticker
  2. Train on data BEFORE the cutoff date (model never sees test period)
  3. Simulate daily trading decisions on the test period
  4. Return full trade log + equity curve + performance metrics
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from src.features.technical_indicators import add_technical_indicators
from src.models.evaluate import sharpe_ratio, max_drawdown

FEATURES = ['SMA_5', 'SMA_10', 'Close_lag_1', 'Close_lag_2', 'Return']


def run_walk_forward(
    df_raw: pd.DataFrame,
    ticker: str,
    cutoff_date: str,
    starting_cash: float = 100_000.0,
    confidence_threshold: float = 0.55,
    position_size_pct: float = 0.10,
    model=None,               # NEW: pass any sklearn classifier, defaults to LogisticRegression
    feature_set: list = None, # NEW: pass a list of feature column names, defaults to FEATURES
) -> dict:
    """
    Train on data before cutoff_date, simulate trading after it.
    The model never sees the test period data during training.
    """
    features = feature_set or FEATURES  # use provided feature set or fall back to default

    df = add_technical_indicators(df_raw.copy())
    cutoff = pd.Timestamp(cutoff_date)

    train_df = df[df['Date'] < cutoff].copy()
    test_df  = df[df['Date'] >= cutoff].copy()

    if len(train_df) < 50:
        raise ValueError(f"Not enough training data before {cutoff_date}.")
    if len(test_df) < 5:
        raise ValueError(f"Not enough test data after {cutoff_date}.")

    # Train model on training period ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[features])
    y_train = train_df['Target']

    if model is None:
        model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train_scaled)[:, 1])

    # Predict on test period (unseen data)
    X_test_scaled = scaler.transform(test_df[features])
    test_df = test_df.copy()
    test_df['Prediction'] = model.predict(X_test_scaled)
    test_df['Pred_Prob']  = model.predict_proba(X_test_scaled)[:, 1]

    # Simulate daily trading
    cash        = starting_cash
    shares_held = 0
    avg_cost    = 0.0
    trades      = []
    equity_curve = []

    for _, row in test_df.iterrows():
        date      = row['Date']
        price     = float(row['Close'])
        pred      = int(row['Prediction'])
        prob      = float(row['Pred_Prob'])
        port_val  = cash + shares_held * price
        action    = "HOLD"

        # BUY: model says UP, confident enough, not already holding
        if pred == 1 and prob >= confidence_threshold and shares_held == 0:
            spend = starting_cash * position_size_pct
            shares = int(spend // price)
            if shares > 0 and cash >= shares * price:
                cash       -= shares * price
                shares_held = shares
                avg_cost    = price
                action      = "BUY"
                trades.append({
                    "date": date.strftime("%Y-%m-%d"), "ticker": ticker,
                    "action": "BUY", "shares": shares, "price": round(price, 4),
                    "value": round(shares * price, 2), "confidence": round(prob, 4),
                    "pnl": None,
                })

        # SELL: model says DOWN, confident enough, currently holding
        elif pred == 0 and (1 - prob) >= confidence_threshold and shares_held > 0:
            proceeds = shares_held * price
            pnl      = (price - avg_cost) * shares_held
            cash    += proceeds
            action   = "SELL"
            trades.append({
                "date": date.strftime("%Y-%m-%d"), "ticker": ticker,
                "action": "SELL", "shares": shares_held, "price": round(price, 4),
                "value": round(proceeds, 2), "confidence": round(1 - prob, 4),
                "pnl": round(pnl, 2),
            })
            shares_held = 0
            avg_cost    = 0.0

        equity_curve.append({
            "Date": date,
            "Portfolio_Value": round(cash + shares_held * price, 2),
            "Cash": round(cash, 2),
            "Shares_Held": shares_held,
            "Close": price,
            "Action": action,
        })

    # Force-close any open position at end of test period
    if shares_held > 0:
        last_price = float(test_df.iloc[-1]['Close'])
        pnl        = (last_price - avg_cost) * shares_held
        cash      += shares_held * last_price
        trades.append({
            "date": test_df.iloc[-1]['Date'].strftime("%Y-%m-%d"), "ticker": ticker,
            "action": "SELL (end)", "shares": shares_held,
            "price": round(last_price, 4), "value": round(shares_held * last_price, 2),
            "confidence": None, "pnl": round(pnl, 2),
        })

    eq_df = pd.DataFrame(equity_curve)

    # Buy-and-hold benchmark over same period
    bh_start             = float(test_df.iloc[0]['Close'])
    bh_shares = (starting_cash * position_size_pct) // bh_start
    bh_leftover          = starting_cash - bh_shares * bh_start
    eq_df['BuyHold_Value'] = bh_shares * eq_df['Close'] + bh_leftover

    # Performance metrics
    final_value  = eq_df['Portfolio_Value'].iloc[-1]
    final_bh     = eq_df['BuyHold_Value'].iloc[-1]
    total_return = (final_value - starting_cash) / starting_cash
    bh_return    = (final_bh   - starting_cash) / starting_cash

    eq_df['Daily_Return']    = eq_df['Portfolio_Value'].pct_change().fillna(0)
    eq_df['BH_Daily_Return'] = eq_df['BuyHold_Value'].pct_change().fillna(0)

    sell_trades    = [t for t in trades if 'SELL' in t['action']]
    winning_trades = [t for t in sell_trades if t['pnl'] and t['pnl'] > 0]
    win_rate       = len(winning_trades) / len(sell_trades) if sell_trades else 0
    realized_pnl   = sum(t['pnl'] for t in trades if t['pnl'] is not None)

    metrics = {
        "ticker":               ticker,
        "cutoff_date":          cutoff_date,
        "train_days":           len(train_df),
        "test_days":            len(test_df),
        "starting_cash":        starting_cash,
        "final_value":          round(final_value, 2),
        "total_return_pct":     round(total_return * 100, 2),
        "bh_final_value":       round(final_bh, 2),
        "bh_return_pct":        round(bh_return * 100, 2),
        "alpha":                round((total_return - bh_return) * 100, 2),
        "sharpe_ratio":         sharpe_ratio(eq_df['Daily_Return']),
        "bh_sharpe":            sharpe_ratio(eq_df['BH_Daily_Return']),
        "max_drawdown":         max_drawdown(eq_df['Portfolio_Value'] / starting_cash),
        "bh_max_drawdown":      max_drawdown(eq_df['BuyHold_Value']   / starting_cash),
        "num_trades":           len(trades),
        "num_sells":            len(sell_trades),
        "win_rate_pct":         round(win_rate * 100, 2),
        "realized_pnl":         round(realized_pnl, 2),
        "winning_trades":       len(winning_trades),
        "losing_trades":        len(sell_trades) - len(winning_trades),
        "train_roc_auc":        round(train_auc, 4),
        "confidence_threshold": confidence_threshold,
        "model_name":           type(model).__name__,  # NEW: track which model was used
        "feature_set":          features,              # NEW: track which features were used
    }

    return {
        "trades":       trades,
        "equity_curve": eq_df,
        "metrics":      metrics,
        "test_df":      test_df,
    }


def run_multi_ticker_walk_forward(
    tickers: list,
    raw_data: dict,
    cutoff_date: str,
    starting_cash: float = 100_000.0,
    confidence_threshold: float = 0.55,
    position_size_pct: float = 0.10,
    model=None,
    feature_set: list = None,
) -> dict:
    """
    Run walk-forward backtest across multiple tickers.
    Capital is split evenly across tickers that successfully complete —
    skipped tickers do not silently reduce the starting portfolio value.
    """
    import copy

    # ── Step 1: run every ticker at full capital so return % and trade
    #    logic are unaffected, collect only the ones that succeed ────────
    results    = {}
    all_trades = []

    per_ticker_cash = starting_cash / len(tickers)

    for ticker in tickers:
        if ticker not in raw_data:
            continue
        try:
            r = run_walk_forward(
                df_raw=raw_data[ticker],
                ticker=ticker,
                cutoff_date=cutoff_date,
                starting_cash=per_ticker_cash,
                confidence_threshold=confidence_threshold,
                position_size_pct=position_size_pct,
                model=copy.deepcopy(model),
                feature_set=feature_set,
            )
            results[ticker] = r
        except Exception as e:
            print(f"[{ticker}] Skipped: {e}")

    if not results:
        return {}
    # ── Step 3: combine equity curves across all successful tickers ──────
    equity_dfs = []
    for ticker, r in results.items():
        eq = r["equity_curve"][["Date", "Portfolio_Value", "BuyHold_Value"]].copy()
        eq = eq.rename(columns={
            "Portfolio_Value": f"{ticker}_Value",
            "BuyHold_Value":   f"{ticker}_BH",
        })
        equity_dfs.append(eq.set_index("Date"))

    combined = pd.concat(equity_dfs, axis=1)
    combined = combined.ffill().dropna().reset_index()

    strat_cols = [c for c in combined.columns if c.endswith("_Value")]
    bh_cols    = [c for c in combined.columns if c.endswith("_BH")]
    combined["Total_Strategy"] = combined[strat_cols].sum(axis=1)
    combined["Total_BuyHold"]  = combined[bh_cols].sum(axis=1)

    total_final  = combined["Total_Strategy"].iloc[-1]
    total_bh     = combined["Total_BuyHold"].iloc[-1]
    total_return = (total_final - starting_cash) / starting_cash * 100
    bh_return    = (total_bh   - starting_cash) / starting_cash * 100

    return {
        "per_ticker":      results,
        "combined_equity": combined,
        "all_trades":      sorted(all_trades, key=lambda x: x["date"]),
        "summary": {
            "tickers":          list(results.keys()),
            "cutoff_date":      cutoff_date,
            "starting_cash":    starting_cash,
            "final_value":      round(total_final, 2),
            "total_return_pct": round(total_return, 2),
            "bh_return_pct":    round(bh_return, 2),
            "alpha":            round(total_return - bh_return, 2),
            "num_trades":       len(all_trades),
        },
    }