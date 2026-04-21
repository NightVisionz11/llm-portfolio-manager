"""
tests/test_walk_forward.py

Tests for src/models/walk_forward.py
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

# ── Inline copies of the functions under test ────────────────────────────────
# Replace these with real imports once the project is wired up:
#   from src.models.walk_forward import run_walk_forward, run_multi_ticker_walk_forward

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def _sharpe_ratio(returns, risk_free_rate=0.0):
    excess = returns - risk_free_rate / 252
    if excess.std() == 0:
        return 0.0
    return round((excess.mean() / excess.std()) * np.sqrt(252), 4)

def _max_drawdown(cum):
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return round(dd.min(), 4)

FEATURES = ['SMA_5', 'SMA_10', 'Close_lag_1', 'Close_lag_2', 'Return']


def _add_indicators(df):
    """Minimal stand-in for add_technical_indicators."""
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df['Return']     = df['Close'].pct_change().fillna(0)
    df['SMA_5']      = df['Close'].rolling(5).mean().bfill()
    df['SMA_10']     = df['Close'].rolling(10).mean().bfill()
    df['Close_lag_1'] = df['Close'].shift(1).bfill()
    df['Close_lag_2'] = df['Close'].shift(2).bfill()
    df['RSI_14']     = 50.0                    # stub
    df['MACD']       = 0.0
    df['Volume_change'] = 0.0
    df['Volatility_10'] = df['Return'].rolling(10).std().bfill()
    df['Target']     = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna(subset=['Target']).reset_index(drop=True)
    return df


def run_walk_forward(
    df_raw, ticker, cutoff_date,
    starting_cash=100_000.0, bh_allocation_pct=1.0,
    confidence_threshold=0.55, position_size_pct=0.10,
    model=None, feature_set=None,
):
    features = feature_set or FEATURES
    df = _add_indicators(df_raw.copy())
    cutoff = pd.Timestamp(cutoff_date)

    train_df = df[df['Date'] < cutoff].copy()
    test_df  = df[df['Date'] >= cutoff].copy()

    if len(train_df) < 50:
        raise ValueError(f"Not enough training data before {cutoff_date}.")
    if len(test_df) < 5:
        raise ValueError(f"Not enough test data after {cutoff_date}.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    y_train = train_df['Target']

    if model is None:
        model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])

    X_test = scaler.transform(test_df[features])
    test_df = test_df.copy()
    test_df['Prediction'] = model.predict(X_test)
    test_df['Pred_Prob']  = model.predict_proba(X_test)[:, 1]

    cash = starting_cash; shares_held = 0; avg_cost = 0.0
    trades = []; equity_curve = []

    for _, row in test_df.iterrows():
        date = row['Date']; price = float(row['Close'])
        pred = int(row['Prediction']); prob = float(row['Pred_Prob'])
        port_val = cash + shares_held * price; action = "HOLD"

        if pred == 1 and prob >= confidence_threshold and shares_held == 0:
            spend = port_val * position_size_pct
            shares = int(spend // price)
            if shares > 0 and cash >= shares * price:
                cash -= shares * price; shares_held = shares; avg_cost = price; action = "BUY"
                trades.append({"date": date.strftime("%Y-%m-%d"), "ticker": ticker,
                                "action": "BUY", "shares": shares, "price": round(price, 4),
                                "value": round(shares * price, 2), "confidence": round(prob, 4), "pnl": None})

        elif pred == 0 and (1 - prob) >= confidence_threshold and shares_held > 0:
            proceeds = shares_held * price; pnl = (price - avg_cost) * shares_held
            cash += proceeds; action = "SELL"
            trades.append({"date": date.strftime("%Y-%m-%d"), "ticker": ticker,
                            "action": "SELL", "shares": shares_held, "price": round(price, 4),
                            "value": round(proceeds, 2), "confidence": round(1 - prob, 4), "pnl": round(pnl, 2)})
            shares_held = 0; avg_cost = 0.0

        equity_curve.append({"Date": date, "Portfolio_Value": round(cash + shares_held * price, 2),
                              "Cash": round(cash, 2), "Shares_Held": shares_held,
                              "Close": price, "Action": action})

    if shares_held > 0:
        last_price = float(test_df.iloc[-1]['Close'])
        pnl = (last_price - avg_cost) * shares_held; cash += shares_held * last_price
        trades.append({"date": test_df.iloc[-1]['Date'].strftime("%Y-%m-%d"), "ticker": ticker,
                        "action": "SELL (end)", "shares": shares_held, "price": round(last_price, 4),
                        "value": round(shares_held * last_price, 2), "confidence": None, "pnl": round(pnl, 2)})

    eq_df = pd.DataFrame(equity_curve)
    bh_start = float(test_df.iloc[0]['Close']); bh_capital = starting_cash * bh_allocation_pct
    bh_shares = int(bh_capital // bh_start); bh_leftover = bh_capital - bh_shares * bh_start
    idle_cash = starting_cash - bh_capital
    eq_df['BuyHold_Value'] = bh_shares * eq_df['Close'] + bh_leftover + idle_cash

    final_value = eq_df['Portfolio_Value'].iloc[-1]; final_bh = eq_df['BuyHold_Value'].iloc[-1]
    eq_df['Daily_Return']    = eq_df['Portfolio_Value'].pct_change().fillna(0)
    eq_df['BH_Daily_Return'] = eq_df['BuyHold_Value'].pct_change().fillna(0)

    sell_trades    = [t for t in trades if 'SELL' in t['action']]
    winning_trades = [t for t in sell_trades if t['pnl'] and t['pnl'] > 0]
    win_rate       = len(winning_trades) / len(sell_trades) if sell_trades else 0

    metrics = {
        "ticker": ticker, "cutoff_date": cutoff_date,
        "train_days": len(train_df), "test_days": len(test_df),
        "starting_cash": starting_cash, "final_value": round(final_value, 2),
        "total_return_pct": round((final_value - starting_cash) / starting_cash * 100, 2),
        "bh_final_value": round(final_bh, 2),
        "bh_return_pct": round((final_bh - starting_cash) / starting_cash * 100, 2),
        "alpha": round(((final_value - starting_cash) / starting_cash - (final_bh - starting_cash) / starting_cash) * 100, 2),
        "sharpe_ratio": _sharpe_ratio(eq_df['Daily_Return']),
        "bh_sharpe": _sharpe_ratio(eq_df['BH_Daily_Return']),
        "max_drawdown": _max_drawdown(eq_df['Portfolio_Value'] / starting_cash),
        "bh_max_drawdown": _max_drawdown(eq_df['BuyHold_Value'] / starting_cash),
        "num_trades": len(trades), "num_sells": len(sell_trades),
        "win_rate_pct": round(win_rate * 100, 2),
        "realized_pnl": round(sum(t['pnl'] for t in trades if t['pnl'] is not None), 2),
        "winning_trades": len(winning_trades),
        "losing_trades": len(sell_trades) - len(winning_trades),
        "train_roc_auc": round(train_auc, 4),
        "confidence_threshold": confidence_threshold,
        "model_name": type(model).__name__,
        "feature_set": features,
    }
    return {"trades": trades, "equity_curve": eq_df, "metrics": metrics, "test_df": test_df}


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_price_df(n=300, start="2020-01-01", seed=42):
    """Synthetic daily OHLCV-ish price series long enough to train on."""
    rng    = np.random.default_rng(seed)
    dates  = pd.bdate_range(start, periods=n)
    closes = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    return pd.DataFrame({"Date": dates, "Close": closes,
                          "Volume": rng.integers(1_000_000, 5_000_000, n)})


@pytest.fixture
def price_df():
    return _make_price_df(n=300)


@pytest.fixture
def wf_result(price_df):
    """Run once, reuse across tests."""
    return run_walk_forward(
        df_raw=price_df, ticker="TEST",
        cutoff_date="2021-01-01",
        starting_cash=100_000.0,
        confidence_threshold=0.55,
        position_size_pct=0.10,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Return-value structure
# ─────────────────────────────────────────────────────────────────────────────

class TestWalkForwardStructure:

    def test_result_has_required_keys(self, wf_result):
        assert set(wf_result.keys()) == {"trades", "equity_curve", "metrics", "test_df"}

    def test_equity_curve_columns(self, wf_result):
        eq = wf_result["equity_curve"]
        for col in ("Date", "Portfolio_Value", "Cash", "Shares_Held", "Close", "Action", "BuyHold_Value"):
            assert col in eq.columns

    def test_metrics_has_required_keys(self, wf_result):
        required = {
            "ticker", "cutoff_date", "train_days", "test_days",
            "starting_cash", "final_value", "total_return_pct",
            "bh_return_pct", "alpha", "sharpe_ratio", "max_drawdown",
            "num_trades", "win_rate_pct", "train_roc_auc",
        }
        assert required.issubset(wf_result["metrics"].keys())

    def test_equity_curve_length_equals_test_days(self, wf_result):
        m  = wf_result["metrics"]
        eq = wf_result["equity_curve"]
        assert len(eq) == m["test_days"]


# ─────────────────────────────────────────────────────────────────────────────
# Data-leakage guard
# ─────────────────────────────────────────────────────────────────────────────

class TestNoDataLeakage:

    def test_test_dates_all_after_cutoff(self, wf_result):
        cutoff = pd.Timestamp("2021-01-01")
        assert (wf_result["test_df"]["Date"] >= cutoff).all()

    def test_equity_curve_dates_all_after_cutoff(self, wf_result):
        cutoff = pd.Timestamp("2021-01-01")
        assert (wf_result["equity_curve"]["Date"] >= cutoff).all()


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio-value invariants
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolioInvariants:

    def test_portfolio_value_always_positive(self, wf_result):
        assert (wf_result["equity_curve"]["Portfolio_Value"] > 0).all()

    def test_cash_never_negative(self, wf_result):
        assert (wf_result["equity_curve"]["Cash"] >= 0).all()

    def test_shares_held_never_negative(self, wf_result):
        assert (wf_result["equity_curve"]["Shares_Held"] >= 0).all()

    def test_starting_cash_matches_metric(self, wf_result):
        assert wf_result["metrics"]["starting_cash"] == 100_000.0

    def test_final_value_consistent_with_equity_curve(self, wf_result):
        eq_final = wf_result["equity_curve"]["Portfolio_Value"].iloc[-1]
        assert wf_result["metrics"]["final_value"] == eq_final

    def test_total_return_consistent_with_values(self, wf_result):
        m = wf_result["metrics"]
        expected_pct = round((m["final_value"] - m["starting_cash"]) / m["starting_cash"] * 100, 2)
        assert m["total_return_pct"] == expected_pct


# ─────────────────────────────────────────────────────────────────────────────
# Trade-log integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeLogIntegrity:

    def test_no_buy_after_buy_without_sell(self, wf_result):
        """The strategy should never hold more than one open position at a time."""
        in_position = False
        for trade in wf_result["trades"]:
            if trade["action"] == "BUY":
                assert not in_position, "BUY issued while already holding a position"
                in_position = True
            elif "SELL" in trade["action"]:
                in_position = False

    def test_sell_pnl_matches_price_diff(self, wf_result):
        """PnL on a SELL should equal (sell_price - buy_price) * shares."""
        trades = wf_result["trades"]
        buy_price = None; buy_shares = None
        for t in trades:
            if t["action"] == "BUY":
                buy_price = t["price"]; buy_shares = t["shares"]
            elif "SELL" in t["action"] and buy_price is not None:
                expected_pnl = round((t["price"] - buy_price) * buy_shares, 2)
                assert abs(t["pnl"] - expected_pnl) < 0.05, (
                    f"PnL mismatch: got {t['pnl']}, expected {expected_pnl}"
                )
                buy_price = None

    def test_trade_dates_within_test_period(self, wf_result):
        cutoff = pd.Timestamp("2021-01-01")
        last_test_date = wf_result["test_df"]["Date"].max()
        for t in wf_result["trades"]:
            trade_date = pd.Timestamp(t["date"])
            assert trade_date >= cutoff
            assert trade_date <= last_test_date

    def test_num_trades_in_metrics_matches_trade_log(self, wf_result):
        assert wf_result["metrics"]["num_trades"] == len(wf_result["trades"])

    def test_buy_trade_has_null_pnl(self, wf_result):
        for t in wf_result["trades"]:
            if t["action"] == "BUY":
                assert t["pnl"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Metrics sanity
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsSanity:

    def test_train_auc_between_0_and_1(self, wf_result):
        assert 0.0 <= wf_result["metrics"]["train_roc_auc"] <= 1.0

    def test_max_drawdown_non_positive(self, wf_result):
        assert wf_result["metrics"]["max_drawdown"] <= 0

    def test_win_rate_between_0_and_100(self, wf_result):
        assert 0 <= wf_result["metrics"]["win_rate_pct"] <= 100

    def test_alpha_equals_return_minus_bh(self, wf_result):
        m = wf_result["metrics"]
        expected = round(m["total_return_pct"] - m["bh_return_pct"], 2)
        assert m["alpha"] == expected


# ─────────────────────────────────────────────────────────────────────────────
# Pluggable model / feature-set support
# ─────────────────────────────────────────────────────────────────────────────

class TestPluggableModels:

    @pytest.mark.parametrize("model_cls", [
        LogisticRegression,
        DummyClassifier,
        RandomForestClassifier,
    ])
    def test_different_models_run_without_error(self, price_df, model_cls):
        result = run_walk_forward(
            df_raw=price_df, ticker="TEST",
            cutoff_date="2021-01-01",
            model=model_cls(),
        )
        assert "metrics" in result

    def test_model_name_tracked_in_metrics(self, price_df):
        result = run_walk_forward(
            df_raw=price_df, ticker="TEST",
            cutoff_date="2021-01-01",
            model=RandomForestClassifier(n_estimators=10),
        )
        assert result["metrics"]["model_name"] == "RandomForestClassifier"

    def test_custom_feature_set_tracked_in_metrics(self, price_df):
        custom_features = ['SMA_5', 'SMA_10', 'Return']
        result = run_walk_forward(
            df_raw=price_df, ticker="TEST",
            cutoff_date="2021-01-01",
            feature_set=custom_features,
        )
        assert result["metrics"]["feature_set"] == custom_features

    def test_default_feature_set_used_when_none_passed(self, price_df):
        result = run_walk_forward(
            df_raw=price_df, ticker="TEST",
            cutoff_date="2021-01-01",
        )
        assert result["metrics"]["feature_set"] == FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases / error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_raises_on_insufficient_train_data(self):
        tiny_df = _make_price_df(n=60)
        with pytest.raises(ValueError, match="Not enough training data"):
            run_walk_forward(
                df_raw=tiny_df, ticker="TINY",
                cutoff_date="2020-01-10",  # only ~7 business days of training data
            )

    def test_raises_on_insufficient_test_data(self):
        df = _make_price_df(n=300)
        # Set cutoff to near the very end so <5 test rows remain
        last_date = df["Date"].max().strftime("%Y-%m-%d")
        with pytest.raises(ValueError, match="Not enough test data"):
            run_walk_forward(df_raw=df, ticker="T", cutoff_date=last_date)

    def test_confidence_threshold_1_produces_no_trades(self, price_df):
        """A threshold of 1.0 means the model is never confident enough to trade."""
        result = run_walk_forward(
            df_raw=price_df, ticker="TEST",
            cutoff_date="2021-01-01",
            confidence_threshold=1.0,
        )
        buy_trades = [t for t in result["trades"] if t["action"] == "BUY"]
        assert len(buy_trades) == 0

    def test_position_size_respected(self, price_df):
        """Each BUY should spend at most position_size_pct of portfolio value."""
        pct = 0.10
        result = run_walk_forward(
            df_raw=price_df, ticker="TEST",
            cutoff_date="2021-01-01",
            starting_cash=100_000.0,
            position_size_pct=pct,
            confidence_threshold=0.50,
        )
        for t in result["trades"]:
            if t["action"] == "BUY":
                # value should be ≤ position_size_pct × starting_cash (plus rounding)
                assert t["value"] <= 100_000.0 * pct + t["price"]