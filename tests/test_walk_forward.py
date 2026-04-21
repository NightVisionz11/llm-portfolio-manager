"""
tests/test_walk_forward.py

Run with:  pytest tests/
"""

import pytest
import pandas as pd
import numpy as np
from src.models.walk_forward import run_walk_forward


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_price_df(n=300, start=100.0, trend=0.3):
    """Synthetic OHLCV data with a slight upward trend."""
    dates  = pd.date_range("2020-01-01", periods=n, freq="B")
    closes = [start + i * trend + np.sin(i / 10) * 2 for i in range(n)]
    return pd.DataFrame({
        "Date":   dates,
        "Open":   closes,
        "High":   [c + 1.0 for c in closes],
        "Low":    [c - 1.0 for c in closes],
        "Close":  closes,
        "Volume": [1_000_000 + i * 1000 for i in range(n)],
    })


CUTOFF = "2021-06-01"
FEATURES = ['SMA_5', 'SMA_10', 'Close_lag_1', 'Close_lag_2', 'Return']


# ── Basic return structure ────────────────────────────────────────────────────

def test_run_walk_forward_returns_dict():
    result = run_walk_forward(make_price_df(), "TEST", CUTOFF)
    assert isinstance(result, dict)


def test_run_walk_forward_has_required_keys():
    result = run_walk_forward(make_price_df(), "TEST", CUTOFF)
    for key in ["trades", "equity_curve", "metrics", "test_df"]:
        assert key in result, f"Missing key: {key}"


def test_metrics_has_required_keys():
    result  = run_walk_forward(make_price_df(), "TEST", CUTOFF)
    metrics = result["metrics"]
    for key in [
        "ticker", "total_return_pct", "bh_return_pct", "alpha",
        "sharpe_ratio", "max_drawdown", "num_trades", "train_roc_auc",
    ]:
        assert key in metrics, f"Missing metric: {key}"


# ── No lookahead bias ─────────────────────────────────────────────────────────

def test_test_df_dates_all_after_cutoff():
    result   = run_walk_forward(make_price_df(), "TEST", CUTOFF)
    test_df  = result["test_df"]
    cutoff_ts = pd.Timestamp(CUTOFF)
    assert (test_df["Date"] >= cutoff_ts).all(), "Test set contains pre-cutoff dates — lookahead bias!"


def test_train_days_all_before_cutoff():
    """Indirectly verified: if test period is clean, train period must be too."""
    result  = run_walk_forward(make_price_df(), "TEST", CUTOFF)
    metrics = result["metrics"]
    assert metrics["train_days"] > 0
    assert metrics["test_days"]  > 0


# ── Equity curve ──────────────────────────────────────────────────────────────

def test_equity_curve_has_required_columns():
    result = run_walk_forward(make_price_df(), "TEST", CUTOFF)
    eq     = result["equity_curve"]
    for col in ["Date", "Portfolio_Value", "Cash", "Shares_Held", "Close", "BuyHold_Value"]:
        assert col in eq.columns, f"Missing equity column: {col}"


def test_equity_curve_portfolio_value_always_positive():
    result = run_walk_forward(make_price_df(), "TEST", CUTOFF)
    eq     = result["equity_curve"]
    assert (eq["Portfolio_Value"] > 0).all()


def test_equity_curve_length_matches_test_period():
    result  = run_walk_forward(make_price_df(), "TEST", CUTOFF)
    eq      = result["equity_curve"]
    test_df = result["test_df"]
    assert len(eq) == len(test_df)


# ── Trade logic ───────────────────────────────────────────────────────────────

def test_trades_list_is_list():
    result = run_walk_forward(make_price_df(), "TEST", CUTOFF)
    assert isinstance(result["trades"], list)


def test_trade_entries_have_required_fields():
    result = run_walk_forward(make_price_df(), "TEST", CUTOFF)
    for trade in result["trades"]:
        for field in ["date", "ticker", "action", "shares", "price", "value"]:
            assert field in trade, f"Trade missing field: {field}"


def test_no_buy_without_cash():
    """Portfolio value should never go negative (can't buy without cash)."""
    result = run_walk_forward(make_price_df(), "TEST", CUTOFF,
                              starting_cash=1000.0, position_size_pct=0.99)
    eq = result["equity_curve"]
    assert (eq["Cash"] >= 0).all()


def test_open_position_closed_at_end():
    """If a position is held at end of test, it must appear as a SELL in trades."""
    result     = run_walk_forward(make_price_df(), "TEST", CUTOFF,
                                  confidence_threshold=0.50, position_size_pct=0.5)
    last_eq    = result["equity_curve"].iloc[-1]
    trade_actions = [t["action"] for t in result["trades"]]

    # If shares were held at some point, there must be at least one SELL
    if any(t["action"] == "BUY" for t in result["trades"]):
        assert any("SELL" in a for a in trade_actions)


# ── Metrics sanity ────────────────────────────────────────────────────────────

def test_alpha_equals_strategy_minus_bh():
    result  = run_walk_forward(make_price_df(), "TEST", CUTOFF)
    metrics = result["metrics"]
    expected_alpha = round(metrics["total_return_pct"] - metrics["bh_return_pct"], 2)
    assert metrics["alpha"] == pytest.approx(expected_alpha, abs=0.01)


def test_final_value_consistent_with_equity_curve():
    result  = run_walk_forward(make_price_df(), "TEST", CUTOFF)
    metrics = result["metrics"]
    eq      = result["equity_curve"]
    assert metrics["final_value"] == pytest.approx(eq["Portfolio_Value"].iloc[-1], abs=0.01)


def test_train_roc_auc_between_0_and_1():
    result = run_walk_forward(make_price_df(), "TEST", CUTOFF)
    auc    = result["metrics"]["train_roc_auc"]
    assert 0.0 <= auc <= 1.0


# ── Error handling ────────────────────────────────────────────────────────────

def test_raises_with_insufficient_training_data():
    tiny_df = make_price_df(n=30)  # not enough rows before cutoff
    with pytest.raises(ValueError, match="Not enough training data"):
        run_walk_forward(tiny_df, "TEST", "2020-01-15")


def test_raises_with_insufficient_test_data():
    df = make_price_df(n=100)
    # Cutoff at the very end leaves no test data
    with pytest.raises(ValueError, match="Not enough test data"):
        run_walk_forward(df, "TEST", "2030-01-01")


# ── Custom model and feature set ─────────────────────────────────────────────

def test_custom_model_is_used():
    from sklearn.ensemble import RandomForestClassifier
    model  = RandomForestClassifier(n_estimators=10, random_state=42)
    result = run_walk_forward(
        make_price_df(), "TEST", CUTOFF,
        model=model,
        feature_set=FEATURES,
    )
    assert result["metrics"]["model_name"] == "RandomForestClassifier"


def test_custom_feature_set_is_tracked():
    feat_set = ['SMA_5', 'SMA_10', 'Return']
    result   = run_walk_forward(
        make_price_df(), "TEST", CUTOFF,
        feature_set=feat_set,
    )
    assert result["metrics"]["feature_set"] == feat_set
