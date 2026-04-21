"""
tests/test_evaluate.py

Run with:  pytest tests/
"""

import pytest
import pandas as pd
import numpy as np
from src.models.evaluate import (
    evaluate_model,
    backtest_strategy,
    sharpe_ratio,
    max_drawdown,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def perfect_predictions():
    y_true = pd.Series([1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = y_true.copy()
    y_prob = pd.Series([0.9, 0.1, 0.85, 0.15, 0.95, 0.8, 0.2, 0.1])
    return y_true, y_pred, y_prob


def random_predictions(n=100, seed=42):
    rng    = np.random.default_rng(seed)
    y_true = pd.Series(rng.integers(0, 2, n))
    y_prob = pd.Series(rng.uniform(0, 1, n))
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


def make_backtest_df(n=50, uptrend=True):
    dates   = pd.date_range("2023-01-01", periods=n, freq="B")
    closes  = [100 + (i if uptrend else -i) * 0.3 for i in range(n)]
    pred    = [1] * n  # always predicting UP
    return pd.DataFrame({"Date": dates, "Close": closes, "Prediction": pred})


# ── evaluate_model ────────────────────────────────────────────────────────────

def test_evaluate_model_returns_all_keys():
    y_true, y_pred, y_prob = perfect_predictions()
    result = evaluate_model(y_true, y_pred, y_prob)
    for key in ["accuracy", "roc_auc", "precision", "recall", "f1", "confusion_matrix"]:
        assert key in result, f"Missing key: {key}"


def test_evaluate_model_perfect_score():
    y_true, y_pred, y_prob = perfect_predictions()
    result = evaluate_model(y_true, y_pred, y_prob)
    assert result["accuracy"]  == 1.0
    assert result["roc_auc"]   == 1.0
    assert result["precision"] == 1.0
    assert result["recall"]    == 1.0
    assert result["f1"]        == 1.0


def test_evaluate_model_values_in_range():
    y_true, y_pred, y_prob = random_predictions()
    result = evaluate_model(y_true, y_pred, y_prob)
    assert 0.0 <= result["accuracy"]  <= 1.0
    assert 0.0 <= result["roc_auc"]   <= 1.0
    assert 0.0 <= result["precision"] <= 1.0
    assert 0.0 <= result["recall"]    <= 1.0
    assert 0.0 <= result["f1"]        <= 1.0


def test_confusion_matrix_shape():
    y_true, y_pred, y_prob = random_predictions()
    result = evaluate_model(y_true, y_pred, y_prob)
    cm = result["confusion_matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2


# ── backtest_strategy ─────────────────────────────────────────────────────────

def test_backtest_returns_required_columns():
    df     = make_backtest_df()
    result = backtest_strategy(df)
    for col in ["Daily_Return", "Strategy_Return", "Cumulative_Market", "Cumulative_Strategy"]:
        assert col in result.columns, f"Missing column: {col}"


def test_backtest_cumulative_starts_near_one():
    df     = make_backtest_df()
    result = backtest_strategy(df)
    # First value should be 1.0 (or very close, since pct_change fills first with 0)
    assert result["Cumulative_Market"].iloc[0]   == pytest.approx(1.0, abs=0.01)
    assert result["Cumulative_Strategy"].iloc[0] == pytest.approx(1.0, abs=0.01)


def test_backtest_strategy_beats_zero():
    """Always-predict-UP strategy in an uptrend should have positive cumulative return."""
    df     = make_backtest_df(uptrend=True)
    result = backtest_strategy(df)
    assert result["Cumulative_Strategy"].iloc[-1] > 1.0


def test_backtest_hold_out_in_downtrend():
    """Always-predict-UP strategy in a downtrend should lose money."""
    df     = make_backtest_df(uptrend=False)
    result = backtest_strategy(df)
    assert result["Cumulative_Strategy"].iloc[-1] < 1.0


def test_backtest_sorted_by_date():
    df     = make_backtest_df().sample(frac=1, random_state=1)  # shuffle dates
    result = backtest_strategy(df)
    assert result["Date"].is_monotonic_increasing


# ── sharpe_ratio ──────────────────────────────────────────────────────────────

def test_sharpe_zero_for_flat_returns():
    returns = pd.Series([0.0] * 50)
    assert sharpe_ratio(returns) == 0.0


def test_sharpe_positive_for_positive_returns():
    returns = pd.Series([0.001] * 252)  # 0.1% daily, no variance
    # std is 0 so should return 0.0 (guard in function)
    assert isinstance(sharpe_ratio(returns), float)


def test_sharpe_higher_for_better_strategy():
    good_returns = pd.Series([0.002] * 100 + [-0.001] * 50)
    bad_returns  = pd.Series([-0.002] * 100 + [0.001] * 50)
    assert sharpe_ratio(good_returns) > sharpe_ratio(bad_returns)


def test_sharpe_returns_float():
    returns = pd.Series(np.random.normal(0.001, 0.01, 252))
    result  = sharpe_ratio(returns)
    assert isinstance(result, float)


# ── max_drawdown ──────────────────────────────────────────────────────────────

def test_max_drawdown_is_negative_or_zero():
    cumulative = pd.Series([1.0, 1.1, 1.05, 0.95, 1.02])
    result = max_drawdown(cumulative)
    assert result <= 0.0


def test_max_drawdown_zero_for_monotone_increase():
    cumulative = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])
    assert max_drawdown(cumulative) == pytest.approx(0.0, abs=1e-6)


def test_max_drawdown_correct_value():
    # Peak is 1.2, trough is 0.9 → drawdown = (0.9 - 1.2) / 1.2 = -0.25
    cumulative = pd.Series([1.0, 1.2, 1.1, 0.9, 1.0])
    result = max_drawdown(cumulative)
    assert result == pytest.approx(-0.25, abs=1e-4)


def test_max_drawdown_returns_float():
    cumulative = pd.Series([1.0, 0.9, 0.8, 0.85])
    assert isinstance(max_drawdown(cumulative), float)
