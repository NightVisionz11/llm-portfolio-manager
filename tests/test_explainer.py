"""
tests/test_explainer.py

Run with:  pytest tests/

Note: Ollama tests are skipped automatically if Ollama isn't running.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.llm.explainer import (
    ollama_available,
    explain_prediction,
    explain_backtest,
    _rule_based_prediction,
    _rule_based_backtest,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_ROW = {
    "Close":        150.0,
    "SMA_5":        148.0,
    "SMA_10":       145.0,
    "RSI_14":       62.5,
    "MACD":         1.234,
    "Return":       0.012,
    "Volatility_10": 0.018,
    "Volume_change": 0.05,
}


# ── Rule-based prediction ─────────────────────────────────────────────────────

def test_rule_based_prediction_returns_string():
    result = _rule_based_prediction("TSLA", 1, 0.72, SAMPLE_ROW)
    assert isinstance(result, str)
    assert len(result) > 0


def test_rule_based_prediction_contains_ticker():
    result = _rule_based_prediction("NVDA", 1, 0.65, SAMPLE_ROW)
    assert "NVDA" in result


def test_rule_based_prediction_up_direction():
    result = _rule_based_prediction("AAPL", 1, 0.70, SAMPLE_ROW)
    assert "UP" in result or "up" in result or "📈" in result


def test_rule_based_prediction_down_direction():
    result = _rule_based_prediction("AAPL", 0, 0.65, SAMPLE_ROW)
    assert "DOWN" in result or "down" in result or "📉" in result


def test_rule_based_prediction_high_confidence_label():
    result = _rule_based_prediction("TSLA", 1, 0.80, SAMPLE_ROW)
    assert "high" in result


def test_rule_based_prediction_moderate_confidence_label():
    result = _rule_based_prediction("TSLA", 1, 0.58, SAMPLE_ROW)
    assert "moderate" in result


def test_rule_based_prediction_low_confidence_label():
    result = _rule_based_prediction("TSLA", 1, 0.51, SAMPLE_ROW)
    assert "low" in result


def test_rule_based_prediction_uptrend_comment():
    row = {**SAMPLE_ROW, "SMA_5": 155.0, "SMA_10": 145.0}  # SMA5 > SMA10
    result = _rule_based_prediction("TSLA", 1, 0.65, row)
    assert "upward" in result.lower() or "above" in result.lower()


def test_rule_based_prediction_downtrend_comment():
    row = {**SAMPLE_ROW, "SMA_5": 140.0, "SMA_10": 150.0}  # SMA5 < SMA10
    result = _rule_based_prediction("TSLA", 0, 0.65, row)
    assert "downward" in result.lower() or "below" in result.lower()


def test_rule_based_prediction_missing_indicators_doesnt_crash():
    """Should handle missing optional indicator keys gracefully."""
    result = _rule_based_prediction("TSLA", 1, 0.65, {})
    assert isinstance(result, str)


# ── Rule-based backtest ───────────────────────────────────────────────────────

def test_rule_based_backtest_returns_string():
    result = _rule_based_backtest(1.2, -0.15, 1.3, 1.1)
    assert isinstance(result, str)
    assert len(result) > 0


def test_rule_based_backtest_outperformed():
    result = _rule_based_backtest(1.2, -0.10, 1.5, 1.2)  # strat > market
    assert "outperformed" in result


def test_rule_based_backtest_underperformed():
    result = _rule_based_backtest(0.3, -0.30, 0.9, 1.2)  # strat < market
    assert "underperformed" in result


def test_rule_based_backtest_strong_sharpe_comment():
    result = _rule_based_backtest(1.5, -0.08, 1.2, 1.1)  # sharpe > 1
    assert "strong" in result


def test_rule_based_backtest_weak_sharpe_comment():
    result = _rule_based_backtest(0.2, -0.35, 0.9, 1.1)  # sharpe < 0.5
    assert "weak" in result


def test_rule_based_backtest_significant_drawdown_comment():
    result = _rule_based_backtest(0.4, -0.40, 1.1, 1.0)  # max_dd < -0.25
    assert "significant" in result


# ── explain_prediction falls back when Ollama is down ────────────────────────

def test_explain_prediction_falls_back_when_ollama_down():
    """When Ollama is unreachable, should return the rule-based string."""
    with patch("src.llm.explainer.ollama_available", return_value=False):
        result = explain_prediction("TSLA", 1, 0.72, SAMPLE_ROW)
    assert isinstance(result, str)
    assert "TSLA" in result


def test_explain_backtest_falls_back_when_ollama_down():
    with patch("src.llm.explainer.ollama_available", return_value=False):
        result = explain_backtest(
            sharpe=1.1, max_dd=-0.12,
            final_strat=1.3, final_market=1.1,
        )
    assert isinstance(result, str)
    assert "outperformed" in result


# ── explain_prediction falls back when Ollama call errors ────────────────────

def test_explain_prediction_falls_back_on_ollama_error():
    """Even if Ollama is 'available', a bad response should fall back."""
    with patch("src.llm.explainer.ollama_available", return_value=True), \
         patch("src.llm.explainer._call_ollama", return_value=None):
        result = explain_prediction("NVDA", 0, 0.68, SAMPLE_ROW)
    assert isinstance(result, str)
    assert len(result) > 0


# ── Ollama live tests (skipped if not running) ────────────────────────────────

ollama_running = ollama_available()

@pytest.mark.skipif(not ollama_running, reason="Ollama not running locally")
def test_ollama_explain_prediction_returns_non_empty():
    result = explain_prediction("TSLA", 1, 0.72, SAMPLE_ROW)
    assert isinstance(result, str)
    assert len(result) > 50  # should be a meaningful paragraph


@pytest.mark.skipif(not ollama_running, reason="Ollama not running locally")
def test_ollama_explain_backtest_returns_non_empty():
    result = explain_backtest(
        sharpe=1.1, max_dd=-0.12,
        final_strat=1.3, final_market=1.1,
        ticker="NVDA", total_return=30.0,
        bh_return=10.0, num_trades=12, win_rate=58.0,
    )
    assert isinstance(result, str)
    assert len(result) > 50
