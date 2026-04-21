"""
tests/test_portfolio.py

Run with:  pytest tests/

All tests use in-memory state dicts — no files are written to disk.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.portfolio.portfolio import (
    _default_state,
    execute_signal,
    get_portfolio_summary,
    reset_portfolio,
    load_portfolio,
    _portfolio_value,
    _record_trade,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def fresh_state(cash=100_000.0):
    """Return a clean in-memory portfolio state (no file I/O)."""
    return _default_state(cash)


def state_with_position(ticker="TSLA", shares=10, avg_cost=200.0, cash=80_000.0):
    """Return a state that already holds a position."""
    state = fresh_state(cash=cash + shares * avg_cost)
    state["cash"] = cash
    state["positions"][ticker] = {"shares": shares, "avg_cost": avg_cost}
    return state


# ── _default_state ────────────────────────────────────────────────────────────

def test_default_state_structure():
    state = _default_state(50_000.0)
    assert state["starting_cash"] == 50_000.0
    assert state["cash"]          == 50_000.0
    assert state["positions"]     == {}
    assert state["trades"]        == []


def test_default_state_default_cash():
    state = _default_state()
    assert state["cash"] == 100_000.0


# ── _portfolio_value ──────────────────────────────────────────────────────────

def test_portfolio_value_no_positions():
    state = fresh_state(100_000.0)
    assert _portfolio_value(state, {}) == pytest.approx(100_000.0)


def test_portfolio_value_with_position():
    state = state_with_position("NVDA", shares=10, avg_cost=500.0, cash=50_000.0)
    # 10 shares @ current price $600 + $50k cash = $56k
    val = _portfolio_value(state, {"NVDA": 600.0})
    assert val == pytest.approx(56_000.0)


def test_portfolio_value_falls_back_to_avg_cost():
    """If current price not provided, should use avg_cost."""
    state = state_with_position("AAPL", shares=5, avg_cost=150.0, cash=90_000.0)
    val = _portfolio_value(state, {})  # no price provided
    assert val == pytest.approx(90_000.0 + 5 * 150.0)


def test_portfolio_value_multiple_positions():
    state = fresh_state(50_000.0)
    state["positions"] = {
        "TSLA": {"shares": 10, "avg_cost": 200.0},
        "NVDA": {"shares": 5,  "avg_cost": 500.0},
    }
    prices = {"TSLA": 220.0, "NVDA": 520.0}
    expected = 50_000 + 10 * 220 + 5 * 520
    assert _portfolio_value(state, prices) == pytest.approx(expected)


# ── execute_signal — BUY ──────────────────────────────────────────────────────

@patch("src.portfolio.portfolio.save_portfolio")
def test_buy_signal_creates_position(mock_save):
    state = fresh_state(100_000.0)
    trade = execute_signal(state, "TSLA", prediction=1, probability=0.75,
                           current_price=200.0, confidence_threshold=0.60,
                           position_size_pct=0.10)
    assert trade is not None
    assert trade["action"] == "BUY"
    assert "TSLA" in state["positions"]


@patch("src.portfolio.portfolio.save_portfolio")
def test_buy_reduces_cash(mock_save):
    state = fresh_state(100_000.0)
    execute_signal(state, "TSLA", 1, 0.75, 200.0,
                   confidence_threshold=0.60, position_size_pct=0.10)
    assert state["cash"] < 100_000.0


@patch("src.portfolio.portfolio.save_portfolio")
def test_buy_records_correct_shares(mock_save):
    state = fresh_state(100_000.0)
    # 10% of $100k = $10k spend, price $200 → 50 shares
    execute_signal(state, "TSLA", 1, 0.75, 200.0,
                   confidence_threshold=0.60, position_size_pct=0.10)
    assert state["positions"]["TSLA"]["shares"] == 50


@patch("src.portfolio.portfolio.save_portfolio")
def test_buy_records_avg_cost(mock_save):
    state = fresh_state(100_000.0)
    execute_signal(state, "TSLA", 1, 0.75, 200.0,
                   confidence_threshold=0.60, position_size_pct=0.10)
    assert state["positions"]["TSLA"]["avg_cost"] == pytest.approx(200.0)


@patch("src.portfolio.portfolio.save_portfolio")
def test_no_double_buy_same_ticker(mock_save):
    """Should not buy again if we already hold the ticker."""
    state = state_with_position("TSLA", shares=10, avg_cost=200.0, cash=80_000.0)
    trade = execute_signal(state, "TSLA", 1, 0.80, 210.0,
                           confidence_threshold=0.60, position_size_pct=0.10)
    assert trade is None
    assert state["positions"]["TSLA"]["shares"] == 10  # unchanged


@patch("src.portfolio.portfolio.save_portfolio")
def test_buy_below_confidence_threshold_does_nothing(mock_save):
    state = fresh_state(100_000.0)
    trade = execute_signal(state, "TSLA", 1, 0.50,  # below 0.60 threshold
                           200.0, confidence_threshold=0.60)
    assert trade is None
    assert state["positions"] == {}
    assert state["cash"] == 100_000.0


@patch("src.portfolio.portfolio.save_portfolio")
def test_buy_with_insufficient_cash_does_nothing(mock_save):
    """If cash is too low to buy even 1 share, no trade should execute."""
    state = fresh_state(10.0)  # only $10
    trade = execute_signal(state, "TSLA", 1, 0.80, 200.0,
                           confidence_threshold=0.60, position_size_pct=0.10)
    assert trade is None
    assert state["positions"] == {}


# ── execute_signal — SELL ─────────────────────────────────────────────────────

@patch("src.portfolio.portfolio.save_portfolio")
def test_sell_signal_closes_position(mock_save):
    state = state_with_position("TSLA", shares=10, avg_cost=200.0, cash=80_000.0)
    trade = execute_signal(state, "TSLA", prediction=0, probability=0.70,
                           current_price=220.0, confidence_threshold=0.60)
    assert trade is not None
    assert trade["action"] == "SELL"
    assert "TSLA" not in state["positions"]


@patch("src.portfolio.portfolio.save_portfolio")
def test_sell_increases_cash(mock_save):
    state = state_with_position("TSLA", shares=10, avg_cost=200.0, cash=80_000.0)
    execute_signal(state, "TSLA", 0, 0.70, 220.0, confidence_threshold=0.60)
    assert state["cash"] == pytest.approx(80_000.0 + 10 * 220.0)


@patch("src.portfolio.portfolio.save_portfolio")
def test_sell_records_correct_pnl(mock_save):
    state = state_with_position("TSLA", shares=10, avg_cost=200.0, cash=80_000.0)
    trade = execute_signal(state, "TSLA", 0, 0.70, 220.0, confidence_threshold=0.60)
    # PnL = (220 - 200) * 10 = $200
    assert trade["pnl"] == pytest.approx(200.0)


@patch("src.portfolio.portfolio.save_portfolio")
def test_sell_records_negative_pnl_on_loss(mock_save):
    state = state_with_position("TSLA", shares=10, avg_cost=200.0, cash=80_000.0)
    trade = execute_signal(state, "TSLA", 0, 0.70, 180.0, confidence_threshold=0.60)
    # PnL = (180 - 200) * 10 = -$200
    assert trade["pnl"] == pytest.approx(-200.0)


@patch("src.portfolio.portfolio.save_portfolio")
def test_sell_with_no_position_does_nothing(mock_save):
    state = fresh_state(100_000.0)
    trade = execute_signal(state, "TSLA", 0, 0.75, 200.0, confidence_threshold=0.60)
    assert trade is None


@patch("src.portfolio.portfolio.save_portfolio")
def test_sell_below_threshold_does_nothing(mock_save):
    state = state_with_position("TSLA", shares=10, avg_cost=200.0, cash=80_000.0)
    trade = execute_signal(state, "TSLA", 0, 0.50,  # below threshold
                           220.0, confidence_threshold=0.60)
    assert trade is None
    assert "TSLA" in state["positions"]  # position still open


# ── Trade recording ───────────────────────────────────────────────────────────

@patch("src.portfolio.portfolio.save_portfolio")
def test_trade_appended_to_history(mock_save):
    state = fresh_state(100_000.0)
    execute_signal(state, "TSLA", 1, 0.75, 200.0, confidence_threshold=0.60)
    assert len(state["trades"]) == 1


@patch("src.portfolio.portfolio.save_portfolio")
def test_trade_record_has_required_fields(mock_save):
    state = fresh_state(100_000.0)
    trade = execute_signal(state, "TSLA", 1, 0.75, 200.0, confidence_threshold=0.60)
    for field in ["date", "ticker", "action", "shares", "price", "value", "probability", "pnl"]:
        assert field in trade, f"Trade missing field: {field}"


@patch("src.portfolio.portfolio.save_portfolio")
def test_buy_trade_pnl_is_none(mock_save):
    """BUY trades should have pnl=None since P&L isn't realized yet."""
    state = fresh_state(100_000.0)
    trade = execute_signal(state, "TSLA", 1, 0.75, 200.0, confidence_threshold=0.60)
    assert trade["pnl"] is None


# ── get_portfolio_summary ─────────────────────────────────────────────────────

def test_summary_keys_present():
    state   = fresh_state(100_000.0)
    summary = get_portfolio_summary(state, {})
    for key in ["cash", "equity", "total_value", "starting_cash",
                "total_return", "total_return_pct", "realized_pnl",
                "positions", "num_trades"]:
        assert key in summary, f"Summary missing key: {key}"


def test_summary_empty_portfolio():
    state   = fresh_state(100_000.0)
    summary = get_portfolio_summary(state, {})
    assert summary["cash"]         == pytest.approx(100_000.0)
    assert summary["equity"]       == pytest.approx(0.0)
    assert summary["total_value"]  == pytest.approx(100_000.0)
    assert summary["total_return"] == pytest.approx(0.0)
    assert summary["positions"]    == []


def test_summary_total_value_equals_cash_plus_equity():
    state   = state_with_position("NVDA", shares=10, avg_cost=500.0, cash=50_000.0)
    prices  = {"NVDA": 600.0}
    summary = get_portfolio_summary(state, prices)
    assert summary["total_value"] == pytest.approx(summary["cash"] + summary["equity"])


def test_summary_unrealized_pnl_correct():
    state   = state_with_position("NVDA", shares=10, avg_cost=500.0, cash=50_000.0)
    prices  = {"NVDA": 600.0}
    summary = get_portfolio_summary(state, prices)
    pos     = summary["positions"][0]
    # Unrealized PnL = (600 - 500) * 10 = $1000
    assert pos["unrealized_pnl"] == pytest.approx(1000.0)


def test_summary_pnl_pct_correct():
    state   = state_with_position("NVDA", shares=10, avg_cost=500.0, cash=50_000.0)
    prices  = {"NVDA": 550.0}
    summary = get_portfolio_summary(state, prices)
    pos     = summary["positions"][0]
    # PnL % = (550 - 500) / 500 * 100 = 10%
    assert pos["pnl_pct"] == pytest.approx(10.0)


def test_summary_realized_pnl_sums_trades():
    state = fresh_state(100_000.0)
    state["trades"] = [
        {"pnl": 500.0},
        {"pnl": -200.0},
        {"pnl": None},   # BUY trade — should be ignored
    ]
    summary = get_portfolio_summary(state, {})
    assert summary["realized_pnl"] == pytest.approx(300.0)


def test_summary_total_return_pct_correct():
    # Start with $100k, now worth $110k → +10%
    state = fresh_state(100_000.0)
    state["cash"] = 110_000.0
    summary = get_portfolio_summary(state, {})
    assert summary["total_return_pct"] == pytest.approx(10.0)


# ── reset_portfolio ───────────────────────────────────────────────────────────

@patch("src.portfolio.portfolio.save_portfolio")
def test_reset_clears_positions_and_trades(mock_save):
    state = state_with_position("TSLA", shares=10, avg_cost=200.0)
    state["trades"] = [{"pnl": 100.0}]
    reset_state = reset_portfolio(starting_cash=50_000.0)
    assert reset_state["positions"] == {}
    assert reset_state["trades"]    == []
    assert reset_state["cash"]      == 50_000.0
