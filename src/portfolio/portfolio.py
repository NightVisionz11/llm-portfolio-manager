"""
src/portfolio/portfolio.py

Simulated paper-trading portfolio manager.
Persists state to data/portfolio/portfolio.json so it survives app restarts.
"""

import json
import os
from datetime import datetime
from src.utils.config import PORTFOLIO_DIR


PORTFOLIO_FILE = os.path.join(PORTFOLIO_DIR, "portfolio.json")


def _default_state(starting_cash: float = 100_000.0) -> dict:
    return {
        "starting_cash": starting_cash,
        "cash": starting_cash,
        "positions": {},   # ticker -> {shares, avg_cost}
        "trades": [],      # list of trade dicts
    }


def load_portfolio() -> dict:
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    return _default_state()


def save_portfolio(state: dict):
    os.makedirs(PORTFOLIO_DIR, exist_ok=True)
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(state, f, indent=2)


def reset_portfolio(starting_cash: float = 100_000.0) -> dict:
    state = _default_state(starting_cash)
    save_portfolio(state)
    return state


# ── Trade execution ──────────────────────────────────────────────────────────

def execute_signal(
    state: dict,
    ticker: str,
    prediction: int,
    probability: float,
    current_price: float,
    confidence_threshold: float = 0.60,
    position_size_pct: float = 0.10,   # risk 10% of portfolio per trade
) -> dict | None:
    """
    Auto-execute a trade based on model signal.

    Rules:
    - Prediction == 1 AND prob >= threshold  → BUY (if not already long)
    - Prediction == 0 AND prob >= threshold  → SELL / close position (if long)
    - Below threshold → hold, do nothing

    Returns the trade dict if a trade was made, else None.
    """
    portfolio_value = _portfolio_value(state, {ticker: current_price})
    trade = None

    if prediction == 1 and probability >= confidence_threshold:
        # Only buy if we don't already hold this ticker
        if ticker not in state["positions"]:
            spend = portfolio_value * position_size_pct
            shares = int(spend // current_price)
            if shares > 0 and state["cash"] >= shares * current_price:
                cost = shares * current_price
                state["cash"] -= cost
                state["positions"][ticker] = {
                    "shares": shares,
                    "avg_cost": current_price,
                }
                trade = _record_trade(state, ticker, "BUY", shares, current_price, probability)

    elif prediction == 0 and probability >= confidence_threshold:
        # Close position if we hold this ticker
        if ticker in state["positions"]:
            pos = state["positions"].pop(ticker)
            proceeds = pos["shares"] * current_price
            state["cash"] += proceeds
            pnl = (current_price - pos["avg_cost"]) * pos["shares"]
            trade = _record_trade(state, ticker, "SELL", pos["shares"], current_price, probability, pnl=pnl)

    if trade:
        save_portfolio(state)

    return trade


def _record_trade(state, ticker, action, shares, price, probability, pnl=None) -> dict:
    trade = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ticker": ticker,
        "action": action,
        "shares": shares,
        "price": round(price, 4),
        "value": round(shares * price, 2),
        "probability": round(probability, 4),
        "pnl": round(pnl, 2) if pnl is not None else None,
    }
    state["trades"].append(trade)
    return trade


# ── Valuation ────────────────────────────────────────────────────────────────

def _portfolio_value(state: dict, prices: dict) -> float:
    equity = sum(
        pos["shares"] * prices.get(ticker, pos["avg_cost"])
        for ticker, pos in state["positions"].items()
    )
    return state["cash"] + equity


def get_portfolio_summary(state: dict, current_prices: dict) -> dict:
    """
    Returns a summary dict with total value, P&L, positions detail.
    current_prices: {ticker: price}
    """
    positions_detail = []
    total_equity = 0.0

    for ticker, pos in state["positions"].items():
        price = current_prices.get(ticker, pos["avg_cost"])
        market_value = pos["shares"] * price
        unrealized_pnl = (price - pos["avg_cost"]) * pos["shares"]
        pnl_pct = ((price - pos["avg_cost"]) / pos["avg_cost"]) * 100
        total_equity += market_value
        positions_detail.append({
            "ticker": ticker,
            "shares": pos["shares"],
            "avg_cost": round(pos["avg_cost"], 4),
            "current_price": round(price, 4),
            "market_value": round(market_value, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
        })

    total_value = state["cash"] + total_equity
    total_return = total_value - state["starting_cash"]
    total_return_pct = (total_return / state["starting_cash"]) * 100

    realized_pnl = sum(t["pnl"] for t in state["trades"] if t["pnl"] is not None)

    return {
        "cash": round(state["cash"], 2),
        "equity": round(total_equity, 2),
        "total_value": round(total_value, 2),
        "starting_cash": state["starting_cash"],
        "total_return": round(total_return, 2),
        "total_return_pct": round(total_return_pct, 2),
        "realized_pnl": round(realized_pnl, 2),
        "positions": positions_detail,
        "num_trades": len(state["trades"]),
    }