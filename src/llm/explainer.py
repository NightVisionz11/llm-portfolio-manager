"""
src/llm/explainer.py

Generates human-readable explanations for model predictions and backtest results.

Priority order:
  1. Ollama (local LLM) — if running at localhost:11434
  2. Rule-based template fallback — always works, no dependencies

Setup for Ollama:
  1. Download from https://ollama.com
  2. Run: ollama pull llama3.2
  3. Ollama starts automatically — no extra steps needed
"""

import requests
from src.llm.prompts import prediction_prompt, backtest_prompt

OLLAMA_BASE_URL  = "http://localhost:11434"
OLLAMA_API_URL   = f"{OLLAMA_BASE_URL}/api/generate"
DEFAULT_MODEL    = "llama3.2"
REQUEST_TIMEOUT  = 60  # seconds — LLM inference can be slow on CPU


# ── Ollama availability check ─────────────────────────────────────────────────

def ollama_available() -> bool:
    """
    Returns True if the Ollama server is reachable.
    Used by the Streamlit app to show/hide the LLM status indicator.
    """
    try:
        r = requests.get(OLLAMA_BASE_URL, timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def list_ollama_models() -> list[str]:
    """
    Returns a list of locally available Ollama model names.
    Falls back to a sensible default list if Ollama isn't running.
    """
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if r.status_code == 200:
            models = r.json().get("models", [])
            return [m["name"] for m in models] if models else [DEFAULT_MODEL]
    except Exception:
        pass
    return [DEFAULT_MODEL, "mistral", "llama3.1:8b", "phi3"]


# ── Core LLM call ─────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, model: str) -> str | None:
    """
    Sends a prompt to Ollama and returns the response text.
    Returns None on any failure so callers can fall back gracefully.
    """
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        print("[Ollama] Not reachable — is Ollama running?")
    except requests.exceptions.Timeout:
        print(f"[Ollama] Request timed out after {REQUEST_TIMEOUT}s.")
    except Exception as e:
        print(f"[Ollama] Unexpected error: {e}")
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def explain_prediction(
    stock_name: str,
    prediction: int,
    probability: float,
    latest_row: dict,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Returns a plain-English explanation of a next-day price prediction.

    Uses Ollama if available, otherwise falls back to the rule-based template.

    Args:
        stock_name:  Ticker symbol or company name, e.g. "TSLA"
        prediction:  0 (down) or 1 (up)
        probability: Model confidence for the predicted class (0–1)
        latest_row:  Dict of the most recent row's feature values
        model:       Ollama model name to use (default: llama3.2)
    """
    if ollama_available():
        prompt = prediction_prompt(stock_name, prediction, probability, latest_row)
        result = _call_ollama(prompt, model)
        if result:
            return result
        # Ollama was reachable but the call failed — fall through to template
        print("[Ollama] Call failed — using rule-based fallback.")

    return _rule_based_prediction(stock_name, prediction, probability, latest_row)


def explain_backtest(
    sharpe: float,
    max_dd: float,
    final_strat: float,
    final_market: float,
    ticker: str = "",
    total_return: float = 0.0,
    bh_return: float = 0.0,
    num_trades: int = 0,
    win_rate: float = 0.0,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Returns a plain-English summary of backtest performance.

    Uses Ollama if available, otherwise falls back to the rule-based template.

    Args:
        sharpe:       Annualised Sharpe ratio
        max_dd:       Max drawdown as a negative fraction, e.g. -0.18
        final_strat:  Cumulative strategy multiplier, e.g. 1.12
        final_market: Cumulative buy-and-hold multiplier, e.g. 1.08
        ticker:       Ticker symbol, e.g. "NVDA"
        total_return: Strategy total return %, e.g. 12.5
        bh_return:    Buy-and-hold total return %, e.g. 8.2
        num_trades:   Total number of trades executed
        win_rate:     Percentage of winning trades, e.g. 55.0
        model:        Ollama model name to use (default: llama3.2)
    """
    if ollama_available():
        prompt = backtest_prompt(
            ticker, total_return, bh_return, sharpe, max_dd, num_trades, win_rate
        )
        result = _call_ollama(prompt, model)
        if result:
            return result
        print("[Ollama] Call failed — using rule-based fallback.")

    return _rule_based_backtest(sharpe, max_dd, final_strat, final_market)


# ── Rule-based fallbacks ──────────────────────────────────────────────────────

def _rule_based_prediction(
    stock_name: str,
    prediction: int,
    probability: float,
    latest_row: dict,
) -> str:
    """Template-based explanation — no LLM required."""
    direction  = "UP 📈" if prediction == 1 else "DOWN 📉"
    confidence = (
        "high" if probability >= 0.65
        else "moderate" if probability >= 0.55
        else "low"
    )

    sma5  = latest_row.get("SMA_5")
    sma10 = latest_row.get("SMA_10")
    ret   = latest_row.get("Return")

    trend_comment = ""
    if sma5 is not None and sma10 is not None:
        if sma5 > sma10:
            trend_comment = (
                "The short-term moving average is above the longer-term average, "
                "suggesting recent upward momentum."
            )
        else:
            trend_comment = (
                "The short-term moving average is below the longer-term average, "
                "suggesting recent downward pressure."
            )

    return_comment = ""
    if ret is not None:
        pct = round(float(ret) * 100, 2)
        if pct > 0:
            return_comment = f"The stock rose {pct}% today."
        elif pct < 0:
            return_comment = f"The stock fell {abs(pct)}% today."
        else:
            return_comment = "The stock was roughly flat today."

    return (
        f"**{stock_name} Prediction: {direction}**\n\n"
        f"The model is {confidence} confidence ({probability:.0%}) that {stock_name} "
        f"will move {direction.split()[0].lower()} tomorrow.\n\n"
        f"{return_comment} {trend_comment}\n\n"
        f"*Note: This is a machine learning estimate based on historical price patterns, "
        f"not financial advice. Always do your own research.*"
    )


def _rule_based_backtest(
    sharpe: float,
    max_dd: float,
    final_strat: float,
    final_market: float,
) -> str:
    """Template-based backtest summary — no LLM required."""
    outperform = "outperformed" if final_strat >= final_market else "underperformed"

    sr_comment = (
        "strong risk-adjusted returns" if sharpe > 1
        else "acceptable risk-adjusted returns" if sharpe > 0.5
        else "weak risk-adjusted returns"
    )
    dd_comment = (
        "relatively shallow drawdowns" if max_dd > -0.10
        else "moderate drawdowns" if max_dd > -0.25
        else "significant drawdowns — position sizing matters"
    )

    return (
        f"The strategy {outperform} buy-and-hold "
        f"({final_strat:.2f}x vs {final_market:.2f}x cumulative return).\n"
        f"Sharpe ratio of {sharpe} indicates {sr_comment}.\n"
        f"Max drawdown of {max_dd:.1%} suggests {dd_comment}."
    )