"""
src/llm/prompts.py

Prompt templates for Ollama LLM explanations.
Keeps all prompt logic in one place so they're easy to iterate on
without touching the explainer or app code.
"""


def prediction_prompt(
    stock_name: str,
    prediction: int,
    probability: float,
    latest_row: dict,
) -> str:
    """
    Builds a prompt asking the LLM to explain a next-day price prediction.

    Args:
        stock_name:  e.g. "TSLA"
        prediction:  0 (down) or 1 (up)
        probability: model confidence for the predicted class
        latest_row:  dict of the latest row's feature values
    """
    action = "UP" if prediction == 1 else "DOWN"

    # Pull indicator values safely — format to 2-4 dp where numeric
    def _fmt(val, dp=2):
        try:
            return f"{float(val):.{dp}f}"
        except (TypeError, ValueError):
            return "N/A"

    rsi        = _fmt(latest_row.get("RSI_14"), 1)
    macd       = _fmt(latest_row.get("MACD"), 4)
    ret        = latest_row.get("Return", 0)
    ret_pct    = _fmt(float(ret) * 100 if ret is not None else 0, 2)
    sma5       = _fmt(latest_row.get("SMA_5"))
    sma10      = _fmt(latest_row.get("SMA_10"))
    close      = _fmt(latest_row.get("Close"))
    volatility = _fmt(latest_row.get("Volatility_10"), 4)
    volume_chg = latest_row.get("Volume_change")
    vol_pct    = _fmt(float(volume_chg) * 100 if volume_chg is not None else 0, 1)

    return (
        f"A machine learning model predicts that {stock_name} stock will go {action} "
        f"tomorrow with {probability:.0%} confidence.\n\n"
        f"Here are today's technical indicators for {stock_name}:\n"
        f"- Closing price: ${close}\n"
        f"- Daily return today: {ret_pct}%\n"
        f"- 5-day SMA: {sma5} | 10-day SMA: {sma10}\n"
        f"- RSI (14): {rsi}\n"
        f"- MACD: {macd}\n"
        f"- 10-day volatility: {volatility}\n"
        f"- Volume change vs yesterday: {vol_pct}%\n\n"
        f"In 3-4 sentences, explain what these indicators suggest and why the model "
        f"might be predicting the price will go {action} tomorrow. "
        f"Use simple language suitable for a beginner investor. "
        f"Be honest if the signal is weak or mixed. "
        f"End with a one-sentence reminder that this is not financial advice."
    )


def backtest_prompt(
    ticker: str,
    total_return: float,
    bh_return: float,
    sharpe: float,
    max_dd: float,
    num_trades: int,
    win_rate: float,
) -> str:
    """
    Builds a prompt asking the LLM to summarise backtest results.

    Args:
        ticker:       e.g. "NVDA"
        total_return: strategy total return as a percentage, e.g. 12.5
        bh_return:    buy-and-hold return as a percentage, e.g. 18.2
        sharpe:       annualised Sharpe ratio
        max_dd:       max drawdown as a negative fraction, e.g. -0.18
        num_trades:   total number of trades executed
        win_rate:     percentage of winning trades, e.g. 55.0
    """
    alpha = total_return - bh_return
    outperform = "outperformed" if alpha >= 0 else "underperformed"

    return (
        f"A machine learning trading strategy for {ticker} produced the following "
        f"backtest results over the test period:\n\n"
        f"- Strategy return:       {total_return:+.2f}%\n"
        f"- Buy & Hold return:     {bh_return:+.2f}%\n"
        f"- Alpha vs market:       {alpha:+.2f}% (strategy {outperform} buy-and-hold)\n"
        f"- Sharpe ratio:          {sharpe}\n"
        f"- Max drawdown:          {max_dd:.1%}\n"
        f"- Number of trades:      {num_trades}\n"
        f"- Win rate:              {win_rate:.0f}%\n\n"
        f"In 4-5 sentences, summarise how the strategy performed, whether the results "
        f"are encouraging or concerning, and what the main risks are based on the "
        f"drawdown and Sharpe ratio. Be balanced and honest — if the strategy "
        f"underperformed buy-and-hold, say so clearly. "
        f"Use plain language a beginner investor would understand."
    )