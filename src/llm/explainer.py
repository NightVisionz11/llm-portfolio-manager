"""
explainer.py

Generates human-readable explanations for model predictions.
No LLM API key required — uses rule-based templates.
When you're ready to add an Anthropic API key, swap in the
`explain_with_llm()` function below.
"""

import pandas as pd


def explain_prediction(
    stock_name: str,
    prediction: int,
    probability: float,
    latest_row: dict,
) -> str:
    """
    Returns a plain-English explanation based on the model output
    and the most recent feature values.

    Args:
        stock_name:  e.g. "Tesla"
        prediction:  0 (down) or 1 (up)
        probability: confidence probability for the predicted class
        latest_row:  dict of the last row's feature values
    """
    direction = "UP 📈" if prediction == 1 else "DOWN 📉"
    confidence = "high" if probability >= 0.65 else "moderate" if probability >= 0.55 else "low"

    sma5 = latest_row.get("SMA_5", None)
    sma10 = latest_row.get("SMA_10", None)
    ret = latest_row.get("Return", None)
    close = latest_row.get("Close", None)

    trend_comment = ""
    if sma5 is not None and sma10 is not None:
        if sma5 > sma10:
            trend_comment = "The short-term moving average is above the longer-term average, suggesting recent upward momentum."
        else:
            trend_comment = "The short-term moving average is below the longer-term average, suggesting recent downward pressure."

    return_comment = ""
    if ret is not None:
        pct = round(ret * 100, 2)
        if pct > 0:
            return_comment = f"The stock rose {pct}% today."
        elif pct < 0:
            return_comment = f"The stock fell {abs(pct)}% today."
        else:
            return_comment = "The stock was roughly flat today."

    explanation = (
        f"**{stock_name} Prediction: {direction}**\n\n"
        f"The model is {confidence} confidence ({probability:.0%}) that {stock_name} "
        f"will move {direction.split()[0].lower()} tomorrow.\n\n"
        f"{return_comment} {trend_comment}\n\n"
        f"*Note: This is a machine learning estimate based on historical price patterns, "
        f"not financial advice. Always do your own research.*"
    )
    return explanation


def explain_backtest(sharpe: float, max_dd: float, final_strat: float, final_market: float) -> str:
    """
    Summarises backtest performance in plain English.
    """
    outperform = "outperformed" if final_strat >= final_market else "underperformed"
    sr_comment = (
        "strong risk-adjusted returns" if sharpe > 1
        else "acceptable risk-adjusted returns" if sharpe > 0.5
        else "weak risk-adjusted returns"
    )
    dd_comment = (
        "relatively shallow drawdowns" if max_dd > -0.1
        else "moderate drawdowns" if max_dd > -0.25
        else "significant drawdowns — position sizing matters"
    )

    return (
        f"The strategy {outperform} buy-and-hold "
        f"({final_strat:.2f}x vs {final_market:.2f}x cumulative return).\n"
        f"Sharpe Ratio of {sharpe} indicates {sr_comment}.\n"
        f"Max drawdown of {max_dd:.1%} suggests {dd_comment}."
    )


# ── Optional: drop-in LLM version when you have an API key ──────────────────
#
# import anthropic
#
# def explain_with_llm(stock_name, prediction, probability, latest_row) -> str:
#     client = anthropic.Anthropic()
#     prompt = (
#         f"The ML model predicts {stock_name} will go "
#         f"{'up' if prediction == 1 else 'down'} tomorrow "
#         f"with {probability:.0%} confidence. "
#         f"Recent indicators: {latest_row}. "
#         f"Explain this prediction simply for a beginner investor in 3 sentences."
#     )
#     message = client.messages.create(
#         model="claude-sonnet-4-20250514",
#         max_tokens=256,
#         messages=[{"role": "user", "content": prompt}],
#     )
#     return message.content[0].text