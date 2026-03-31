import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(y_true, y_pred, y_prob) -> dict:
    """
    Returns a dict of key classification metrics.
    """
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def backtest_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple backtest: go long when Prediction == 1, stay flat otherwise.
    Compares cumulative strategy return vs. buy-and-hold.

    Expects df to have columns: ['Date', 'Close', 'Prediction']
    Returns df with added columns: ['Daily_Return', 'Strategy_Return',
                                     'Cumulative_Market', 'Cumulative_Strategy']
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["Daily_Return"] = df["Close"].pct_change().fillna(0)

    # Strategy: only take the market return on days we predicted UP
    df["Strategy_Return"] = df["Daily_Return"] * df["Prediction"].shift(1).fillna(0)

    df["Cumulative_Market"] = (1 + df["Daily_Return"]).cumprod()
    df["Cumulative_Strategy"] = (1 + df["Strategy_Return"]).cumprod()

    return df


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualised Sharpe ratio (252 trading days).
    """
    excess = returns - risk_free_rate / 252
    if excess.std() == 0:
        return 0.0
    return round((excess.mean() / excess.std()) * np.sqrt(252), 4)


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Maximum drawdown from peak as a negative fraction.
    """
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return round(drawdown.min(), 4)


if __name__ == "__main__":
    # Quick smoke-test with dummy data
    import joblib
    from src.data.loader import load_stock_csv
    from src.features.technical_indicators import add_technical_indicators
    from src.utils.config import MODEL_DIR

    df = load_stock_csv("Tesla.csv")
    df = add_technical_indicators(df)

    model = joblib.load(f"{MODEL_DIR}/logreg_model.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")

    features = ["SMA_5", "SMA_10", "Close_lag_1", "Close_lag_2", "Return"]
    X_scaled = scaler.transform(df[features])
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    metrics = evaluate_model(df["Target"], y_pred, y_prob)
    print("Metrics:", metrics)

    df["Prediction"] = y_pred
    bt = backtest_strategy(df)
    sr = sharpe_ratio(bt["Strategy_Return"])
    md = max_drawdown(bt["Cumulative_Strategy"])
    print(f"Sharpe Ratio: {sr}")
    print(f"Max Drawdown: {md}")
    print(bt[["Date", "Cumulative_Market", "Cumulative_Strategy"]].tail())