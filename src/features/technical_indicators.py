import pandas as pd
import numpy as np

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Flatten MultiIndex columns from yfinance if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Normalize yfinance column names
    rename_map = {}
    for col in df.columns:
        for base in ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']:
            if col.startswith(base):
                rename_map[col] = base
    df.rename(columns=rename_map, inplace=True)

    # Ensure numeric types
    df['Close']  = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    # --- Lagged features for safe prediction ---
    df['Close_lag_1'] = df['Close'].shift(1)
    df['Close_lag_2'] = df['Close'].shift(2)
    df['Return']      = df['Close'].pct_change().shift(1)  # yesterday's return

    # Rolling indicators (shifted so they only see past data)
    df['SMA_5']  = df['Close'].shift(1).rolling(5).mean()
    df['SMA_10'] = df['Close'].shift(1).rolling(10).mean()

    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean().shift(1)
    loss  = -delta.clip(upper=0).rolling(14).mean().shift(1)
    rs    = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    df['MACD'] = (df['Close'].shift(1).ewm(span=12).mean() -
                  df['Close'].shift(1).ewm(span=26).mean())

    df['Volume_change'] = df['Volume'].pct_change().shift(1)
    df['Volatility_10'] = df['Return'].rolling(10).std().shift(1)

    # Target: 1 if next-day Close higher than today
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Drop all rows with NaN (first few rows due to rolling/lag)
    df = df.dropna().reset_index(drop=True)

    return df