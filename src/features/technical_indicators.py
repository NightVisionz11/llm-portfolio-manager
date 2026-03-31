import pandas as pd

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Simple moving averages
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    # Lag features
    df['Close_lag_1'] = df['Close'].shift(1)
    df['Close_lag_2'] = df['Close'].shift(2)
    # Daily return
    df['Return'] = df['Close'].pct_change()
    # Target: 1 if next-day Close higher, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    # Drop rows with NaN
    df = df.dropna().reset_index(drop=True)
    return df