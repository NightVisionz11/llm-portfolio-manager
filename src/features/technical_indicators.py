import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Flatten MultiIndex columns from yfinance if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Normalise columns that yfinance appends ticker to e.g. 'Close_TSLA'
    rename_map = {}
    for col in df.columns:
        for base in ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']:
            if col.startswith(base):
                rename_map[col] = base
    df.rename(columns=rename_map, inplace=True)

    # Ensure numeric types before any calculations
    df['Close']  = pd.to_numeric(df['Close'],  errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')  # FIX: was missing, caused crash

    # Simple moving averages
    df['SMA_5']  = df['Close'].rolling(5).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()

    # Lag features
    df['Close_lag_1'] = df['Close'].shift(1)
    df['Close_lag_2'] = df['Close'].shift(2)

    # Daily return
    df['Return'] = df['Close'].pct_change()

    # Target: 1 if next-day Close higher, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # RSI — must come before dropna so rolling has full history to work with
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = -delta.clip(upper=0).rolling(14).mean()
    rs    = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()

    # Volume change
    df['Volume_change'] = df['Volume'].pct_change()

    # Volatility
    df['Volatility_10'] = df['Return'].rolling(10).std()

    # Single dropna at the end after all indicators are calculated
    df = df.dropna().reset_index(drop=True)

    return df