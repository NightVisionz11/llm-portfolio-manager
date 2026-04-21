"""
src/features/technical_indicators.py

Enhanced feature engineering for ML-based trading.

Feature categories:
  1. Trend         – momentum over multiple horizons
  2. Mean-reversion – distance from moving averages, Bollinger Bands
  3. Volatility    – ATR, realised vol, vol regime
  4. Volume        – OBV, VWAP deviation, volume spikes
  5. Regime        – 200-day MA slope, ADX, bear/bull flag
  6. Interaction   – volatility-adjusted RSI, volume-confirmed momentum

Drop-in replacement: add_technical_indicators(df) still returns a df
with a 'Target' column and no NaN rows (same contract as before).
"""

import numpy as np
import pandas as pd


# ── helpers ────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl  = df['High'] - df['Low']
    hpc = (df['High'] - df['Close'].shift(1)).abs()
    lpc = (df['Low']  - df['Close'].shift(1)).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df['Close'].diff()).fillna(0)
    return (direction * df['Volume']).cumsum()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index – measures trend strength (0-100)."""
    up   = df['High'].diff()
    down = -df['Low'].diff()
    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    atr_s      = _atr(df, period)
    plus_di    = 100 * pd.Series(plus_dm,  index=df.index).ewm(com=period-1, min_periods=period).mean() / atr_s
    minus_di   = 100 * pd.Series(minus_dm, index=df.index).ewm(com=period-1, min_periods=period).mean() / atr_s
    dx         = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(com=period-1, min_periods=period).mean()


# ── main function ───────────────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a DataFrame with at minimum: Date, Open, High, Low, Close, Volume.
    Returns the same DataFrame with new feature columns and a binary 'Target'
    column (1 = next-day close higher than today's close).
    Rows with NaN values are dropped.
    """
    df = df.copy().sort_values('Date').reset_index(drop=True)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Close', 'Volume'])  

    c = df['Close']
    v = df['Volume']

    # ── 1. Trend / Momentum ────────────────────────────────────────────────

    # Simple moving averages (short-term)
    df['SMA_5']  = c.rolling(5).mean()
    df['SMA_10'] = c.rolling(10).mean()
    df['SMA_20'] = c.rolling(20).mean()
    df['SMA_50'] = c.rolling(50).mean()

    # Price position relative to SMAs (normalised, scale-invariant)
    df['Price_vs_SMA20']  = (c - df['SMA_20']) / df['SMA_20']
    df['Price_vs_SMA50']  = (c - df['SMA_50']) / df['SMA_50']

    # SMA crossover slopes (positive = short > long, gaining)
    df['SMA_cross_5_20']  = (df['SMA_5']  - df['SMA_20']) / df['SMA_20']
    df['SMA_cross_10_50'] = (df['SMA_10'] - df['SMA_50']) / df['SMA_50']

    # EMA-based MACD
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    df['MACD']        = ema12 - ema26
    df['MACD_signal'] = _ema(df['MACD'], 9)
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']   # ← crossover proxy

    # Multi-horizon returns (log returns are more ML-friendly than pct)
    for n in [1, 2, 3, 5, 10]:
        df[f'LogReturn_{n}d'] = np.log(c / c.shift(n))

    # Alias kept for backward compat with existing FEATURE_SETS
    df['Return'] = df['LogReturn_1d']

    # Lagged closes (normalised to avoid scale issues)
    df['Close_lag_1'] = c.shift(1)
    df['Close_lag_2'] = c.shift(2)

    # Rate of change (%)
    df['ROC_5']  = c.pct_change(5)
    df['ROC_10'] = c.pct_change(10)

    # ── 2. Mean-reversion (Bollinger Bands) ────────────────────────────────

    bb_std = c.rolling(20).std()
    df['BB_upper'] = df['SMA_20'] + 2 * bb_std
    df['BB_lower'] = df['SMA_20'] - 2 * bb_std
    bb_range = (df['BB_upper'] - df['BB_lower']).replace(0, np.nan)
    df['BB_pct']   = (c - df['BB_lower']) / bb_range   # 0 = at lower band, 1 = upper
    df['BB_width'] = bb_range / df['SMA_20']           # bandwidth = vol proxy

    # ── 3. Volatility ──────────────────────────────────────────────────────

    df['ATR_14']       = _atr(df, 14)
    df['ATR_pct']      = df['ATR_14'] / c                  # normalised ATR

    # Realised volatility (rolling std of log-returns)
    df['Volatility_5']  = df['LogReturn_1d'].rolling(5).std()
    df['Volatility_10'] = df['LogReturn_1d'].rolling(10).std()
    df['Volatility_20'] = df['LogReturn_1d'].rolling(20).std()

    # Volatility regime: is current vol above its 60-day median? (0/1)
    vol_med = df['Volatility_20'].rolling(60).median()
    df['High_Vol_Regime'] = (df['Volatility_20'] > vol_med).astype(int)

    # Vol ratio: short-term vol vs. longer-term vol
    df['Vol_ratio'] = df['Volatility_5'] / df['Volatility_20'].replace(0, np.nan)

    # ── 4. Volume ──────────────────────────────────────────────────────────

    df['Volume_change']   = v.pct_change()
    df['Volume_SMA20']    = v.rolling(20).mean()
    df['Volume_ratio']    = v / df['Volume_SMA20'].replace(0, np.nan)  # >1 = high volume day

    # On-Balance Volume (trend confirmation)
    df['OBV']           = _obv(df)
    df['OBV_slope']     = df['OBV'].diff(5) / df['OBV'].abs().rolling(5).mean().replace(0, np.nan)

    # VWAP deviation (intraday proxy using daily OHLCV)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap_5 = (typical_price * v).rolling(5).sum() / v.rolling(5).sum()
    df['VWAP_dev'] = (c - vwap_5) / vwap_5

    # ── 5. RSI ─────────────────────────────────────────────────────────────

    df['RSI_14'] = _rsi(c, 14)
    df['RSI_7']  = _rsi(c, 7)

    # RSI extremes as binary signals
    df['RSI_oversold']   = (df['RSI_14'] < 30).astype(int)
    df['RSI_overbought'] = (df['RSI_14'] > 70).astype(int)

    # ── 6. Trend-strength regime (ADX) ────────────────────────────────────

    df['ADX_14'] = _adx(df, 14)
    # ADX > 25 = trending market; < 25 = ranging/choppy
    df['Trending_Market'] = (df['ADX_14'] > 25).astype(int)

    # Long-term trend direction (200-day MA slope as %)
    sma200 = c.rolling(200).mean()
    df['Trend_200d'] = sma200.pct_change(20)   # 20-day slope of 200-MA
    df['Bull_Market'] = (c > sma200).astype(int)

    # ── 7. Interaction features ────────────────────────────────────────────

    # Volatility-adjusted momentum: strong move relative to current vol
    df['Adj_momentum'] = df['LogReturn_5d'] / df['Volatility_10'].replace(0, np.nan)

    # Volume-confirmed trend: momentum × volume spike
    df['Vol_confirmed_return'] = df['LogReturn_1d'] * np.log1p(df['Volume_ratio'].clip(lower=0))

    # RSI in context of vol regime (oversold in high-vol = stronger signal)
    df['RSI_vol_adj'] = df['RSI_14'] * (1 + df['High_Vol_Regime'] * 0.1)

    # ── 8. Target ──────────────────────────────────────────────────────────
    # 1 if tomorrow's close > today's close, else 0
    df['Target'] = (c.shift(-1) > c).astype(int)

    # Drop NaN rows (from rolling windows, lags, and the last target row)
    df = df.dropna().reset_index(drop=True)

    return df


# ── Curated feature sets for experiments.py ────────────────────────────────

FEATURE_SETS_V2 = {
    # Original sets kept for comparison
    "Baseline":    ['SMA_5', 'SMA_10', 'Close_lag_1', 'Close_lag_2', 'Return'],
    "With RSI":    ['SMA_5', 'SMA_10', 'RSI_14', 'Return', 'Close_lag_1'],
    "With Volume": ['SMA_5', 'SMA_10', 'Return', 'Volume_change', 'Volatility_10'],
    "Full_v1":     ['SMA_5', 'SMA_10', 'RSI_14', 'MACD', 'Volume_change',
                    'Volatility_10', 'Return', 'Close_lag_1', 'Close_lag_2'],

    # New sets
    "Mean_Reversion": [
        'BB_pct', 'BB_width', 'RSI_14', 'Price_vs_SMA20',
        'ATR_pct', 'Vol_ratio', 'LogReturn_1d',
    ],
    "Momentum": [
        'LogReturn_1d', 'LogReturn_3d', 'LogReturn_5d', 'ROC_10',
        'MACD_hist', 'SMA_cross_5_20', 'ADX_14', 'Trending_Market',
    ],
    "Regime_Aware": [
        'Bull_Market', 'Trending_Market', 'High_Vol_Regime',
        'RSI_14', 'MACD_hist', 'BB_pct',
        'Adj_momentum', 'OBV_slope',
    ],
    "Volume_Confirmed": [
        'Volume_ratio', 'OBV_slope', 'VWAP_dev',
        'Vol_confirmed_return', 'LogReturn_1d',
        'RSI_14', 'BB_pct', 'ATR_pct',
    ],
    "Full_v2": [
        # Trend
        'SMA_cross_5_20', 'SMA_cross_10_50', 'MACD_hist', 'ROC_5', 'ROC_10',
        # Mean-reversion
        'BB_pct', 'BB_width', 'Price_vs_SMA20',
        # Volatility
        'ATR_pct', 'Volatility_10', 'Vol_ratio', 'High_Vol_Regime',
        # Volume
        'Volume_ratio', 'OBV_slope', 'VWAP_dev',
        # Oscillators
        'RSI_14', 'RSI_7',
        # Regime
        'Bull_Market', 'Trending_Market', 'ADX_14',
        # Interactions
        'Adj_momentum', 'Vol_confirmed_return',
        # Lags
        'LogReturn_1d', 'LogReturn_2d', 'LogReturn_3d',
    ],
}


if __name__ == "__main__":
    # Smoke test — generate random OHLCV and verify no NaNs in output
    np.random.seed(42)
    n = 500
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    test_df = pd.DataFrame({
        'Date':   pd.date_range('2020-01-01', periods=n, freq='B'),
        'Open':   closes * (1 - np.abs(np.random.randn(n) * 0.002)),
        'High':   closes * (1 + np.abs(np.random.randn(n) * 0.005)),
        'Low':    closes * (1 - np.abs(np.random.randn(n) * 0.005)),
        'Close':  closes,
        'Volume': np.random.randint(1_000_000, 5_000_000, n),
    })

    out = add_technical_indicators(test_df)
    print(f"Input rows: {n}  →  Output rows after dropna: {len(out)}")
    print(f"Features available: {len(out.columns)} columns")
    print("NaN check:", out.isnull().sum().sum(), "NaNs")
    print("\nSample columns:")
    sample_cols = ['BB_pct', 'ATR_pct', 'OBV_slope', 'Bull_Market',
                   'Adj_momentum', 'Vol_confirmed_return', 'Target']
    print(out[sample_cols].tail())