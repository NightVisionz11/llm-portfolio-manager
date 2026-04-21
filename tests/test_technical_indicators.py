"""
tests/test_technical_indicators.py

Run with:  pytest tests/
"""

import pytest
import pandas as pd
import numpy as np
from src.features.technical_indicators import add_technical_indicators


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_df(n=60, start_price=100.0):
    """Create a minimal OHLCV dataframe with n rows."""
    dates  = pd.date_range("2023-01-01", periods=n, freq="B")
    closes = [start_price + i * 0.5 for i in range(n)]  # steady uptrend
    return pd.DataFrame({
        "Date":   dates,
        "Open":   closes,
        "High":   [c + 1 for c in closes],
        "Low":    [c - 1 for c in closes],
        "Close":  closes,
        "Volume": [1_000_000] * n,
    })


# ── Column presence ───────────────────────────────────────────────────────────

def test_expected_columns_present():
    df = add_technical_indicators(make_df())
    expected = [
        "SMA_5", "SMA_10", "Close_lag_1", "Close_lag_2",
        "Return", "Target", "RSI_14", "MACD",
        "Volume_change", "Volatility_10",
    ]
    for col in expected:
        assert col in df.columns, f"Missing column: {col}"


# ── No NaNs after processing ──────────────────────────────────────────────────

def test_no_nans_in_output():
    df = add_technical_indicators(make_df(n=60))
    assert not df.isnull().any().any(), "Output contains NaN values"


# ── SMA values are correct ────────────────────────────────────────────────────

def test_sma5_is_rolling_mean():
    raw = make_df(n=60)
    df  = add_technical_indicators(raw)
    # SMA_5 on the last row should equal mean of last 5 closes in the processed df
    expected_sma5 = df["Close"].iloc[-5:].mean()
    assert abs(df["SMA_5"].iloc[-1] - expected_sma5) < 1e-6


def test_sma10_greater_than_zero():
    df = add_technical_indicators(make_df())
    assert (df["SMA_10"] > 0).all()


# ── Lag features ──────────────────────────────────────────────────────────────

def test_close_lag1_shifted_correctly():
    df = add_technical_indicators(make_df(n=60))
    # lag_1 at row i should equal Close at row i-1
    for i in range(1, min(5, len(df))):
        assert df["Close_lag_1"].iloc[i] == pytest.approx(df["Close"].iloc[i - 1])


# ── Return ────────────────────────────────────────────────────────────────────

def test_return_is_pct_change():
    df = add_technical_indicators(make_df(n=60))
    for i in range(1, min(5, len(df))):
        expected = (df["Close"].iloc[i] - df["Close"].iloc[i - 1]) / df["Close"].iloc[i - 1]
        assert df["Return"].iloc[i] == pytest.approx(expected, rel=1e-5)


# ── Target ────────────────────────────────────────────────────────────────────

def test_target_is_binary():
    df = add_technical_indicators(make_df())
    assert set(df["Target"].unique()).issubset({0, 1})


def test_target_is_1_when_next_day_higher():
    """In a steady uptrend every target should be 1 (next close > current close)."""
    df = add_technical_indicators(make_df(n=60))
    # All targets should be 1 in an uptrend (last row excluded — no next day)
    assert (df["Target"] == 1).all()


# ── RSI bounds ────────────────────────────────────────────────────────────────

def test_rsi_between_0_and_100():
    df = add_technical_indicators(make_df())
    assert (df["RSI_14"] >= 0).all() and (df["RSI_14"] <= 100).all()


# ── Sufficient rows required ──────────────────────────────────────────────────

def test_short_df_returns_empty_or_small():
    """With only 5 rows there won't be enough data for a 14-period RSI."""
    df = add_technical_indicators(make_df(n=5))
    assert isinstance(df, pd.DataFrame)  # should not crash — may just be empty


# ── MultiIndex columns are flattened ─────────────────────────────────────────

def test_multiindex_columns_are_flattened():
    raw = make_df(n=60)
    raw.columns = pd.MultiIndex.from_tuples([(c, "") for c in raw.columns])
    df = add_technical_indicators(raw)
    assert not isinstance(df.columns, pd.MultiIndex)


# ── Volume is coerced to numeric ──────────────────────────────────────────────

def test_volume_as_string_is_coerced():
    raw = make_df(n=60)
    raw["Volume"] = raw["Volume"].astype(str)  # simulate string volume column
    df = add_technical_indicators(raw)
    assert pd.api.types.is_numeric_dtype(df["Volume"])
