# src/data/preprocessing.py
import pandas as pd

def remove_outliers(df: pd.DataFrame, col: str, z_thresh: float = 3.0) -> pd.DataFrame:
    mean, std = df[col].mean(), df[col].std()
    return df[(df[col] - mean).abs() < z_thresh * std]

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Volume_SMA_5'] = df['Volume'].rolling(5).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_SMA_5']
    return df