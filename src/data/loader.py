import pandas as pd
from src.utils.config import RAW_DATA_DIR

def load_stock_csv(filename: str):
    path = f"{RAW_DATA_DIR}/{filename}"
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

if __name__ == "__main__":
    df = load_stock_csv("Tesla.csv")
    print(df.head())