import pandas as pd
import re
import os
from src.utils.config import RAW_DATA_DIR


def load_stock_csv(filename: str) -> pd.DataFrame:
    """
    Load a stock CSV file with robust validation and friendly error messages.
    """
    # 1. Basic input validation
    if not filename or not isinstance(filename, str):
        raise ValueError("filename must be a non-empty string.")

    filename = filename.strip()
    if not filename:
        raise ValueError("filename cannot be empty.")

    # 2. Extract ticker and validate format
    base_name = os.path.splitext(os.path.basename(filename))[0]
    ticker = base_name.strip().upper()

    if not re.match(r'^[A-Z0-9.\-]{1,10}$', ticker):
        raise ValueError(
            f"Invalid ticker symbol: '{ticker}'.\n"
            "Ticker must be 1-10 characters using only uppercase letters, numbers, dots (.), or hyphens (-).\n"
            "Examples: NVDA, TSLA, BRK.B, AAPL"
        )

    # 3. Ensure .csv extension
    if not filename.lower().endswith('.csv'):
        filename = f"{filename}.csv"

    path = f"{RAW_DATA_DIR}/{filename}"

    # 4. Load with friendly error handling
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        # This is the main error you were seeing — now it's friendly
        available_files = []
        if os.path.exists(RAW_DATA_DIR):
            available_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]
        
        msg = f"❌ Stock data file not found for ticker '{ticker}'.\n"
        msg += f"   Looked for: {path}\n\n"
        msg += "💡 Make sure the CSV file exists in your RAW_DATA_DIR folder.\n"
        
        if available_files:
            msg += f"\nAvailable files:\n" + "\n".join(f"   - {f}" for f in sorted(available_files)[:15])
            if len(available_files) > 15:
                msg += f"\n   ... and {len(available_files)-15} more"
        else:
            msg += f"\n   (No .csv files found in {RAW_DATA_DIR})"
        
        raise FileNotFoundError(msg) from None   # Clean error, no long traceback

    except pd.errors.EmptyDataError:
        raise ValueError(f"❌ The file '{filename}' is empty.") from None
    except Exception as e:
        # Catch any other pandas/IO errors and make them readable
        raise ValueError(f"❌ Failed to load '{filename}': {e}") from None

    # 5. Validate the DataFrame content
    if df.empty:
        raise ValueError(f"❌ Loaded CSV for '{ticker}' is empty.")

    required_columns = ['Date', 'Close']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"❌ CSV for '{ticker}' is missing required columns: {missing}.\n"
            f"   Found columns: {list(df.columns)}"
        )

    # 6. Convert date and sort
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception as e:
        raise ValueError(f"❌ Could not parse 'Date' column in {filename}: {e}") from None

    df = df.sort_values('Date').reset_index(drop=True)

    print(f"✅ Successfully loaded {ticker} — {len(df):,} rows "
          f"({df['Date'].min().date()} to {df['Date'].max().date()})")

    return df


if __name__ == "__main__":
    try:
        df = load_stock_csv("Tesla.csv")
        print(df.head())
    except Exception as e:
        print(e)          # This will now show your clean friendly message