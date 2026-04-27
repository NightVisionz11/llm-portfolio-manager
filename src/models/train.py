import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib

from src.features.technical_indicators import add_technical_indicators
from src.data.loader import load_stock_csv
from src.utils.config import MODEL_DIR


def train_logreg_model(csv_file: str, ticker: str = None):
    """
    Train a logistic regression model with robust checks.
    """
    try:
        # Load data with our improved loader (this will raise friendly errors)
        df = load_stock_csv(csv_file)
        
        # Extra safety check in case something slips through
        if df.empty or len(df) < 50:   # Need enough data for training + indicators
            raise ValueError(
                f"❌ Not enough data for training '{ticker or csv_file}'. "
                f"Only {len(df)} rows found. Need at least 50 rows."
            )

        df = add_technical_indicators(df)

        features = ['SMA_5', 'SMA_10', 'Close_lag_1', 'Close_lag_2', 'Return']
        target = 'Target'

        # Check if features exist and we have data after adding indicators
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"❌ Missing features after adding indicators: {missing_features}")

        X = df[features]
        y = df[target]

        if len(X) == 0:
            raise ValueError("❌ No rows left after feature engineering.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_scaled, y, test_size=0.1, shuffle=False
        )

        # Extra check after split
        if len(X_train) == 0:
            raise ValueError("❌ Not enough data for training split.")

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        valid_auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])

        print(f"✅ [{ticker or csv_file}] Train ROC-AUC: {train_auc:.3f} | Valid ROC-AUC: {valid_auc:.3f}")

        prefix = ticker if ticker else "logreg"
        joblib.dump(model, f"{MODEL_DIR}/{prefix}_logreg_model.pkl")
        joblib.dump(scaler, f"{MODEL_DIR}/{prefix}_scaler.pkl")
        print(f"✅ Model saved as {prefix}_logreg_model.pkl")

    except Exception as e:
        # This catches both our friendly errors and any remaining issues
        print(f"❌ Training failed for {ticker or csv_file}:")
        print(f"   {e}")
        # Optionally re-raise if you want Streamlit to show it as error
        # raise

if __name__ == "__main__":
    train_logreg_model("Tesla.csv", ticker="TSLA")