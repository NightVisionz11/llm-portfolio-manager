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
    Train a logistic regression model for the given CSV file.
    Saves as {ticker}_logreg_model.pkl and {ticker}_scaler.pkl.
    Falls back to generic names if ticker not provided.
    """
    df = load_stock_csv(csv_file)
    df = add_technical_indicators(df)

    features = ['SMA_5', 'SMA_10', 'Close_lag_1', 'Close_lag_2', 'Return']
    target = 'Target'

    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_scaled, y, test_size=0.1, shuffle=False
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    valid_auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])
    print(f"[{ticker or csv_file}] Train ROC-AUC: {train_auc:.3f} | Valid ROC-AUC: {valid_auc:.3f}")

    prefix = ticker if ticker else "logreg"
    joblib.dump(model, f"{MODEL_DIR}/{prefix}_logreg_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/{prefix}_scaler.pkl")
    print(f"Model saved as {prefix}_logreg_model.pkl")


if __name__ == "__main__":
    train_logreg_model("Tesla.csv", ticker="TSLA")