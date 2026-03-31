import joblib
import pandas as pd
from src.features.technical_indicators import add_technical_indicators
from src.data.loader import load_stock_csv
from src.utils.config import MODEL_DIR

def predict_next_day(csv_file: str):
    model = joblib.load(f"{MODEL_DIR}/logreg_model.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")

    df = load_stock_csv(csv_file)
    df = add_technical_indicators(df)

    features = ['SMA_5', 'SMA_10', 'Close_lag_1', 'Close_lag_2', 'Return']
    X = df[features]
    X_scaled = scaler.transform(X)

    df['Prediction'] = model.predict(X_scaled)
    df['Pred_Prob'] = model.predict_proba(X_scaled)[:,1]

    print(df[['Date', 'Close', 'Prediction', 'Pred_Prob']].tail())

if __name__ == "__main__":
    predict_next_day("Tesla.csv")