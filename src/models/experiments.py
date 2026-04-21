"""
src/models/experiments.py
Runs multi-model, multi-feature-set experiments through the walk-forward engine.
Updated to include FEATURE_SETS_V2 from the enhanced technical_indicators module.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

from src.models.walk_forward import run_walk_forward
from src.features.technical_indicators import FEATURE_SETS_V2   # ← NEW

MODEL_REGISTRY = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    "SVM":                 SVC(probability=True, kernel='rbf'),
    "Baseline (Dummy)":    DummyClassifier(strategy="most_frequent"),
}

# Original feature sets kept for apples-to-apples comparison
FEATURE_SETS_V1 = {
    "Baseline":    ['SMA_5', 'SMA_10', 'Close_lag_1', 'Close_lag_2', 'Return'],
    "With RSI":    ['SMA_5', 'SMA_10', 'RSI_14', 'Return', 'Close_lag_1'],
    "With Volume": ['SMA_5', 'SMA_10', 'Return', 'Volume_change', 'Volatility_10'],
    "Full_v1":     ['SMA_5', 'SMA_10', 'RSI_14', 'MACD', 'Volume_change',
                    'Volatility_10', 'Return', 'Close_lag_1', 'Close_lag_2'],
}

# Use V2 for new runs (or combine both for a full comparison grid)
FEATURE_SETS = FEATURE_SETS_V2


def run_experiments(
    df_raw: pd.DataFrame,
    ticker: str,
    cutoff_date: str,
    starting_cash: float = 100_000.0,
    confidence_threshold: float = 0.55,
    position_size_pct: float = 0.95,
    feature_sets: dict = None,   # ← pass FEATURE_SETS_V1, FEATURE_SETS_V2, or a merged dict
) -> pd.DataFrame:
    """
    Runs every model x feature set combination and returns a results dataframe.

    To compare old vs new features in one call:
        run_experiments(..., feature_sets={**FEATURE_SETS_V1, **FEATURE_SETS_V2})
    """
    import copy
    fsets = feature_sets or FEATURE_SETS
    rows = []

    for model_name, model_template in MODEL_REGISTRY.items():
        for feat_name, feat_list in fsets.items():
            try:
                result = run_walk_forward(
                    df_raw=df_raw,
                    ticker=ticker,
                    cutoff_date=cutoff_date,
                    starting_cash=starting_cash,
                    confidence_threshold=confidence_threshold,
                    position_size_pct=position_size_pct,
                    model=copy.deepcopy(model_template),
                    feature_set=feat_list,
                )
                m = result["metrics"]
                rows.append({
                    "Model":         model_name,
                    "Features":      feat_name,
                    "Return %":      m["total_return_pct"],
                    "B&H Return %":  m["bh_return_pct"],
                    "Alpha":         m["alpha"],
                    "Sharpe":        m["sharpe_ratio"],
                    "B&H Sharpe":    m["bh_sharpe"],
                    "Max Drawdown":  m["max_drawdown"],
                    "Win Rate %":    m["win_rate_pct"],
                    "# Trades":      m["num_trades"],
                    "Train AUC":     m["train_roc_auc"],
                    "Test Days":     m["test_days"],
                })
            except Exception as e:
                print(f"[{model_name} | {feat_name}] failed: {e}")

    return pd.DataFrame(rows)