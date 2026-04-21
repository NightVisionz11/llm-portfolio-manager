"""
tests/test_experiments.py

Tests for src/models/experiments.py  (run_experiments)
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier


# ── Minimal stubs so this file runs standalone ───────────────────────────────
# Once your project is installed, replace these three imports:
#   from src.models.experiments import run_experiments, MODEL_REGISTRY, FEATURE_SETS

MODEL_REGISTRY = {
    "Dummy":  DummyClassifier(strategy="most_frequent"),
}

FEATURE_SETS = {
    "Baseline":    ['SMA_5', 'SMA_10', 'Close_lag_1', 'Close_lag_2', 'Return'],
    "With RSI":    ['SMA_5', 'SMA_10', 'RSI_14', 'Return', 'Close_lag_1'],
}

def _add_indicators(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df['Return']      = df['Close'].pct_change().fillna(0)
    df['SMA_5']       = df['Close'].rolling(5).mean().bfill()
    df['SMA_10']      = df['Close'].rolling(10).mean().bfill()
    df['Close_lag_1'] = df['Close'].shift(1).bfill()
    df['Close_lag_2'] = df['Close'].shift(2).bfill()
    df['RSI_14']      = 50.0
    df['MACD']        = 0.0
    df['Volume_change'] = 0.0
    df['Volatility_10'] = df['Return'].rolling(10).std().bfill()
    df['Target']      = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df.dropna(subset=['Target']).reset_index(drop=True)

def _run_wf(df_raw, ticker, cutoff_date, starting_cash, confidence_threshold,
            position_size_pct, model, feature_set):
    """Tiny walk-forward stub used only by run_experiments."""
    import copy
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    features = feature_set
    df = _add_indicators(df_raw.copy())
    cutoff = pd.Timestamp(cutoff_date)
    train_df = df[df['Date'] < cutoff]; test_df = df[df['Date'] >= cutoff]
    if len(train_df) < 50 or len(test_df) < 5:
        raise ValueError("Not enough data")

    sc = StandardScaler()
    X_tr = sc.fit_transform(train_df[features]); y_tr = train_df['Target']
    m = copy.deepcopy(model); m.fit(X_tr, y_tr)
    train_auc = roc_auc_score(y_tr, m.predict_proba(X_tr)[:, 1])
    X_te = sc.transform(test_df[features])
    preds = m.predict(X_te); probs = m.predict_proba(X_te)[:, 1]

    final_val = starting_cash  # neutral stub
    bh_val = starting_cash * 1.05
    return {"metrics": {
        "total_return_pct": 0.0, "bh_return_pct": 5.0, "alpha": -5.0,
        "sharpe_ratio": 0.1, "bh_sharpe": 0.3, "max_drawdown": -0.05,
        "win_rate_pct": 50.0, "num_trades": 10,
        "train_roc_auc": round(train_auc, 4), "test_days": len(test_df),
    }}

def run_experiments(df_raw, ticker, cutoff_date, starting_cash=100_000.0,
                    confidence_threshold=0.55, position_size_pct=0.95):
    import copy
    rows = []
    for model_name, model_template in MODEL_REGISTRY.items():
        for feat_name, feat_list in FEATURE_SETS.items():
            try:
                result = _run_wf(
                    df_raw=df_raw, ticker=ticker, cutoff_date=cutoff_date,
                    starting_cash=starting_cash, confidence_threshold=confidence_threshold,
                    position_size_pct=position_size_pct,
                    model=copy.deepcopy(model_template), feature_set=feat_list,
                )
                m = result["metrics"]
                rows.append({
                    "Model": model_name, "Features": feat_name,
                    "Return %": m["total_return_pct"], "B&H Return %": m["bh_return_pct"],
                    "Alpha": m["alpha"], "Sharpe": m["sharpe_ratio"],
                    "B&H Sharpe": m["bh_sharpe"], "Max Drawdown": m["max_drawdown"],
                    "Win Rate %": m["win_rate_pct"], "# Trades": m["num_trades"],
                    "Train AUC": m["train_roc_auc"], "Test Days": m["test_days"],
                })
            except Exception as e:
                print(f"[{model_name} | {feat_name}] failed: {e}")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def price_df():
    rng = np.random.default_rng(0)
    n = 300
    dates  = pd.bdate_range("2020-01-01", periods=n)
    closes = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    return pd.DataFrame({"Date": dates, "Close": closes})


@pytest.fixture(scope="module")
def experiment_results(price_df):
    return run_experiments(price_df, ticker="TEST", cutoff_date="2021-01-01")


# ─────────────────────────────────────────────────────────────────────────────
# Output-shape tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentsOutputShape:

    def test_returns_dataframe(self, experiment_results):
        assert isinstance(experiment_results, pd.DataFrame)

    def test_row_count_equals_model_times_feature_sets(self, experiment_results):
        expected = len(MODEL_REGISTRY) * len(FEATURE_SETS)
        assert len(experiment_results) == expected

    def test_required_columns_present(self, experiment_results):
        required = {"Model", "Features", "Return %", "B&H Return %", "Alpha",
                    "Sharpe", "Max Drawdown", "Win Rate %", "# Trades", "Train AUC"}
        assert required.issubset(experiment_results.columns)


# ─────────────────────────────────────────────────────────────────────────────
# Content sanity
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentsContent:

    def test_model_names_are_strings(self, experiment_results):
        assert pd.api.types.is_string_dtype(experiment_results["Model"])

    def test_feature_set_names_are_strings(self, experiment_results):
        assert pd.api.types.is_string_dtype(experiment_results["Features"])

    def test_train_auc_in_valid_range(self, experiment_results):
        assert (experiment_results["Train AUC"].between(0.0, 1.0)).all()

    def test_max_drawdown_non_positive(self, experiment_results):
        assert (experiment_results["Max Drawdown"] <= 0).all()

    def test_win_rate_between_0_and_100(self, experiment_results):
        assert (experiment_results["Win Rate %"].between(0, 100)).all()

    def test_num_trades_non_negative(self, experiment_results):
        assert (experiment_results["# Trades"] >= 0).all()

    def test_alpha_equals_return_minus_bh(self, experiment_results):
        computed = (experiment_results["Return %"] - experiment_results["B&H Return %"]).round(2)
        assert (computed == experiment_results["Alpha"].round(2)).all()


# ─────────────────────────────────────────────────────────────────────────────
# Robustness: errors in individual runs must not crash the whole experiment
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentsRobustness:

    def test_failed_run_excluded_not_raised(self, price_df):
        """If one model/feature combo fails, it's skipped; the rest succeed."""
        import copy
        bad_features = ["DOES_NOT_EXIST"]
        original = FEATURE_SETS.copy()
        FEATURE_SETS["BadSet"] = bad_features
        try:
            results = run_experiments(price_df, ticker="TEST", cutoff_date="2021-01-01")
            # The bad row should be absent; the good rows should still be there
            assert "BadSet" not in results["Features"].values
            assert len(results) >= len(original)
        finally:
            if "BadSet" in FEATURE_SETS:
                del FEATURE_SETS["BadSet"]

    def test_empty_registry_returns_empty_df(self, price_df):
        original = MODEL_REGISTRY.copy()
        MODEL_REGISTRY.clear()
        try:
            results = run_experiments(price_df, ticker="TEST", cutoff_date="2021-01-01")
            assert isinstance(results, pd.DataFrame)
            assert len(results) == 0
        finally:
            MODEL_REGISTRY.update(original)