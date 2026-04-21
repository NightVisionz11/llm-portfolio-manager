"""
tests/test_evaluate.py

Tests for src/models/evaluate.py
"""

import pytest
import numpy as np
import pandas as pd

# ── Inline the functions under test so the file runs standalone ──────────────
# (Remove these imports and use the real ones once the project is wired up)
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
)


def evaluate_model(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy":         round(accuracy_score(y_true, y_pred), 4),
        "roc_auc":          round(roc_auc_score(y_true, y_prob), 4),
        "precision":        round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":           round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":               round(f1_score(y_true, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def backtest_strategy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["Daily_Return"]       = df["Close"].pct_change().fillna(0)
    df["Strategy_Return"]    = df["Daily_Return"] * df["Prediction"].shift(1).fillna(0)
    df["Cumulative_Market"]  = (1 + df["Daily_Return"]).cumprod()
    df["Cumulative_Strategy"] = (1 + df["Strategy_Return"]).cumprod()
    return df


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = returns - risk_free_rate / 252
    if excess.std() == 0:
        return 0.0
    return round((excess.mean() / excess.std()) * np.sqrt(252), 4)


def max_drawdown(cumulative_returns: pd.Series) -> float:
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return round(drawdown.min(), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_predictions():
    """y_pred perfectly matches y_true."""
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = y_true.copy()
    y_prob = y_true.astype(float)
    return y_true, y_pred, y_prob


@pytest.fixture
def random_predictions():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.uniform(0, 1, size=200)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


@pytest.fixture
def sample_backtest_df():
    """Small price series with an alternating prediction pattern."""
    dates  = pd.date_range("2024-01-01", periods=10, freq="B")
    closes = [100, 102, 101, 105, 107, 106, 110, 108, 112, 115]
    preds  = [1,   1,   0,   1,   1,   0,   1,   0,   1,   1]
    return pd.DataFrame({"Date": dates, "Close": closes, "Prediction": preds})


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_model
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateModel:

    def test_returns_required_keys(self, random_predictions):
        y_true, y_pred, y_prob = random_predictions
        result = evaluate_model(y_true, y_pred, y_prob)
        assert set(result.keys()) == {"accuracy", "roc_auc", "precision", "recall", "f1", "confusion_matrix"}

    def test_perfect_predictions_give_1s(self, perfect_predictions):
        y_true, y_pred, y_prob = perfect_predictions
        result = evaluate_model(y_true, y_pred, y_prob)
        assert result["accuracy"]  == 1.0
        assert result["roc_auc"]   == 1.0
        assert result["precision"] == 1.0
        assert result["recall"]    == 1.0
        assert result["f1"]        == 1.0

    def test_metrics_in_valid_range(self, random_predictions):
        y_true, y_pred, y_prob = random_predictions
        result = evaluate_model(y_true, y_pred, y_prob)
        for key in ("accuracy", "precision", "recall", "f1"):
            assert 0.0 <= result[key] <= 1.0, f"{key} out of range"
        assert 0.0 <= result["roc_auc"] <= 1.0

    def test_confusion_matrix_shape(self, random_predictions):
        y_true, y_pred, y_prob = random_predictions
        result = evaluate_model(y_true, y_pred, y_prob)
        cm = result["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2

    def test_confusion_matrix_sums_to_n(self, random_predictions):
        y_true, y_pred, y_prob = random_predictions
        result = evaluate_model(y_true, y_pred, y_prob)
        cm = result["confusion_matrix"]
        total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
        assert total == len(y_true)

    def test_all_zeros_prediction_no_crash(self):
        """precision/recall with zero_division=0 should not raise."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4])
        result = evaluate_model(y_true, y_pred, y_prob)
        assert result["precision"] == 0.0
        assert result["recall"]    == 0.0

    def test_values_are_rounded_to_4dp(self, random_predictions):
        y_true, y_pred, y_prob = random_predictions
        result = evaluate_model(y_true, y_pred, y_prob)
        for key in ("accuracy", "roc_auc", "precision", "recall", "f1"):
            val = result[key]
            assert val == round(val, 4)


# ─────────────────────────────────────────────────────────────────────────────
# backtest_strategy
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestStrategy:

    def test_output_columns_present(self, sample_backtest_df):
        result = backtest_strategy(sample_backtest_df)
        for col in ("Daily_Return", "Strategy_Return", "Cumulative_Market", "Cumulative_Strategy"):
            assert col in result.columns

    def test_cumulative_market_starts_near_1(self, sample_backtest_df):
        result = backtest_strategy(sample_backtest_df)
        # First daily return is 0 (fillna), so first cumulative value == 1.0
        assert result["Cumulative_Market"].iloc[0] == pytest.approx(1.0)

    def test_cumulative_strategy_starts_near_1(self, sample_backtest_df):
        result = backtest_strategy(sample_backtest_df)
        assert result["Cumulative_Strategy"].iloc[0] == pytest.approx(1.0)

    def test_flat_prediction_zero_strategy_return(self):
        """When all predictions are 0, strategy never takes a position."""
        dates  = pd.date_range("2024-01-01", periods=5, freq="B")
        closes = [100, 110, 120, 130, 140]
        preds  = [0, 0, 0, 0, 0]
        df = pd.DataFrame({"Date": dates, "Close": closes, "Prediction": preds})
        result = backtest_strategy(df)
        assert (result["Strategy_Return"] == 0).all()

    def test_all_buy_equals_market(self):
        """When predictions are always 1, strategy return ~ market return (1-day lag)."""
        dates  = pd.date_range("2024-01-01", periods=6, freq="B")
        closes = [100, 102, 104, 103, 105, 107]
        preds  = [1, 1, 1, 1, 1, 1]
        df = pd.DataFrame({"Date": dates, "Close": closes, "Prediction": preds})
        result = backtest_strategy(df)
        # Due to shift(1), from day 2 onward strategy return == daily return
        strategy = result["Strategy_Return"].iloc[2:].reset_index(drop=True).rename(None)
        market   = result["Daily_Return"].iloc[2:].reset_index(drop=True).rename(None)
        pd.testing.assert_series_equal(strategy, market)

    def test_row_count_unchanged(self, sample_backtest_df):
        result = backtest_strategy(sample_backtest_df)
        assert len(result) == len(sample_backtest_df)

    def test_unsorted_input_is_sorted(self):
        dates  = pd.date_range("2024-01-01", periods=5, freq="B")
        closes = [100, 110, 105, 115, 120]
        preds  = [1, 1, 0, 1, 1]
        df = pd.DataFrame({"Date": dates, "Close": closes, "Prediction": preds})
        shuffled = df.sample(frac=1, random_state=0)
        result = backtest_strategy(shuffled)
        assert list(result["Date"]) == sorted(result["Date"])


# ─────────────────────────────────────────────────────────────────────────────
# sharpe_ratio
# ─────────────────────────────────────────────────────────────────────────────

class TestSharpeRatio:

    def test_zero_std_returns_zero(self):
        returns = pd.Series([0.01] * 50)  # all identical → std of excess == 0
        # With identical returns and risk_free_rate=0, excess has zero std
        # Function should return 0.0 not raise
        result = sharpe_ratio(pd.Series([0.0] * 50))
        assert result == 0.0

    def test_positive_returns_positive_sharpe(self):
        returns = pd.Series([0.001] * 252)
        assert sharpe_ratio(returns) > 0

    def test_negative_returns_negative_sharpe(self):
        returns = pd.Series([-0.001] * 252)
        assert sharpe_ratio(returns) < 0

    def test_result_is_rounded_to_4dp(self):
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.normal(0.0005, 0.01, 252))
        result = sharpe_ratio(returns)
        assert result == round(result, 4)

    def test_annualisation_factor(self):
        """Manually verify the sqrt(252) annualisation."""
        returns = pd.Series([0.001] * 252)
        expected = round((returns.mean() / returns.std()) * np.sqrt(252), 4)
        assert sharpe_ratio(returns) == pytest.approx(expected, rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# max_drawdown
# ─────────────────────────────────────────────────────────────────────────────

class TestMaxDrawdown:

    def test_monotonically_increasing_zero_drawdown(self):
        cum = pd.Series([1.0, 1.05, 1.10, 1.15, 1.20])
        assert max_drawdown(cum) == 0.0

    def test_correct_drawdown_value(self):
        # Peak is 1.2 at index 2; trough is 0.9 at index 4 → dd = (0.9-1.2)/1.2 = -0.25
        cum = pd.Series([1.0, 1.1, 1.2, 1.0, 0.9])
        result = max_drawdown(cum)
        assert result == pytest.approx(-0.25, abs=1e-3)

    def test_result_is_non_positive(self):
        rng = np.random.default_rng(99)
        prices = (1 + rng.normal(0.0005, 0.01, 300)).cumprod()
        cum = pd.Series(prices)
        assert max_drawdown(cum) <= 0

    def test_result_rounded_to_4dp(self):
        cum = pd.Series([1.0, 1.1, 1.2, 1.0, 0.9])
        result = max_drawdown(cum)
        assert result == round(result, 4)

    def test_single_element(self):
        cum = pd.Series([1.0])
        assert max_drawdown(cum) == 0.0