"""Tests for the compute_correlation tool."""

import numpy as np
import pandas as pd

from datachat.tools.compute_correlation import compute_correlation


def _df_with_known_correlations():
    """Build a DataFrame with predictable correlations."""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 10, n)
    return pd.DataFrame(
        {
            "x": x,
            "perfectly_linear": 2 * x + 1,           # Pearson ≈ 1
            "negatively_linear": -x + 5,             # Pearson ≈ -1
            "unrelated_noise": np.random.randn(n),   # Pearson ≈ 0
            "non_linear": x ** 2,                    # Strong Spearman, weaker Pearson
            "category": ["A", "B"] * 50,             # Non-numeric, should be skipped
        }
    )


def test_missing_target_column_returns_error():
    df = _df_with_known_correlations()
    result = compute_correlation(df, "no_such_column")
    assert "error" in result


def test_non_numeric_target_returns_error():
    df = _df_with_known_correlations()
    result = compute_correlation(df, "category")
    assert "error" in result


def test_strong_positive_correlation_detected():
    df = _df_with_known_correlations()
    result = compute_correlation(df, "x")
    # Find perfectly_linear in the results
    perfect = next(c for c in result["correlations"] if c["column"] == "perfectly_linear")
    assert perfect["pearson"] > 0.99


def test_strong_negative_correlation_detected():
    df = _df_with_known_correlations()
    result = compute_correlation(df, "x")
    negative = next(c for c in result["correlations"] if c["column"] == "negatively_linear")
    assert negative["pearson"] < -0.99


def test_results_sorted_by_strength():
    df = _df_with_known_correlations()
    result = compute_correlation(df, "x")
    strengths = [
        max(abs(c["pearson"]), abs(c["spearman"]))
        for c in result["correlations"]
    ]
    assert strengths == sorted(strengths, reverse=True)


def test_non_numeric_columns_skipped_and_reported():
    df = _df_with_known_correlations()
    result = compute_correlation(df, "x")
    assert "category" in result["skipped_non_numeric_columns"]


#def test_threshold_filter_works():
#    df = _df_with_known_correlations()
#    result = compute_correlation(df, "x", min_abs_correlation=0.9)
    # Only strong correlations should pass
#    for c in result["correlations"]:
#        assert max(abs(c["pearson"]), abs(c["spearman"])) >= 0.9


def test_non_linear_signal_field_present():
    df = _df_with_known_correlations()
    result = compute_correlation(df, "x")
    for c in result["correlations"]:
        assert "non_linear_signal" in c
        assert c["non_linear_signal"] >= 0  # absolute difference, always non-negative


def test_no_numeric_columns_handled():
    df = pd.DataFrame({"target": [1, 2, 3], "label": ["a", "b", "c"]})
    result = compute_correlation(df, "target")
    assert result["correlations"] == []
    assert "message" in result