"""Tests for the run_query tool."""

import pandas as pd

from datachat.tools.query import run_query


def _sample_df():
    return pd.DataFrame(
        {
            "country": ["UK", "UK", "US", "FR", "UK", "DE"],
            "revenue": [100, 200, 150, 75, 300, 120],
            "churned": [False, True, False, True, True, False],
        }
    )


def test_basic_filter_returns_match_count():
    df = _sample_df()
    result = run_query(df, "country == 'UK'")
    assert result["match_count"] == 3
    assert result["total_rows"] == 6
    assert result["match_percentage"] == 50.0


def test_sample_rows_returned():
    df = _sample_df()
    result = run_query(df, "country == 'UK'", sample_size=2)
    assert len(result["sample_rows"]) == 2


def test_sample_size_clamped():
    df = _sample_df()
    result = run_query(df, "revenue > 0", sample_size=500)
    # Only 6 rows in the test df, so sample_size is effectively bounded by data
    assert len(result["sample_rows"]) <= 20


def test_no_matches_returns_clear_message():
    df = _sample_df()
    result = run_query(df, "country == 'JP'")
    assert result["match_count"] == 0
    assert "message" in result
    assert result["sample_rows"] == []


def test_invalid_query_returns_error_not_exception():
    df = _sample_df()
    result = run_query(df, "this is not valid query syntax $$$$")
    assert "error" in result
    assert "hint" in result
    assert result["query_attempted"] == "this is not valid query syntax $$$$"


def test_missing_column_returns_error():
    df = _sample_df()
    result = run_query(df, "nonexistent_column > 10")
    assert "error" in result


def test_numeric_summary_computed_for_subset():
    df = _sample_df()
    result = run_query(df, "country == 'UK'")
    assert "numeric_summary" in result
    assert "revenue" in result["numeric_summary"]
    # UK revenues are 100, 200, 300 → mean 200
    assert result["numeric_summary"]["revenue"]["mean"] == 200.0


def test_compound_query_works():
    df = _sample_df()
    result = run_query(df, "country == 'UK' and churned == True")
    assert result["match_count"] == 2