"""Tests for the describe_dataset tool."""

import pandas as pd
import numpy as np

from src.datachat.tools.describe import describe_dataset


def test_basic_shape():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = describe_dataset(df)
    assert result["shape"]["rows"] == 3
    assert result["shape"]["columns"] == 2


def test_missing_values_detected():
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    result = describe_dataset(df)
    assert result["missing_values"] == {"a": 1}
    # Column b has no nulls, so it should not appear
    assert "b" not in result["missing_values"]


def test_sample_size_clamped():
    df = pd.DataFrame({"a": range(100)})
    result = describe_dataset(df, sample_size=500)
    assert len(result["sample_rows"]) == 20  # capped at 20


def test_numeric_stats_present():
    df = pd.DataFrame({"revenue": [100, 200, 300, 400, 500]})
    result = describe_dataset(df)
    assert "numeric_stats" in result
    assert "revenue" in result["numeric_stats"]


def test_categorical_summary_for_low_cardinality():
    df = pd.DataFrame({"country": ["UK", "UK", "US", "FR"]})
    result = describe_dataset(df)
    assert "categorical_summary" in result
    assert result["categorical_summary"]["country"]["unique_values"] == 3
    assert "top_values" in result["categorical_summary"]["country"]


def test_categorical_summary_skips_top_values_for_high_cardinality():
    # 100 unique IDs — too many to list top values usefully
    df = pd.DataFrame({"user_id": [f"id_{i}" for i in range(100)]})
    result = describe_dataset(df)
    assert result["categorical_summary"]["user_id"]["unique_values"] == 100
    assert "top_values" not in result["categorical_summary"]["user_id"]