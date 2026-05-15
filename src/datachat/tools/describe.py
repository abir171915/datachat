"""
Tool: describe_dataset

Returns a structured summary of a pandas DataFrame designed for an LLM
to reason about. Includes shape, column types, missing values, sample rows,
numeric statistics, and categorical summaries.
"""

import pandas as pd


# ────────────────────────────────────────────────────────────────
# Tool definition (for Claude )
# ────────────────────────────────────────────────────────────────

TOOL_DEFINITION = {
    "name": "describe_dataset",
    "description": (
        "Returns a comprehensive summary of the current dataset including "
        "its shape, column names and types, missing value counts, numeric "
        "statistics (mean, std, min, max, quartiles), sample rows, and "
        "categorical value counts. Always call this first when you need "
        "to understand the structure of a new dataset before any analysis."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sample_size": {
                "type": "integer",
                "description": "Number of sample rows to include (default 5, max 20).",
                "default": 5,
            }
        },
        "required": [],
    },
}


# ────────────────────────────────────────────────────────────────
# Implementation (what your code does)
# ────────────────────────────────────────────────────────────────

def describe_dataset(df: pd.DataFrame, sample_size: int = 5) -> dict:
    """
    Generate a structured summary of the DataFrame.

    Args:
        df: The pandas DataFrame to describe.
        sample_size: Number of sample rows to include (capped at 20).

    Returns:
        A dict containing shape, columns, types, nulls, stats, samples,
        and categorical summaries.
    """
    sample_size = min(max(sample_size, 1), 20)

    # Build the base summary
    summary = {
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {
            col: int(df[col].isnull().sum())
            for col in df.columns
            if df[col].isnull().any()
        },
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 3),
    }

    # Sample rows: always include head, and add a random sample for variety.
    # Random sample uses a fixed seed for reproducibility across calls.
    head_rows = df.head(sample_size).to_dict(orient="records")

    if len(df) > sample_size:
        random_rows = (
            df.sample(n=sample_size, random_state=42)
            .to_dict(orient="records")
        )
    else:
        random_rows = []  # Dataset too small; head already covers everything

    summary["sample_rows"] = {
        "head": head_rows,
        "random_sample": random_rows,
    }

    # Numeric column statistics
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        numeric_stats = df[numeric_cols].describe().round(4).to_dict()
        summary["numeric_stats"] = numeric_stats

    # Categorical column summaries (object / category / bool dtypes)
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if categorical_cols:
        cat_summary = {}
        for col in categorical_cols:
            unique_count = df[col].nunique()
            cat_info = {"unique_values": int(unique_count)}
            # Only include top values if cardinality is reasonable
            if unique_count <= 20:
                cat_info["top_values"] = df[col].value_counts().head(10).to_dict()
            cat_summary[col] = cat_info
        summary["categorical_summary"] = cat_summary

    return summary