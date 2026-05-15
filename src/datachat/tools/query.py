"""
Tool: run_query

Filter a pandas DataFrame using a query string (pandas .query() syntax)
and return metadata + a sample of matching rows. Designed for LLM
reasoning, not for bulk data retrieval.
"""

import pandas as pd


# ────────────────────────────────────────────────────────────────
# Tool definition (what Claude sees)
# ────────────────────────────────────────────────────────────────

TOOL_DEFINITION = {
    "name": "run_query",
    "description": (
        "Filter the dataset to a subset matching a condition, then return "
        "the count and a summary of the filtered rows. Use pandas query "
        "syntax, e.g. \"revenue > 10000\", \"region == 'UK' and churned == True\", "
        "\"signup_date >= '2024-01-01'\". "
        "\n\n"
        "Use this when you need to narrow the dataset to a subset before "
        "further analysis (e.g. 'look at only Q3 rows', 'focus on churned "
        "customers'). The tool returns the count of matching rows, a sample "
        "of up to 20 rows, and a brief summary of the subset — NOT the full "
        "filtered data. For bulk data export, the user has a separate UI option."
        "This is basically another describe tool however this a bit short compared"
        "to what we have."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A pandas query string. Column names must exist in the "
                    "dataset. String values must be in quotes. "
                    "Example: \"age > 30 and country == 'UK'\""
                ),
            },
            "sample_size": {
                "type": "integer",
                "description": "Number of sample rows to return (default 10, max 20).",
                "default": 10,
            },
        },
        "required": ["query"],
    },
}


# ────────────────────────────────────────────────────────────────
# Implementation (what my code does)
# ────────────────────────────────────────────────────────────────

def run_query(df: pd.DataFrame, query: str, sample_size: int = 10) -> dict:
    """
    Run a pandas query against the DataFrame and return a summary of the result.

    Args:
        df: The pandas DataFrame to query.
        query: A pandas .query() syntax string.
        sample_size: Number of sample rows to include (capped at 20).

    Returns:
        A dict containing:
          - match_count: number of matching rows
          - total_rows: total rows in the dataset (for context)
          - match_percentage: what fraction matched
          - sample_rows: list of up to sample_size rows
          - numeric_summary: quick stats on numeric columns in the subset
        Or, if the query fails:
          - error: a helpful message Claude can use to retry
    """
    sample_size = min(max(sample_size, 1), 20)

    # Try to run the query — catch errors and return them in a structured way
    # so Claude can self-correct rather than crashing the agent loop.
    try:
        filtered = df.query(query)
    except Exception as e:
        return {
            "error": f"Query failed: {type(e).__name__}: {e}",
            "hint": (
                "Check that column names exist exactly as in the dataset, "
                "string values are quoted, and the syntax is valid pandas .query()."
            ),
            "query_attempted": query,
        }

    match_count = int(len(filtered))
    total_rows = int(len(df))

    # Handle the "no matches" case explicitly — don't leave Claude guessing.
    if match_count == 0:
        return {
            "match_count": 0,
            "total_rows": total_rows,
            "match_percentage": 0.0,
            "sample_rows": [],
            "message": "No rows matched the query.",
            "query": query,
        }

    result = {
        "match_count": match_count,
        "total_rows": total_rows,
        "match_percentage": round(100 * match_count / total_rows, 2),
        "sample_rows": filtered.head(sample_size).to_dict(orient="records"),
        "query": query,
    }

    # Quick numeric summary of the filtered subset (means, min, max)
    # Helps Claude reason about whether the subset "looks different" from the whole
    numeric_cols = filtered.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        result["numeric_summary"] = {
            col: {
                "mean": round(float(filtered[col].mean()), 4),
                "median": round(float(filtered[col].median()), 4),
                "min": round(float(filtered[col].min()), 4),
                "max": round(float(filtered[col].max()), 4),
            }
            for col in numeric_cols
        }

    return result