"""
Tool : compute_correlation

Compute the correlations(pearson and spearman) between target column and
all other numeric columns. Ignores the non-numeric columns.

"""

import pandas as pd 
import matplotlib.pyplot as plt

# Tool definition for ai agent

TOOL_DEFINITION = {
    "name" : "compute_correlation",
    "description" :( "Compute correlations between a target numeric column and all other "
        "numeric columns in the dataset. Returns both Pearson (linear) and "
        "Spearman (rank-based, captures non-linear monotonic relationships) "
        "correlations, sorted by absolute strength. "
        "\n\n"
        "Use this when the user asks what's related to, drives, predicts, or "
        "correlates with a specific variable. The target column must be numeric. "
        "Non-numeric columns in the dataset are skipped and reported separately. "
        "\n\n"
        "Note: a large gap between Pearson and Spearman indicates a non-linear "
        "relationship worth investigating with a chart."
    ),
    "input_schema" : {
        "type" : "object",
        "properties" : {
            "target_column" : {
                "type" : "string",
                "description" : "The numeric column to compute correlation against."
            },
            "min_abs_correlation" : {
                "type" : "number",
                "description" : (
                    "Optional filter: only return correlations with absolute "
                    "value at least this large. Default 0.0 (return all)."
                ),
                "default" : 0.0
            },
        },
         "required": ["target_column"],
    },
}

# Implementation 

def compute_correlation(df : pd.DataFrame, target_column : str, min_abs_correlation : float = 0.0) -> dict :
    """
     Compute Pearson and Spearman correlations against a target column.

    Args:
        df: The pandas DataFrame.
        target_column: The column to compute correlations against. Must be numeric.
        min_abs_correlation: Filter threshold; only return correlations with
            absolute value at least this large.

    Returns:
        A dict with the target, the sorted correlation list, and any skipped columns.
        Or an error dict if the target column is invalid.
    
    """

    # find the target column 

    if target_column not in df.columns:
        return{
            "error" : f"Column '{target_column}' not exist in the dataset." ,
            "hint" : f"Available columns : {list(df.columns)}",
        }
    
    # checking target column is numeric or not 
    
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        return {
            "error": f"Column '{target_column}' is not numeric (dtype: {df[target_column].dtype}).",
            "hint": "Correlation requires a numeric target column.",
        }
    
    # collecting other numeric columns(excluding target column) to compute the correlation with target column 

    numeric_columns = df.select_dtypes(include = "number").columns.tolist()

    numeric_columns = [c for c in numeric_columns if c != target_column]
    skipped_non_numeric = [c for c in df.columns if c != target_column and c not in numeric_columns]

    if not numeric_columns : 
        return{
            "target_column" : target_column,
            "correlations" : [],
            "message" : f"No other columns available to correlate against",
            "skipped_non_numeric_columns" : skipped_non_numeric,
        }
    target = df[target_column]
    correlations = []

    for col in numeric_columns: 
        pearson = target.corr(df[col], method = "pearson")
        spearman = target.corr(df[col], method = "spearman")

        if pd.isna(pearson) or pd.isna(spearman):
            continue

        correlations.append(
            {
                "column": col,
                "pearson": round(float(pearson), 4),
                "spearman": round(float(spearman), 4),
                "non_linear_signal": round(abs(float(spearman) - float(pearson)), 4),
            }
        )

    # Sort by strongest absolute correlation (Pearson or Spearman, whichever is larger)
    correlations.sort(
        key=lambda x: max(abs(x["pearson"]), abs(x["spearman"])),
        reverse=True,
    )

    return {
        "target_column": target_column,
        "method": "pearson + spearman",
        "correlations": correlations,
        "n_correlations": len(correlations),
        "skipped_non_numeric_columns": skipped_non_numeric,
    }

        