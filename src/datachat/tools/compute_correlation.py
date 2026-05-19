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
