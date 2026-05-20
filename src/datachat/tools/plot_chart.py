"""
Tool : plot_chart
Generate a chart from the dataset and save it to the disk.
Return a file path for streamlit UI to display the chart for the users and
return a description for claude to reason about.
"""

import os 
import pandas as pd 
import matplotlib.pyplot as plt

# for claude 

TOOL_DEFINITION = {
    "name" : "plot_chart",
    "description" : (
        "Draw chart from the given column of the dataset and save the chart "
        "to the local machine. Saves the chart to disk and returns a file path for "
        "the UI to display, along with a text description of what the chart shows."
        "\n\n"
        "Use this tool when the user usually wants to see the trend over time, "
        "data distibution, relationship between two given columns. Such examples are"
        "'Show me the revenue trend over time(line chart)', 'What's the distribution "
        "of customer ages(histogram)', 'How does churn vary by region(bar chart)', "
        "'Is there a relationship between tenure and revenue?(scatter plot)'."
        "\n\n"
        "Returns a dict with: chart_path (file location for display), "
        "description (text summary of what the chart shows for Claude to"
        "include in its response), and chart_type. Supported chart types : "
        "histogram, bar, scatter, line. For scatter plots, provide both column"
        "(x-axis), and y_columnn(y-axis)"
    )
}