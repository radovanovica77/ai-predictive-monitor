"""
Builds Plotly chart JSON for the dashboard.
Returns dicts (not figures) so they can be JSON-serialised and sent to the browser.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def metric_history_chart(df: pd.DataFrame, metric_name: str, threshold: float | None = None) -> dict[str, Any]:
    """
    Time-series line chart with optional threshold line.
    Returns a Plotly figure dict suitable for JSON response.
    """
    timestamps = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
    values = df["value"].round(2).tolist()

    traces = [
        {
            "type": "scatter",
            "x": timestamps,
            "y": values,
            "mode": "lines",
            "name": metric_name,
            "line": {"color": "#3b82f6", "width": 2},
            "fill": "tozeroy",
            "fillcolor": "rgba(59,130,246,0.08)",
        }
    ]

    if threshold is not None:
        traces.append({
            "type": "scatter",
            "x": [timestamps[0], timestamps[-1]],
            "y": [threshold, threshold],
            "mode": "lines",
            "name": "Threshold",
            "line": {"color": "#ef4444", "width": 1, "dash": "dash"},
        })

    return {
        "data": traces,
        "layout": {
            "paper_bgcolor": "#111827",
            "plot_bgcolor": "#111827",
            "font": {"color": "#9ca3af", "family": "monospace"},
            "margin": {"l": 50, "r": 20, "t": 30, "b": 40},
            "xaxis": {"gridcolor": "#1f2937", "showgrid": True},
            "yaxis": {"gridcolor": "#1f2937", "showgrid": True},
            "showlegend": True,
            "legend": {"bgcolor": "rgba(0,0,0,0)"},
            "title": {"text": metric_name, "font": {"color": "#e2e8f0", "size": 13}},
        },
    }


def sparkline_data(df: pd.DataFrame, last_n: int = 60) -> dict[str, Any]:
    """Minimal data for a small inline sparkline (no axes, just the trend)."""
    tail = df.tail(last_n)
    return {
        "x": list(range(len(tail))),
        "y": tail["value"].round(2).tolist(),
    }
