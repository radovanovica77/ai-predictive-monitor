"""
Transforms raw time-series values into a feature matrix suitable for ML models.
Features capture trend, volatility, seasonality, and rate-of-change signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame, window_sizes: list[int] | None = None) -> pd.DataFrame:
    """
    Input df: columns [timestamp, value], sorted ascending.
    Returns a DataFrame with engineered features; same index as input.

    Requires at least max(window_sizes) rows. Shorter series return NaN rows
    at the head — callers should dropna() before fitting models.
    """
    if window_sizes is None:
        window_sizes = [5, 10, 30]

    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    features = pd.DataFrame(index=df.index)

    v = df["value"]

    # --- raw value ---
    features["value"] = v

    # --- rolling statistics ---
    for w in window_sizes:
        roll = v.rolling(w, min_periods=1)
        features[f"roll_mean_{w}"] = roll.mean()
        features[f"roll_std_{w}"] = roll.std().fillna(0)
        features[f"roll_min_{w}"] = roll.min()
        features[f"roll_max_{w}"] = roll.max()
        features[f"roll_range_{w}"] = features[f"roll_max_{w}"] - features[f"roll_min_{w}"]

    # --- z-score relative to longest rolling window ---
    w_long = max(window_sizes)
    mean_long = features[f"roll_mean_{w_long}"]
    std_long = features[f"roll_std_{w_long}"].replace(0, 1e-9)
    features["zscore"] = (v - mean_long) / std_long

    # --- rate of change (first and second derivative) ---
    features["diff_1"] = v.diff(1).fillna(0)
    features["diff_2"] = v.diff(2).fillna(0)
    features["pct_change_1"] = v.pct_change(1).replace([np.inf, -np.inf], 0).fillna(0)

    # --- lag features ---
    for lag in [1, 3, 5, 10]:
        features[f"lag_{lag}"] = v.shift(lag).bfill()

    # --- trend: linear slope over last 10 points ---
    def _slope(series: pd.Series) -> float:
        if len(series) < 2:
            return 0.0
        x = np.arange(len(series), dtype=float)
        return float(np.polyfit(x, series.values, 1)[0])

    features["trend_slope_10"] = v.rolling(10, min_periods=2).apply(_slope, raw=False).fillna(0)

    # --- time-of-day encoding (hour as sin/cos for cyclical continuity) ---
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
        features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    else:
        features["hour_sin"] = 0.0
        features["hour_cos"] = 1.0

    # --- deviation from expected (value / rolling mean ratio) ---
    features["value_to_mean_ratio"] = (v / mean_long.replace(0, 1e-9)).clip(-10, 10)

    features["timestamp"] = df["timestamp"]
    return features


def get_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Return only numeric feature columns (excludes timestamp)."""
    return [c for c in features_df.columns if c != "timestamp"]
