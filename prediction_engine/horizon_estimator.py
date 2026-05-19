"""
Estimates how many hours until a metric crosses a critical threshold.

Strategy:
  1. Fit both linear and exponential regression on the recent window
  2. Pick whichever has lower RMSE on the window
  3. Extrapolate forward until the model's predicted value crosses the threshold
  4. Return horizon in hours + a TrendInfo describing current movement

Linear fit:   y = a*t + b
Exponential:  y = b * e^(a*t)   (only when all values positive and growing)
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from prediction_engine.models import TrendInfo

logger = logging.getLogger(__name__)

MAX_HORIZON_HOURS = 24.0
MIN_HORIZON_HOURS = 0.25    # 15 minutes minimum — anything sooner is already an incident
WINDOW_MINUTES = 60         # how much recent history to use for regression


def estimate_horizon(
    df: pd.DataFrame,
    threshold: float,
    direction: str = "above",
    window_minutes: int = WINDOW_MINUTES,
) -> tuple[float | None, TrendInfo]:
    """
    Parameters
    ----------
    df         : DataFrame with [timestamp, value], sorted ascending
    threshold  : The critical value the metric must not cross
    direction  : 'above' (metric must stay below threshold) or
                 'below' (metric must stay above threshold)
    window_minutes : how many recent minutes of data to use for regression

    Returns
    -------
    (horizon_hours, TrendInfo)
      horizon_hours — estimated hours until breach, clamped to [MIN, MAX].
                      None if the metric is moving away from threshold.
    """
    if df.empty or len(df) < 5:
        return None, _zero_trend(window_minutes)

    # trim to analysis window
    now = df["timestamp"].max()
    if hasattr(now, "to_pydatetime"):
        now = now.to_pydatetime()
    cutoff = pd.Timestamp(now) - pd.Timedelta(minutes=window_minutes)
    window = df[df["timestamp"] >= cutoff].copy()

    if len(window) < 5:
        window = df.tail(30).copy()

    values = window["value"].values.astype(float)
    timestamps = pd.to_datetime(window["timestamp"])

    # convert timestamps to minutes-from-start for regression
    t0 = timestamps.iloc[0]
    t_minutes = np.array([(ts - t0).total_seconds() / 60 for ts in timestamps])

    trend = _compute_trend(values, t_minutes, window_minutes)

    # check if already breached
    current = values[-1]
    if direction == "above" and current >= threshold:
        return 0.0, trend
    if direction == "below" and current <= threshold:
        return 0.0, trend

    # check if moving away from threshold — no breach predicted
    if direction == "above" and trend.slope_per_minute <= 0:
        return None, trend
    if direction == "below" and trend.slope_per_minute >= 0:
        return None, trend

    # fit models and pick best
    horizon_minutes = _extrapolate_best(values, t_minutes, threshold, direction)

    if horizon_minutes is None or horizon_minutes <= 0:
        # Fallback: simple linear extrapolation from current value and slope
        if trend.slope_per_minute != 0:
            current = float(values[-1])
            gap = (threshold - current) if direction == "above" else (current - threshold)
            if gap > 0 and abs(trend.slope_per_minute) > 1e-9:
                horizon_minutes = gap / abs(trend.slope_per_minute)
            else:
                return None, trend
        else:
            return None, trend

    horizon_hours = horizon_minutes / 60.0
    horizon_hours = max(MIN_HORIZON_HOURS, min(MAX_HORIZON_HOURS, horizon_hours))

    return round(horizon_hours, 2), trend


def _compute_trend(values: np.ndarray, t_minutes: np.ndarray, window_minutes: int) -> TrendInfo:
    """Compute TrendInfo from a values array and time axis in minutes."""
    if len(values) < 2:
        return _zero_trend(window_minutes)

    # linear slope via least squares
    if len(t_minutes) >= 2 and t_minutes[-1] > t_minutes[0]:
        coeffs = np.polyfit(t_minutes, values, 1)
        slope = float(coeffs[0])
    else:
        slope = 0.0

    # second derivative (acceleration) via quadratic fit
    acceleration = 0.0
    if len(values) >= 5:
        try:
            quadratic = np.polyfit(t_minutes, values, 2)
            acceleration = float(quadratic[0] * 2)  # 2a from ax^2+bx+c
        except np.linalg.LinAlgError:
            pass

    # % change from start to end of window
    start_val = float(values[0]) if values[0] != 0 else 1e-9
    end_val = float(values[-1])
    pct_change = (end_val - start_val) / abs(start_val) * 100

    direction = "stable"
    if pct_change > 3:
        direction = "increasing"
    elif pct_change < -3:
        direction = "decreasing"

    return TrendInfo(
        slope_per_minute=round(slope, 4),
        pct_change_over_window=round(pct_change, 1),
        window_minutes=window_minutes,
        direction=direction,
        acceleration=round(acceleration, 6),
    )


def _extrapolate_best(
    values: np.ndarray,
    t_minutes: np.ndarray,
    threshold: float,
    direction: str,
) -> float | None:
    """
    Fit linear and exponential models; pick lower RMSE; extrapolate to threshold.
    Returns minutes until breach, or None.
    """
    linear_horizon = _linear_extrapolate(values, t_minutes, threshold, direction)
    exp_horizon = _exp_extrapolate(values, t_minutes, threshold, direction)

    # compare RMSE on the window to pick better model
    linear_rmse = _linear_rmse(values, t_minutes)
    exp_rmse = _exp_rmse(values, t_minutes)

    if exp_rmse is not None and exp_rmse < linear_rmse and exp_horizon is not None:
        logger.debug("Using exponential fit (RMSE %.3f < linear %.3f)", exp_rmse, linear_rmse)
        return exp_horizon

    return linear_horizon


def _linear_extrapolate(
    values: np.ndarray,
    t_minutes: np.ndarray,
    threshold: float,
    direction: str,
) -> float | None:
    if len(t_minutes) < 2:
        return None
    try:
        a, b = np.polyfit(t_minutes, values, 1)
    except np.linalg.LinAlgError:
        return None

    if abs(a) < 1e-12:
        return None

    t_breach = (threshold - b) / a
    t_now = float(t_minutes[-1])
    minutes_from_now = t_breach - t_now

    return minutes_from_now if minutes_from_now > 0 else None


def _exp_extrapolate(
    values: np.ndarray,
    t_minutes: np.ndarray,
    threshold: float,
    direction: str,
) -> float | None:
    if np.any(values <= 0) or threshold <= 0:
        return None
    try:
        log_values = np.log(values)
        a, log_b = np.polyfit(t_minutes, log_values, 1)
    except (np.linalg.LinAlgError, FloatingPointError):
        return None

    if abs(a) < 1e-12:
        return None

    # y = e^(log_b) * e^(a*t) = threshold  →  t = (log(threshold) - log_b) / a
    try:
        t_breach = (math.log(threshold) - log_b) / a
    except (ValueError, ZeroDivisionError):
        return None

    t_now = float(t_minutes[-1])
    minutes_from_now = t_breach - t_now

    return minutes_from_now if minutes_from_now > 0 else None


def _linear_rmse(values: np.ndarray, t_minutes: np.ndarray) -> float:
    if len(t_minutes) < 2:
        return float("inf")
    try:
        a, b = np.polyfit(t_minutes, values, 1)
        predicted = a * t_minutes + b
        return float(np.sqrt(np.mean((values - predicted) ** 2)))
    except np.linalg.LinAlgError:
        return float("inf")


def _exp_rmse(values: np.ndarray, t_minutes: np.ndarray) -> float | None:
    if np.any(values <= 0):
        return None
    try:
        log_values = np.log(values)
        a, log_b = np.polyfit(t_minutes, log_values, 1)
        predicted = np.exp(log_b + a * t_minutes)
        return float(np.sqrt(np.mean((values - predicted) ** 2)))
    except (np.linalg.LinAlgError, FloatingPointError):
        return None


def _zero_trend(window_minutes: int) -> TrendInfo:
    return TrendInfo(
        slope_per_minute=0.0,
        pct_change_over_window=0.0,
        window_minutes=window_minutes,
        direction="stable",
        acceleration=0.0,
    )
