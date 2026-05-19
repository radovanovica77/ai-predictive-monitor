"""
Combines multiple signals into a single calibrated confidence score [0–1].

Formula:
  confidence = anomaly_score × trend_consistency × stability_factor

Where:
  anomaly_score     — from ML ensemble (0–1)
  trend_consistency — how steady the trend is (not noisy/erratic)
  stability_factor  — penalizes very short horizons (< 2h = less certain)
"""
from __future__ import annotations

import math

from prediction_engine.models import TrendInfo


def compute_confidence(
    anomaly_score: float,
    trend: TrendInfo,
    horizon_hours: float,
    min_confidence: float = 0.40,
) -> float:
    """
    Returns a confidence value in [min_confidence, 0.97].

    The score is deliberately conservative — it should reflect the probability
    that the predicted event actually occurs, not just that something is anomalous.
    """
    # --- trend consistency (0–1): steady trend = higher confidence ---
    # High pct_change + aligned slope = consistent; erratic = lower
    slope_sign = 1 if trend.slope_per_minute >= 0 else -1
    pct_sign = 1 if trend.pct_change_over_window >= 0 else -1
    signs_agree = slope_sign == pct_sign

    # normalize pct_change magnitude to 0–1 (capped at 100%)
    pct_magnitude = min(abs(trend.pct_change_over_window) / 100.0, 1.0)
    trend_consistency = pct_magnitude * (1.0 if signs_agree else 0.5)
    trend_consistency = max(0.1, min(1.0, trend_consistency))

    # --- horizon stability factor ---
    # Short horizons are harder to predict precisely; longer is more certain
    # sigmoid: horizon=2h → 0.65, horizon=6h → 0.82, horizon=12h → 0.90
    horizon_factor = 1.0 / (1.0 + math.exp(-0.5 * (horizon_hours - 3)))
    horizon_factor = max(0.4, min(0.95, horizon_factor))

    # --- acceleration bonus ---
    # Accelerating growth strengthens confidence in the prediction
    accel_bonus = min(abs(trend.acceleration) * 5, 0.1)

    raw = anomaly_score * trend_consistency * horizon_factor + accel_bonus
    clamped = max(min_confidence, min(0.97, raw))

    return round(clamped, 3)


def severity_from_confidence_and_horizon(
    confidence: float,
    horizon_hours: float,
    current_pct_of_threshold: float,
) -> str:
    """
    Derive severity label from confidence, time-to-breach, and how close we are.

    Returns: 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'
    """
    from prediction_engine.models import Severity

    # Already very close to threshold
    if current_pct_of_threshold >= 95 or horizon_hours < 1:
        return Severity.CRITICAL
    if current_pct_of_threshold >= 85 or (horizon_hours < 4 and confidence >= 0.70):
        return Severity.HIGH
    if current_pct_of_threshold >= 70 or (horizon_hours < 12 and confidence >= 0.55):
        return Severity.MEDIUM
    return Severity.LOW
