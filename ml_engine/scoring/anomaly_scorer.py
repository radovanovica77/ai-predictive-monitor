"""
Ensemble anomaly scorer: combines IsolationForest and Prophet scores into
a single calibrated 0–1 score per metric point.

Weighting: 0.4 × IsolationForest + 0.6 × Prophet
(Prophet weighted higher because it's time-aware and forecasts context.)
When only one model is available, uses its score directly.
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from ml_engine.models.base_model import AnomalyScore

logger = logging.getLogger(__name__)

IF_WEIGHT = 0.4
PROPHET_WEIGHT = 0.6


class AnomalyScorer:
    """
    Merges per-model AnomalyScore lists into a single ensemble score.
    Models are optional: if Prophet isn't fitted yet, IF carries full weight.
    """

    def __init__(
        self,
        if_weight: float = IF_WEIGHT,
        prophet_weight: float = PROPHET_WEIGHT,
    ):
        self._if_weight = if_weight
        self._prophet_weight = prophet_weight

    def combine(
        self,
        if_scores: list[AnomalyScore],
        prophet_scores: list[AnomalyScore],
        metric_name: str,
    ) -> list[AnomalyScore]:
        """
        Merge two score lists by timestamp proximity.
        Returns one ensemble AnomalyScore per timestamp in if_scores.
        """
        if not if_scores and not prophet_scores:
            return []

        # Build lookup for prophet by timestamp (nearest match)
        prophet_by_ts: dict[datetime, float] = {s.timestamp: s.score for s in prophet_scores}

        results = []
        for if_score in if_scores:
            prophet_score = self._nearest_prophet_score(if_score.timestamp, prophet_by_ts)

            if prophet_score is None:
                # Prophet not available — IF score alone, adjusted weight
                ensemble = if_score.score
                weights_used = {"if": 1.0, "prophet": 0.0}
            else:
                total_weight = self._if_weight + self._prophet_weight
                ensemble = (
                    self._if_weight * if_score.score + self._prophet_weight * prophet_score
                ) / total_weight
                weights_used = {"if": self._if_weight, "prophet": self._prophet_weight}

            results.append(
                AnomalyScore(
                    metric_name=metric_name,
                    timestamp=if_score.timestamp,
                    score=round(float(ensemble), 4),
                    model_name="ensemble",
                    details={
                        "if_score": round(if_score.score, 4),
                        "prophet_score": round(prophet_score, 4) if prophet_score is not None else None,
                        "weights": weights_used,
                    },
                )
            )
        return results

    @staticmethod
    def _nearest_prophet_score(
        target: datetime,
        prophet_map: dict[datetime, float],
        max_delta_seconds: float = 120,
    ) -> float | None:
        if not prophet_map:
            return None
        target_naive = target.replace(tzinfo=None) if target.tzinfo is not None else target
        closest = min(prophet_map.keys(), key=lambda ts: abs((ts - target_naive).total_seconds()))
        delta = abs((closest - target_naive).total_seconds())
        return prophet_map[closest] if delta <= max_delta_seconds else None
