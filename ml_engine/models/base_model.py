"""
Abstract base class all ML models in this system must implement.
Enforces a consistent fit/score/predict interface across different algorithms.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


@dataclass
class AnomalyScore:
    """Result of scoring a single metric window."""
    metric_name: str
    timestamp: datetime
    score: float          # 0.0 (normal) to 1.0 (highly anomalous)
    model_name: str
    details: dict = field(default_factory=dict)


class BaseAnomalyModel(ABC):
    """
    All anomaly detection models implement this interface.
    Enables the ensemble scorer to treat all models identically.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this model, e.g. 'isolation_forest'."""
        ...

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """
        Train the model on historical data.
        df must have columns: [timestamp, value]
        """
        ...

    @abstractmethod
    def score(self, df: pd.DataFrame) -> list[AnomalyScore]:
        """
        Score a window of recent observations.
        Returns one AnomalyScore per row (or per window, model-dependent).
        df must have columns: [timestamp, value]
        """
        ...

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """True if fit() has been called at least once."""
        ...

    def score_latest(self, df: pd.DataFrame) -> AnomalyScore | None:
        """Convenience: score the full df and return only the last point's score."""
        scores = self.score(df)
        return scores[-1] if scores else None
