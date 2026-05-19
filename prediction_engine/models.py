"""
Pydantic dataclasses for all prediction engine outputs.
These are the objects that flow from predictor → dashboard API → frontend.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Scenario(str, Enum):
    CONNECTION_POOL_EXHAUSTION = "CONNECTION_POOL_EXHAUSTION"
    MEMORY_LEAK = "MEMORY_LEAK"
    MEMORY_PRESSURE = "MEMORY_PRESSURE"
    TRAFFIC_SPIKE = "TRAFFIC_SPIKE"
    LATENCY_DEGRADATION = "LATENCY_DEGRADATION"
    LATENCY_SLA_BREACH = "LATENCY_SLA_BREACH"
    DISK_FILL = "DISK_FILL"
    CPU_SATURATION = "CPU_SATURATION"
    ERROR_RATE_SPIKE = "ERROR_RATE_SPIKE"
    UNKNOWN = "UNKNOWN"


class TrendInfo(BaseModel):
    """Describes how the metric is moving over the recent window."""
    slope_per_minute: float = Field(description="Rate of change per minute")
    pct_change_over_window: float = Field(description="% change vs start of analysis window")
    window_minutes: int = Field(description="Duration of the analysis window in minutes")
    direction: str = Field(description="'increasing', 'decreasing', or 'stable'")
    acceleration: float = Field(default=0.0, description="Second derivative — positive means accelerating growth")

    @property
    def direction_label(self) -> str:
        if self.pct_change_over_window > 5:
            return "increasing"
        elif self.pct_change_over_window < -5:
            return "decreasing"
        return "stable"


class Evidence(BaseModel):
    """Supporting data that backs up a prediction — shown in the dashboard."""
    current_value: float
    threshold_value: float
    current_pct_of_threshold: float = Field(description="current_value / threshold * 100")
    trend: TrendInfo
    anomaly_score: float = Field(ge=0.0, le=1.0)
    contributing_metrics: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class Prediction(BaseModel):
    """
    A single actionable prediction emitted by the prediction engine.
    The `message` field is always specific and human-readable.
    """
    id: str = Field(description="Unique prediction ID: {metric}:{scenario}:{timestamp_iso}")
    metric_name: str
    scenario: Scenario
    severity: Severity
    horizon_hours: float = Field(description="Estimated hours until threshold is crossed")
    confidence: float = Field(ge=0.0, le=1.0, description="Calibrated confidence [0–1]")
    message: str = Field(description="Specific, human-readable prediction message")
    evidence: Evidence
    created_at: datetime
    expires_at: datetime = Field(description="When this prediction should be considered stale")
    is_active: bool = True

    @property
    def horizon_label(self) -> str:
        """e.g. '~6 hours', '~45 minutes', '< 2 hours'"""
        h = self.horizon_hours
        if h < 1:
            minutes = int(h * 60)
            return f"~{minutes} minutes"
        elif h < 2:
            return f"~{h:.1f} hours"
        else:
            return f"~{int(round(h))} hours"

    @property
    def confidence_label(self) -> str:
        """e.g. 'High confidence (84%)'"""
        pct = int(self.confidence * 100)
        if self.confidence >= 0.80:
            label = "High confidence"
        elif self.confidence >= 0.60:
            label = "Medium confidence"
        else:
            label = "Low confidence"
        return f"{label} ({pct}%)"

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")
