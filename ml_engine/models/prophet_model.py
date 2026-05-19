"""
Prophet-based anomaly detector and 24h forecaster.
Anomalies are points that fall outside Prophet's uncertainty interval.
Also provides forecast_horizon(): how many hours until a threshold is crossed.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ml_engine.models.base_model import AnomalyScore, BaseAnomalyModel

logger = logging.getLogger(__name__)


class ProphetModel(BaseAnomalyModel):
    """
    Wraps Facebook Prophet for time-series forecasting.
    Anomaly score = normalized distance of actual value outside the confidence interval.
    """

    def __init__(
        self,
        forecast_horizon_hours: int = 24,
        interval_width: float = 0.95,
        seasonality_mode: str = "multiplicative",
    ):
        self._horizon_hours = forecast_horizon_hours
        self._interval_width = interval_width
        self._seasonality_mode = seasonality_mode
        self._model = None
        self._fitted = False
        self._last_forecast: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return "prophet"

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, df: pd.DataFrame) -> None:
        """
        Train Prophet on historical data.
        df must have columns: [timestamp, value]
        Requires at least 2 full periods (ideally 24h+) of data.
        """
        try:
            from prophet import Prophet  # import here — Prophet has slow import
        except ImportError:
            logger.error("prophet package not installed. Run: pip install prophet")
            return

        if len(df) < 50:
            logger.warning("Prophet needs ≥50 points, got %d. Skipping fit.", len(df))
            return

        train = pd.DataFrame({
            "ds": pd.to_datetime(df["timestamp"]).dt.tz_localize(None),
            "y": df["value"].values,
        })

        self._model = Prophet(
            interval_width=self._interval_width,
            seasonality_mode=self._seasonality_mode,
            daily_seasonality=True,
            weekly_seasonality=False,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        # suppress Prophet's verbose output
        import logging as _logging
        _logging.getLogger("prophet").setLevel(_logging.WARNING)
        _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)

        self._model.fit(train)
        self._fitted = True

        # pre-compute forecast for the next horizon
        self._refresh_forecast(train)
        logger.info("Prophet fitted on %d points, forecast %dh ahead", len(train), self._horizon_hours)

    def _refresh_forecast(self, train: pd.DataFrame) -> None:
        freq_minutes = self._infer_freq_minutes(train)
        periods = int(self._horizon_hours * 60 / freq_minutes)
        future = self._model.make_future_dataframe(periods=periods, freq=f"{freq_minutes}min")
        self._last_forecast = self._model.predict(future)

    @staticmethod
    def _infer_freq_minutes(train: pd.DataFrame) -> int:
        if len(train) < 2:
            return 1
        diffs = pd.to_datetime(train["ds"]).diff().dropna()
        median_seconds = diffs.median().total_seconds()
        return max(1, int(median_seconds / 60))

    def score(self, df: pd.DataFrame) -> list[AnomalyScore]:
        """
        Score each row in df against Prophet's predicted interval.
        Points outside [yhat_lower, yhat_upper] get a score proportional to the breach.
        """
        if not self._fitted or self._last_forecast is None:
            return []

        timestamps = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        forecast = self._last_forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]]

        results = []
        for i, (ts, val) in enumerate(zip(timestamps, df["value"])):
            # find nearest forecast point
            idx = (forecast.index - ts).abs().argmin()
            row = forecast.iloc[idx]

            yhat, lower, upper = row["yhat"], row["yhat_lower"], row["yhat_upper"]
            interval_width = max(upper - lower, 1e-9)

            if val < lower:
                raw_breach = (lower - val) / interval_width
            elif val > upper:
                raw_breach = (val - upper) / interval_width
            else:
                raw_breach = 0.0

            # sigmoid to keep score in [0, 1] with smooth transition
            anomaly_score = 1.0 / (1.0 + np.exp(-2 * (raw_breach - 0.5)))

            ts_orig = df["timestamp"].iloc[i]
            if not isinstance(ts_orig, datetime):
                ts_orig = pd.Timestamp(ts_orig).to_pydatetime()

            results.append(
                AnomalyScore(
                    metric_name=df.get("metric_name", pd.Series(["unknown"])).iloc[0] if "metric_name" in df else "unknown",
                    timestamp=ts_orig,
                    score=float(anomaly_score),
                    model_name=self.name,
                    details={
                        "yhat": float(yhat),
                        "yhat_lower": float(lower),
                        "yhat_upper": float(upper),
                        "actual": float(val),
                        "breach_magnitude": float(raw_breach),
                    },
                )
            )
        return results

    def forecast_dataframe(self) -> pd.DataFrame | None:
        """Return the full Prophet forecast DataFrame (includes yhat, yhat_lower, yhat_upper)."""
        return self._last_forecast

    def time_to_threshold(
        self,
        threshold: float,
        direction: str = "above",
    ) -> float | None:
        """
        Estimate hours until forecast (yhat) crosses `threshold`.
        direction: 'above' or 'below'
        Returns hours as float, or None if threshold is never crossed in forecast horizon.
        """
        if self._last_forecast is None:
            return None

        now = datetime.now()
        future = self._last_forecast[self._last_forecast["ds"] >= now].copy()
        if future.empty:
            return None

        if direction == "above":
            crossings = future[future["yhat"] >= threshold]
        else:
            crossings = future[future["yhat"] <= threshold]

        if crossings.empty:
            return None

        first_crossing = pd.Timestamp(crossings.iloc[0]["ds"])
        delta = first_crossing - pd.Timestamp(now)
        return delta.total_seconds() / 3600
