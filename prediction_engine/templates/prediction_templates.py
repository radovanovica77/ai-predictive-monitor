"""
Maps (scenario, severity) pairs to specific, human-readable prediction messages.

Design rule: every message must answer three questions:
  1. WHAT will happen?     ("Database will hit connection limits")
  2. WHEN?                 ("in ~6 hours")
  3. WHY / what evidence?  ("based on +34% growth over 2h at current traffic pattern")

Never produce: "Something might go wrong" or "Anomaly detected."
"""
from __future__ import annotations

import math
from typing import Callable

from prediction_engine.models import Evidence, Scenario, Severity, TrendInfo


# ------------------------------------------------------------------
# Template functions — each takes Evidence and returns a str message
# ------------------------------------------------------------------

def _fmt_hours(h: float | None) -> str:
    if h is None or h <= 0:
        return "imminently (< 15 min)"
    if h < 1:
        m = int(h * 60)
        return f"~{m} min"
    elif h < 2:
        return f"~{h:.1f}h"
    return f"~{int(round(h))}h"


def _fmt_trend(trend: TrendInfo) -> str:
    """Produce a compact trend description like '+34% over 2h' or '-12% over 30min'."""
    pct = trend.pct_change_over_window
    sign = "+" if pct >= 0 else ""
    w = trend.window_minutes
    window_label = f"{int(w / 60)}h" if w >= 60 else f"{w}min"
    return f"{sign}{pct:.0f}% over {window_label}"


def _fmt_slope(trend: TrendInfo, unit: str = "") -> str:
    """e.g. '+1.2 connections/min' or '+0.8 MB/min'"""
    sign = "+" if trend.slope_per_minute >= 0 else ""
    return f"{sign}{trend.slope_per_minute:.1f}{unit}/min"


def _acceleration_note(trend: TrendInfo) -> str:
    if trend.acceleration > 0.05:
        return " Growth is accelerating."
    elif trend.acceleration < -0.05:
        return " Rate of growth is slowing."
    return ""


# ---- CONNECTION POOL EXHAUSTION ----

def _msg_connection_pool(ev: Evidence, horizon: float, severity: Severity) -> str:
    current_pct = ev.current_pct_of_threshold
    trend_str = _fmt_trend(ev.trend)
    slope_str = _fmt_slope(ev.trend, " conn")
    accel = _acceleration_note(ev.trend)

    if severity == Severity.CRITICAL:
        return (
            f"Database connection pool at {current_pct:.0f}% capacity "
            f"({ev.current_value:.0f}/{ev.threshold_value:.0f} connections). "
            f"At current rate ({trend_str}, {slope_str}), pool will be exhausted in {_fmt_hours(horizon)} "
            f"— new connections will be rejected.{accel} "
            f"Immediate action required: check for connection leaks or add pool capacity."
        )
    elif severity == Severity.HIGH:
        return (
            f"Database connection pool growing toward limit "
            f"({ev.current_value:.0f}/{ev.threshold_value:.0f} connections, {current_pct:.0f}%). "
            f"Current growth: {trend_str} ({slope_str}). "
            f"Projected to hit limit in {_fmt_hours(horizon)}.{accel} "
            f"Recommend reviewing slow queries or increasing pool size."
        )
    else:
        return (
            f"Database connection usage elevated at {current_pct:.0f}% "
            f"({ev.current_value:.0f}/{ev.threshold_value:.0f}). "
            f"Trend: {trend_str}. If growth continues, limit may be reached in {_fmt_hours(horizon)}."
        )


# ---- MEMORY LEAK ----

def _msg_memory_leak(ev: Evidence, horizon: float, severity: Severity) -> str:
    trend_str = _fmt_trend(ev.trend)
    slope_str = _fmt_slope(ev.trend, " MB")
    accel = _acceleration_note(ev.trend)
    gc_note = (
        " GC cycles are not fully reclaiming memory — classic leak pattern."
        if ev.trend.acceleration > 0 else ""
    )

    if severity == Severity.CRITICAL:
        return (
            f"Memory leak confirmed: {ev.current_value:.0f} MB used ({ev.current_pct_of_threshold:.0f}% of limit). "
            f"Growing at {slope_str} ({trend_str}).{gc_note} "
            f"OOM (out-of-memory) kill estimated in {_fmt_hours(horizon)}. "
            f"Schedule immediate restart or heap dump analysis."
        )
    elif severity == Severity.HIGH:
        return (
            f"Memory leak detected: {ev.current_value:.0f} MB ({ev.current_pct_of_threshold:.0f}% of {ev.threshold_value:.0f} MB limit). "
            f"Baseline creeping upward — {trend_str} ({slope_str}).{accel} "
            f"Estimated time to OOM: {_fmt_hours(horizon)}. Plan a maintenance restart."
        )
    else:
        return (
            f"Memory usage trending upward: {ev.current_value:.0f} MB ({trend_str}). "
            f"Pattern consistent with a slow leak. Monitor closely — "
            f"at this rate, limit ({ev.threshold_value:.0f} MB) reached in {_fmt_hours(horizon)}."
        )


# ---- MEMORY PRESSURE (no leak, just high usage) ----

def _msg_memory_pressure(ev: Evidence, horizon: float, severity: Severity) -> str:
    return (
        f"Memory usage at {ev.current_pct_of_threshold:.0f}% "
        f"({ev.current_value:.0f}/{ev.threshold_value:.0f} MB). "
        f"Current trend: {_fmt_trend(ev.trend)}. "
        f"If sustained, will exceed safe threshold in {_fmt_hours(horizon)}. "
        f"Consider scaling or clearing caches."
    )


# ---- TRAFFIC SPIKE ----

def _msg_traffic_spike(ev: Evidence, horizon: float, severity: Severity) -> str:
    multiplier = ev.current_value / max(ev.extra.get("baseline_rps", ev.threshold_value / 5), 1)
    trend_str = _fmt_trend(ev.trend)

    if severity == Severity.CRITICAL:
        return (
            f"Traffic spike in progress: {ev.current_value:.0f} rps "
            f"({multiplier:.1f}× baseline, {trend_str}). "
            f"Current load will cause cascading failures in downstream services within {_fmt_hours(horizon)}. "
            f"Activate rate limiting and auto-scaling immediately."
        )
    elif severity == Severity.HIGH:
        return (
            f"Traffic surge detected: {ev.current_value:.0f} rps ({multiplier:.1f}× normal baseline). "
            f"Trend: {trend_str}. "
            f"If spike continues, capacity limit will be breached in {_fmt_hours(horizon)}. "
            f"Pre-warm additional instances."
        )
    else:
        return (
            f"Above-normal traffic: {ev.current_value:.0f} rps ({trend_str}). "
            f"Approaching capacity threshold in {_fmt_hours(horizon)} if growth continues."
        )


# ---- LATENCY DEGRADATION ----

def _msg_latency_degradation(ev: Evidence, horizon: float, severity: Severity) -> str:
    trend_str = _fmt_trend(ev.trend)
    endpoint = ev.extra.get("endpoint", "API")
    percentile = ev.extra.get("percentile", "p99")
    accel = _acceleration_note(ev.trend)

    if severity == Severity.CRITICAL:
        return (
            f"{endpoint} {percentile} latency at {ev.current_value:.0f}ms "
            f"({ev.current_pct_of_threshold:.0f}% of {ev.threshold_value:.0f}ms SLA). "
            f"Degradation: {trend_str} ({_fmt_slope(ev.trend, ' ms')}).{accel} "
            f"SLA breach imminent in {_fmt_hours(horizon)}. "
            f"Check downstream service health and database query times."
        )
    elif severity == Severity.HIGH:
        return (
            f"{endpoint} {percentile} latency rising: {ev.current_value:.0f}ms "
            f"({trend_str}, {_fmt_slope(ev.trend, ' ms')}).{accel} "
            f"Will exceed {ev.threshold_value:.0f}ms SLA in {_fmt_hours(horizon)}. "
            f"Investigate slow queries or resource contention."
        )
    else:
        return (
            f"{endpoint} {percentile} latency elevated at {ev.current_value:.0f}ms "
            f"(normal: {ev.extra.get('baseline_ms', 'N/A')}ms). "
            f"Trend: {trend_str}. SLA threshold ({ev.threshold_value:.0f}ms) at risk in {_fmt_hours(horizon)}."
        )


# ---- LATENCY SLA BREACH ----

def _msg_latency_sla(ev: Evidence, horizon: float, severity: Severity) -> str:
    endpoint = ev.extra.get("endpoint", "API")
    percentile = ev.extra.get("percentile", "p99")
    return (
        f"SLA BREACH PREDICTED: {endpoint} {percentile} latency "
        f"({ev.current_value:.0f}ms) will cross {ev.threshold_value:.0f}ms "
        f"in {_fmt_hours(horizon)} based on {_fmt_trend(ev.trend)} growth pattern. "
        f"Users will experience unacceptable response times."
    )


# ---- DISK FILL ----

def _msg_disk_fill(ev: Evidence, horizon: float, severity: Severity) -> str:
    free_gb = ev.extra.get("free_gb", "N/A")
    mount = ev.extra.get("mount", "/")
    trend_str = _fmt_trend(ev.trend)

    if severity == Severity.CRITICAL:
        return (
            f"Disk {mount} at {ev.current_pct_of_threshold:.0f}% capacity "
            f"(~{free_gb:.0f} GB free). "
            f"Filling at {trend_str}. "
            f"Will reach 100% in {_fmt_hours(horizon)} — services writing to disk will crash. "
            f"Purge logs, expand volume, or archive data immediately."
        )
    elif severity == Severity.HIGH:
        return (
            f"Disk {mount} usage at {ev.current_value:.0f}% and growing ({trend_str}). "
            f"Estimated full in {_fmt_hours(horizon)}. "
            f"Schedule log rotation or volume expansion."
        )
    else:
        return (
            f"Disk {mount} filling steadily: {ev.current_value:.0f}% used ({trend_str}). "
            f"At this rate, capacity will be reached in {_fmt_hours(horizon)}. "
            f"Review log retention policies."
        )


# ---- CPU SATURATION ----

def _msg_cpu(ev: Evidence, horizon: float, severity: Severity) -> str:
    trend_str = _fmt_trend(ev.trend)
    return (
        f"CPU load at {ev.current_value:.0f}% and rising ({trend_str}). "
        f"Projected to sustain above {ev.threshold_value:.0f}% (throttling threshold) "
        f"in {_fmt_hours(horizon)}. "
        f"Identify hot processes and consider horizontal scaling."
    )


# ---- ERROR RATE SPIKE ----

def _msg_error_rate(ev: Evidence, horizon: float, severity: Severity) -> str:
    trend_str = _fmt_trend(ev.trend)
    if severity == Severity.CRITICAL:
        return (
            f"Error rate at {ev.current_value:.1f}% and accelerating ({trend_str}). "
            f"Will breach {ev.threshold_value:.0f}% error budget in {_fmt_hours(horizon)}. "
            f"Check recent deployments, upstream dependencies, and database connectivity."
        )
    return (
        f"Error rate elevated: {ev.current_value:.1f}% ({trend_str}). "
        f"Expected to exceed {ev.threshold_value:.0f}% threshold in {_fmt_hours(horizon)}."
    )


# ------------------------------------------------------------------
# Registry: scenario → template function
# ------------------------------------------------------------------

TemplateFunc = Callable[[Evidence, float, Severity], str]

_TEMPLATES: dict[Scenario, TemplateFunc] = {
    Scenario.CONNECTION_POOL_EXHAUSTION: _msg_connection_pool,
    Scenario.MEMORY_LEAK: _msg_memory_leak,
    Scenario.MEMORY_PRESSURE: _msg_memory_pressure,
    Scenario.TRAFFIC_SPIKE: _msg_traffic_spike,
    Scenario.LATENCY_DEGRADATION: _msg_latency_degradation,
    Scenario.LATENCY_SLA_BREACH: _msg_latency_sla,
    Scenario.DISK_FILL: _msg_disk_fill,
    Scenario.CPU_SATURATION: _msg_cpu,
    Scenario.ERROR_RATE_SPIKE: _msg_error_rate,
}


def render_message(
    scenario: Scenario,
    evidence: Evidence,
    horizon_hours: float,
    severity: Severity,
) -> str:
    """
    Returns a specific, human-readable prediction message.
    Falls back to a generic (but still informative) message if no template exists.
    """
    template_fn = _TEMPLATES.get(scenario)
    if template_fn:
        return template_fn(evidence, horizon_hours, severity)

    # Fallback — still specific, just generic pattern
    return (
        f"{evidence.metric_name} at {evidence.current_value:.1f} "
        f"({evidence.current_pct_of_threshold:.0f}% of threshold). "
        f"Trend: {_fmt_trend(evidence.trend)}. "
        f"Threshold breach predicted in {_fmt_hours(horizon_hours)} "
        f"based on current pattern."
    )
