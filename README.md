# AI Predictive Monitor

A real-time infrastructure monitoring system that predicts failures **2–24 hours before they occur**, with specific, actionable predictions — not generic alerts.

> "Database connection pool growing toward limit (78/100, 78%). Current growth: +34% over 2h (+0.8 conn/min). Projected to hit limit in ~6h. Recommend reviewing slow queries or increasing pool size."

Instead of: *"Anomaly detected."*

---

## Screenshot

> *Dashboard screenshot — add after first run at `http://localhost:8000`*

---

## Key Features

- **Predictive, not reactive** — forecasts problems 2–24 hours ahead using time-series regression and ML anomaly detection
- **Human-readable predictions** — every alert answers WHAT will happen, WHEN, and WHY (based on what evidence)
- **Ensemble ML** — combines IsolationForest (unsupervised anomaly detection) and Facebook Prophet (time-series forecasting) into a single calibrated 0–1 score
- **5 simulated failure scenarios** — connection pool exhaustion, memory leak, traffic spike, latency degradation, disk fill
- **Real-time dashboard** — WebSocket-powered dark ops interface with live countdowns, severity-coded prediction cards, and automatic reconnect
- **Zero external dependencies to run** — SQLite backend, synthetic data generator, single `uvicorn` command

---

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌───────────────────┐    ┌──────────────────┐
│  Data Pipeline  │───▶│   ML Engine      │───▶│ Prediction Engine │───▶│    Dashboard     │
│                 │    │                  │    │                   │    │                  │
│ MetricSimulator │    │ IsolationForest  │    │ HorizonEstimator  │    │ FastAPI + WS     │
│ SQLite storage  │    │ Prophet forecast │    │ ConfidenceScorer  │    │ HTML/JS frontend │
│ asyncio queue   │    │ Ensemble scoring │    │ Message templates │    │ Live countdowns  │
└─────────────────┘    └──────────────────┘    └───────────────────┘    └──────────────────┘
```

### Phase 1 — Data Pipeline (`data_pipeline/`)
Simulates realistic infrastructure metrics using statistical patterns (exponential saturation for connection exhaustion, sawtooth for memory leaks, Poisson spikes for traffic). Stores time-series data in SQLite via an abstract `TimeSeriesStore` interface (InfluxDB-ready).

### Phase 2 — ML Engine (`ml_engine/`)
Trains two complementary models per metric on historical data:
- **IsolationForest** — detects anomalous feature windows (rolling stats, lag features, rate-of-change)
- **Prophet** — forecasts next 24h and scores actual values against predicted confidence intervals

Ensemble: `0.4 × IsolationForest + 0.6 × Prophet` → normalized 0–1 anomaly score.

### Phase 3 — Prediction Engine (`prediction_engine/`)
Converts anomaly scores into predictions with time horizons:
- **HorizonEstimator** — fits linear + exponential regression on the recent metric window, picks the better fit, extrapolates to threshold crossing time
- **ConfidenceScorer** — combines anomaly score, trend consistency, and rate of acceleration into a calibrated confidence percentage
- **Message templates** — 9 scenario-specific templates, each producing a WHAT/WHEN/WHY message

### Phase 4 — Dashboard (`dashboard/`)
FastAPI server with WebSocket broadcast. The demo engine replays 8 hours of simulated data on startup (training on the first 4 hours), then streams new predictions every 10 seconds. Frontend is plain HTML/JS — no framework, no build step.

---

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Backend API | FastAPI + Uvicorn | 0.115 / 0.32 |
| ML Forecasting | Facebook Prophet | 1.1.6 |
| ML Anomaly Detection | scikit-learn IsolationForest | 1.6.0 |
| Feature Engineering | pandas + numpy | 2.2.3 / 2.2.0 |
| Data Validation | Pydantic v2 | 2.10.3 |
| Storage (default) | SQLite (aiosqlite) | built-in |
| Storage (optional) | InfluxDB | 1.47 client |
| Model Serialization | joblib | 1.4.2 |
| Async I/O | asyncio + aiofiles | stdlib |
| Frontend | Vanilla HTML/JS/CSS | — |
| Testing | pytest + pytest-asyncio | 8.3 / 0.24 |

---

## How to Run

### Prerequisites

- Python 3.11+
- pip

### Install

```bash
git clone https://github.com/your-username/ai-predictive-monitor.git
cd ai-predictive-monitor
pip install -r requirements.txt
```

### Start the dashboard

```bash
python -m uvicorn dashboard.api.main:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000).

On first run, the system will:
1. Generate ~8 hours of synthetic metric data (connection spikes, memory leaks, traffic surges)
2. Train IsolationForest and Prophet models on the first 4 hours
3. Begin scoring and predicting — dashboard populates within ~15–20 seconds

### Alternative: CLI simulation

```bash
# Simulate 12 hours of data at 60× speed
python main.py simulate --hours 12 --speed 60

# Stream live metrics (uses real system metrics via psutil)
python main.py live
```

---

## Example Predictions

These are actual outputs from the prediction engine, not hand-written strings:

**HIGH severity — Connection pool approaching limit:**
```
Database connection pool growing toward limit (78/100 connections, 78%).
Current growth: +34% over 2h (+0.8 conn/min).
Projected to hit limit in ~6h. Growth is accelerating.
Recommend reviewing slow queries or increasing pool size.
```

**CRITICAL severity — Memory leak confirmed:**
```
Memory leak confirmed: 1847 MB used (92% of limit).
Growing at +4.2 MB/min (+31% over 2h). GC cycles are not fully reclaiming
memory — classic leak pattern. OOM (out-of-memory) kill estimated in ~45 min.
Schedule immediate restart or heap dump analysis.
```

**HIGH severity — Disk filling:**
```
Disk / usage at 81% and growing (+18% over 4h).
Estimated full in ~8h.
Schedule log rotation or volume expansion.
```

---

## How It Works

### Anomaly Detection

Each metric is scored by two models and combined via weighted ensemble:

1. **IsolationForest** builds a 20-feature vector per data point (rolling mean/std over 5/10/30-point windows, lag features, rate-of-change, time-of-day encoding). The sklearn `decision_function` score is normalized to [0, 1].

2. **Prophet** fits on historical data and forecasts the next 24 hours with a 95% confidence interval. Points outside the interval are scored by normalized breach distance, then passed through a sigmoid to stay in [0, 1].

3. **Ensemble**: `score = (0.4 × IF_score + 0.6 × Prophet_score)`. Prophet is weighted higher because it is time-aware and provides context about whether the current value is anomalous *for this time of day*.

### Horizon Estimation

When a metric's anomaly score crosses threshold (default 0.65), the `HorizonEstimator` fits both linear and exponential regression on the last 30 minutes of data, selects the model with lower RMSE, and extrapolates to the configured threshold crossing time. The result is a float: hours until predicted event.

### Confidence Scoring

`confidence = anomaly_score × trend_consistency_factor × acceleration_bonus`

Trend consistency is high when multiple regression models agree on the direction. Acceleration (second derivative) pushes confidence up if the metric is worsening faster than linearly. Final confidence is clamped to [0.40, 0.97] to avoid overconfident or useless outputs.

### Severity Mapping

| Confidence + Horizon | Severity |
|---------------------|----------|
| ≥ 0.85 or < 2h | CRITICAL |
| ≥ 0.70 or < 6h | HIGH |
| ≥ 0.55 or < 12h | MEDIUM |
| everything else | LOW |

---

## Project Structure

```
ai-predictive-monitor/
├── data_pipeline/
│   ├── simulator/metric_simulator.py   # 5 failure scenario generators
│   ├── storage/sqlite_backend.py       # async SQLite time-series store
│   └── pipeline.py                     # asyncio queue + batch writer
├── ml_engine/
│   ├── models/isolation_forest.py      # IsolationForest wrapper
│   ├── models/prophet_model.py         # Prophet wrapper + forecast
│   ├── features/feature_engineering.py # 20+ rolling/lag/time features
│   ├── scoring/anomaly_scorer.py       # weighted ensemble combiner
│   └── training/trainer.py             # scheduled re-training loop
├── prediction_engine/
│   ├── horizon_estimator.py            # linear + exponential regression
│   ├── confidence_scorer.py            # calibrated confidence %
│   ├── predictor.py                    # orchestrates scores → predictions
│   └── templates/prediction_templates.py  # 9 scenario message templates
├── dashboard/
│   ├── api/main.py                     # FastAPI app + lifespan
│   ├── api/demo_engine.py              # simulation replay + broadcast loop
│   ├── api/websocket.py                # WebSocket connection manager
│   └── frontend/                       # index.html, app.js, styles.css
├── config/config.yaml                  # all tunable parameters
├── main.py                             # CLI entry point
└── requirements.txt
```

---

## Configuration

All thresholds, model parameters, and simulation settings live in [config/config.yaml](config/config.yaml). Key sections:

```yaml
ml:
  anomaly_threshold: 0.65       # score above this triggers prediction engine
  isolation_forest:
    contamination: 0.05
  prophet:
    interval_width: 0.95
    forecast_horizon_hours: 24

prediction:
  min_confidence: 0.40
  min_horizon_hours: 0.25
  max_horizon_hours: 24.0
```

---

## License

MIT
