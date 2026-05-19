"""
CLI entry point for the AI Predictive Monitor.

Usage:
    python main.py simulate          # Run all failure scenarios, store to SQLite
    python main.py simulate --hours 6 --speed 120
    python main.py live              # Collect real system metrics (Phase 1 live mode)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml

# ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline.pipeline import DataPipeline
from data_pipeline.storage.sqlite_backend import SQLiteStore


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def _load_config(path: str = "config/config.yaml") -> dict:
    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


async def cmd_simulate(args, config: dict) -> None:
    sim_cfg = config.get("simulator", {})
    storage_cfg = config.get("storage", {})

    db_path = storage_cfg.get("sqlite", {}).get("path", "data/metrics.db")
    store = SQLiteStore(db_path=db_path)

    pipeline = DataPipeline(store=store)

    hours = args.hours or 12
    speed = args.speed or sim_cfg.get("speed_multiplier", 60)

    print(f"[simulate] Running {hours}h of failure scenarios at {speed}x speed...")
    print(f"[simulate] Storing to: {db_path}")
    print(f"[simulate] Press Ctrl+C to stop early.\n")

    try:
        await pipeline.run_simulation(duration_hours=hours, speed_multiplier=speed)
        await pipeline.flush()
        metrics = await store.list_metrics()
        print(f"\n[simulate] Done. Stored metrics: {metrics}")
    except KeyboardInterrupt:
        print("\n[simulate] Interrupted. Flushing remaining data...")
        await pipeline.flush()
    finally:
        await store.close()


async def cmd_live(args, config: dict) -> None:
    """Collect real system metrics every 10 seconds."""
    import time
    from data_pipeline.collectors.system_metrics import collect_system_metrics
    from data_pipeline.collectors.app_metrics import collect_app_metrics
    from data_pipeline.collectors.db_metrics import collect_db_metrics

    storage_cfg = config.get("storage", {})
    db_path = storage_cfg.get("sqlite", {}).get("path", "data/metrics.db")
    store = SQLiteStore(db_path=db_path)
    pipeline = DataPipeline(store=store)

    interval = config.get("pipeline", {}).get("collection_interval_seconds", 10)
    print(f"[live] Collecting system metrics every {interval}s. Ctrl+C to stop.")

    try:
        while True:
            points = collect_system_metrics() + collect_app_metrics() + collect_db_metrics()
            await pipeline.ingest_points(points)
            await pipeline.flush()
            print(f"  Collected {len(points)} points", end="\r")
            await asyncio.sleep(interval)
    except KeyboardInterrupt:
        print("\n[live] Stopped.")
    finally:
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Predictive Monitor — Phase 1 Data Pipeline"
    )
    sub = parser.add_subparsers(dest="command")

    sim = sub.add_parser("simulate", help="Run synthetic failure scenario simulation")
    sim.add_argument("--hours", type=float, default=None, help="Duration in hours (default: 12)")
    sim.add_argument("--speed", type=int, default=None, help="Speed multiplier (default: 60)")
    sim.add_argument("--config", default="config/config.yaml", help="Config file path")

    live = sub.add_parser("live", help="Collect real system metrics")
    live.add_argument("--config", default="config/config.yaml", help="Config file path")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    config = _load_config(getattr(args, "config", "config/config.yaml"))
    _setup_logging(config.get("app", {}).get("log_level", "INFO"))

    if args.command == "simulate":
        asyncio.run(cmd_simulate(args, config))
    elif args.command == "live":
        asyncio.run(cmd_live(args, config))


if __name__ == "__main__":
    main()
