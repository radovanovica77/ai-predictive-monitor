"""
Manages trained model instances keyed by metric name.
Persists models to disk with joblib so they survive restarts.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import joblib

if TYPE_CHECKING:
    from ml_engine.models.base_model import BaseAnomalyModel

logger = logging.getLogger(__name__)

_MODELS_DIR = Path("data/models")


class ModelRegistry:
    """
    In-memory store of fitted models, backed by joblib serialization.
    Key: "{metric_name}:{model_name}"
    """

    def __init__(self, models_dir: str | Path = _MODELS_DIR):
        self._dir = Path(models_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, "BaseAnomalyModel"] = {}

    def _key(self, metric_name: str, model_name: str) -> str:
        return f"{metric_name}:{model_name}"

    def _path(self, key: str) -> Path:
        safe = key.replace(".", "_").replace(":", "__")
        return self._dir / f"{safe}.joblib"

    def save(self, metric_name: str, model: "BaseAnomalyModel") -> None:
        key = self._key(metric_name, model.name)
        self._cache[key] = model
        path = self._path(key)
        joblib.dump(model, path)
        logger.debug("Saved model %s → %s", key, path)

    def load(self, metric_name: str, model_name: str) -> "BaseAnomalyModel | None":
        key = self._key(metric_name, model_name)
        if key in self._cache:
            return self._cache[key]
        path = self._path(key)
        if path.exists():
            model = joblib.load(path)
            self._cache[key] = model
            logger.debug("Loaded model %s from %s", key, path)
            return model
        return None

    def get(self, metric_name: str, model_name: str) -> "BaseAnomalyModel | None":
        return self.load(metric_name, model_name)

    def all_for_metric(self, metric_name: str) -> list["BaseAnomalyModel"]:
        """Return all fitted models available for a given metric."""
        models = []
        for path in self._dir.glob(f"{metric_name.replace('.', '_')}__*.joblib"):
            model = joblib.load(path)
            models.append(model)
        return models

    def list_trained(self) -> list[str]:
        return [p.stem for p in self._dir.glob("*.joblib")]
