from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from ultralytics import YOLO

from .utils import detections_from_result


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "extracted_model" / "best.pt"


def configured_model_path() -> Path:
    model_path = os.getenv("YOLO_MODEL_PATH")
    if model_path:
        return Path(model_path).expanduser().resolve()

    return DEFAULT_MODEL_PATH


class YOLODetector:
    """Small wrapper that keeps the YOLO model loaded for the process lifetime."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path = model_path or configured_model_path()
        self.model: Optional[YOLO] = None
        self.class_names: Any = {}

    def load(self) -> None:
        if self.model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found at {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.class_names = getattr(self.model, "names", {}) or {}

    def predict(self, image: np.ndarray, confidence: Optional[float] = None) -> list[dict]:
        result = self._run_inference(image, confidence)
        return detections_from_result(result, getattr(result, "names", self.class_names))

    def predict_with_annotation(
        self, image: np.ndarray, confidence: Optional[float] = None, iou: Optional[float] = None
    ) -> tuple[list[dict], np.ndarray]:
        result = self._run_inference(image, confidence, iou)
        detections = detections_from_result(
            result, getattr(result, "names", self.class_names)
        )
        annotated_image = result.plot()

        return detections, annotated_image

    def _run_inference(self, image: np.ndarray, confidence: Optional[float], iou: Optional[float] = None) -> Any:
        if self.model is None:
            self.load()

        options: dict[str, Any] = {"verbose": False}
        if confidence is not None:
            options["conf"] = confidence
        if iou is not None:
            options["iou"] = iou

        results = self.model.predict(source=image, **options)
        return results[0]
