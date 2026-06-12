from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CNN_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "road_damage.onnx"
)

CNN_CLASSES = ["crack", "pothole"]
CNN_INPUT_SIZE = int(os.getenv("CNN_INPUT_SIZE", "224"))


def _configured_model_path() -> Path:
    env_path = os.getenv("CNN_MODEL_PATH")
    return Path(env_path).expanduser().resolve() if env_path else DEFAULT_CNN_MODEL_PATH


class CNNClassifier:
    """Runs the road-damage classifier using ONNX Runtime (~20 MB RAM vs ~600 MB for TensorFlow)."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path = model_path or _configured_model_path()
        self._session: Any = None

    def load(self) -> None:
        if self._session is not None:
            return

        try:
            import onnxruntime as ort  # type: ignore
        except ImportError:
            logger.warning(
                "onnxruntime is not installed. CNN classification will return 503."
            )
            return

        if not self.model_path.exists():
            logger.warning(
                "ONNX model not found at %s. "
                "Run the build step (convert_model.py) to generate it.",
                self.model_path,
            )
            return

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info("CNN ONNX model loaded from %s", self.model_path)

    @property
    def available(self) -> bool:
        return self._session is not None

    def classify(self, image_bgr: np.ndarray) -> dict:
        if not self.available:
            raise RuntimeError("CNN model is not loaded.")

        resized = cv2.resize(image_bgr, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        batch = (rgb.astype("float32") / 255.0)[np.newaxis]

        raw = self._session.run(None, {self._input_name: batch})[0][0]

        # Support both binary sigmoid (scalar) and softmax (2-class) outputs.
        if raw.shape == () or (raw.ndim == 1 and len(raw) == 1):
            p_pothole = float(raw.flat[0])
            probs = np.array([1.0 - p_pothole, p_pothole])
        else:
            probs = raw.astype(float)

        class_idx = int(np.argmax(probs))
        predicted_class = CNN_CLASSES[class_idx] if class_idx < len(CNN_CLASSES) else str(class_idx)

        return {
            "prediction": predicted_class,
            "confidence": round(float(probs[class_idx]), 6),
            "probabilities": {
                cls: round(float(p), 6) for cls, p in zip(CNN_CLASSES, probs)
            },
        }
