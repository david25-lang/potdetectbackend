from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CNN_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "road_damage.keras"
)

# Alphabetical order matches Keras ImageDataGenerator.flow_from_directory default.
CNN_CLASSES = ["crack", "pothole"]

CNN_INPUT_SIZE = int(os.getenv("CNN_INPUT_SIZE", "224"))


def _configured_model_path() -> Path:
    env_path = os.getenv("CNN_MODEL_PATH")
    return Path(env_path).expanduser().resolve() if env_path else DEFAULT_CNN_MODEL_PATH


class CNNClassifier:
    """Loads a Keras image-classification model once and serves predictions."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path = model_path or _configured_model_path()
        self._model: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._model is not None:
            return

        try:
            from tensorflow.keras.models import load_model  # type: ignore
        except ImportError:
            logger.warning(
                "TensorFlow is not installed. "
                "CNN classification endpoint will return 503. "
                "Run: pip install tensorflow"
            )
            return

        if not self.model_path.exists():
            logger.warning(
                "CNN model not found at %s. "
                "Place road_damage.keras in backend/models/ to enable CNN classification.",
                self.model_path,
            )
            return

        self._model = load_model(str(self.model_path))
        logger.info("CNN model loaded from %s", self.model_path)

    @property
    def available(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify(self, image_bgr: np.ndarray) -> dict:
        """
        Accepts an OpenCV BGR image array, returns classification result.

        Returns:
            {
                "prediction": str,
                "confidence": float,
                "probabilities": {"crack": float, "pothole": float}
            }
        """
        if not self.available:
            raise RuntimeError("CNN model is not loaded.")

        resized = cv2.resize(image_bgr, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype("float32") / 255.0
        batch = np.expand_dims(normalized, axis=0)

        raw: np.ndarray = self._model.predict(batch, verbose=0)[0]

        # Support both binary sigmoid (scalar) and softmax (2-class) outputs.
        if raw.shape == () or (raw.ndim == 1 and len(raw) == 1):
            p_pothole = float(raw.flat[0])
            probs = np.array([1.0 - p_pothole, p_pothole])
        else:
            probs = raw.astype(float)

        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        predicted_class = CNN_CLASSES[class_idx] if class_idx < len(CNN_CLASSES) else str(class_idx)

        probabilities = {
            cls: round(float(p), 6)
            for cls, p in zip(CNN_CLASSES, probs)
        }

        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 6),
            "probabilities": probabilities,
        }
