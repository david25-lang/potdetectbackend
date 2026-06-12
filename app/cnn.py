from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_ONNX_PATH = _MODELS_DIR / "best_cnn.onnx"
DEFAULT_KERAS_PATH = _MODELS_DIR / "best_cnn.keras"

CNN_CLASSES = ["crack", "pothole"]
CNN_INPUT_SIZE = int(os.getenv("CNN_INPUT_SIZE", "224"))


def _configured_path(env_var: str, default: Path) -> Path:
    val = os.getenv(env_var)
    return Path(val).expanduser().resolve() if val else default


class CNNClassifier:
    """
    Loads the road-damage CNN.
    Prefers the ONNX model (onnxruntime, ~20 MB RAM) when available.
    Falls back to the Keras model (TensorFlow) for local development.
    """

    def __init__(self) -> None:
        self._session: Any = None   # onnxruntime session
        self._keras_model: Any = None
        self._input_name: str = ""

    def load(self) -> None:
        if self._session is not None or self._keras_model is not None:
            return

        onnx_path = _configured_path("CNN_MODEL_PATH", DEFAULT_ONNX_PATH)
        keras_path = _configured_path("CNN_KERAS_PATH", DEFAULT_KERAS_PATH)

        # --- Try ONNX first (production / Render) ---
        if onnx_path.exists():
            try:
                import onnxruntime as ort  # type: ignore
                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = 1
                self._session = ort.InferenceSession(
                    str(onnx_path),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self._input_name = self._session.get_inputs()[0].name
                logger.info("CNN loaded via onnxruntime from %s", onnx_path)
                return
            except Exception as exc:
                logger.warning("onnxruntime load failed (%s), trying Keras fallback.", exc)

        # --- Fall back to Keras / TensorFlow (local dev) ---
        if keras_path.exists():
            try:
                import tensorflow as tf  # type: ignore
                self._keras_model = tf.keras.models.load_model(
                    str(keras_path), compile=False
                )
                logger.info("CNN loaded via TensorFlow/Keras from %s", keras_path)
                return
            except Exception as exc:
                logger.warning("Keras load failed: %s", exc)

        logger.warning(
            "CNN model not found. Tried ONNX at %s and Keras at %s. "
            "For local dev place best_cnn.keras in backend/models/. "
            "For production run convert_model.py to generate best_cnn.onnx.",
            onnx_path, keras_path,
        )

    @property
    def available(self) -> bool:
        return self._session is not None or self._keras_model is not None

    def classify(self, image_bgr: np.ndarray) -> dict:
        if not self.available:
            raise RuntimeError("CNN model is not loaded.")

        resized = cv2.resize(image_bgr, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
        # ResNet50 backbone expects caffe-style preprocessing:
        # subtract ImageNet BGR channel means, keep 0-255 scale (no /255).
        x = resized.astype("float32")   # already BGR from OpenCV
        x[..., 0] -= 103.939           # B mean
        x[..., 1] -= 116.779           # G mean
        x[..., 2] -= 123.68            # R mean
        batch = x[np.newaxis]

        if self._session is not None:
            raw = self._session.run(None, {self._input_name: batch})[0][0]
        else:
            raw = self._keras_model.predict(batch, verbose=0)[0]

        # Binary sigmoid output: single value → pothole probability
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
