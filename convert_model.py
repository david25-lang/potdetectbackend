"""
Build-time script: converts best_cnn.keras → best_cnn.onnx
Run once during Render build before installing the runtime requirements.
"""
from __future__ import annotations

import sys
from pathlib import Path

KERAS_PATH = Path("models/best_cnn.keras")
ONNX_PATH = Path("models/best_cnn.onnx")
INPUT_SIZE = 224


def main() -> None:
    if ONNX_PATH.exists():
        print(f"ONNX model already exists at {ONNX_PATH}, skipping conversion.")
        return

    if not KERAS_PATH.exists():
        print(f"ERROR: Keras model not found at {KERAS_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Keras model from {KERAS_PATH} …")
    import tensorflow as tf  # noqa: PLC0415
    import tf2onnx  # noqa: PLC0415

    model = tf.keras.models.load_model(str(KERAS_PATH))

    input_sig = [tf.TensorSpec(shape=(None, INPUT_SIZE, INPUT_SIZE, 3), dtype=tf.float32, name="input")]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_sig, opset=17)

    import onnx  # noqa: PLC0415
    onnx.save(onnx_model, str(ONNX_PATH))
    print(f"Saved ONNX model to {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
