"""
Build-time script: converts both models to ONNX so the runtime needs only onnxruntime.
Run during Render build BEFORE installing requirements.txt.

  pip install tensorflow-cpu tf2onnx onnx ultralytics && python convert_model.py && pip install -r requirements.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CNN_KERAS  = Path("models/best_cnn.keras")
CNN_ONNX   = Path("models/best_cnn.onnx")
YOLO_PT    = Path("extracted_model/best.pt")
YOLO_ONNX  = Path("extracted_model/best.onnx")
YOLO_META  = Path("extracted_model/class_names.json")
CNN_SIZE   = 224


def convert_cnn() -> None:
    if CNN_ONNX.exists():
        print(f"CNN ONNX already exists at {CNN_ONNX}, skipping.")
        return
    if not CNN_KERAS.exists():
        print(f"ERROR: {CNN_KERAS} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Converting CNN {CNN_KERAS} → {CNN_ONNX} …")
    import tensorflow as tf   # noqa: PLC0415
    import tf2onnx             # noqa: PLC0415
    import onnx                # noqa: PLC0415

    model = tf.keras.models.load_model(str(CNN_KERAS), compile=False)
    sig = [tf.TensorSpec(shape=(None, CNN_SIZE, CNN_SIZE, 3), dtype=tf.float32, name="input")]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=sig, opset=17)
    onnx.save(onnx_model, str(CNN_ONNX))
    print(f"CNN saved → {CNN_ONNX} ({CNN_ONNX.stat().st_size / 1e6:.1f} MB)")


def convert_yolo() -> None:
    if YOLO_ONNX.exists():
        print(f"YOLO ONNX already exists at {YOLO_ONNX}, skipping.")
        return
    if not YOLO_PT.exists():
        print(f"ERROR: {YOLO_PT} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Converting YOLO {YOLO_PT} → {YOLO_ONNX} …")
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(str(YOLO_PT))
    YOLO_META.write_text(json.dumps(model.names))
    model.export(format="onnx", imgsz=640, simplify=True, opset=17)

    exported = YOLO_PT.with_suffix(".onnx")
    if exported != YOLO_ONNX:
        exported.rename(YOLO_ONNX)
    print(f"YOLO saved → {YOLO_ONNX} ({YOLO_ONNX.stat().st_size / 1e6:.1f} MB)")
    print(f"Class names saved → {YOLO_META}")


if __name__ == "__main__":
    convert_cnn()
    convert_yolo()
    print("All conversions done.")
