from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

DEFAULT_YOLO_ONNX = Path(__file__).resolve().parents[1] / "extracted_model" / "best.onnx"
DEFAULT_YOLO_PT   = Path(__file__).resolve().parents[1] / "extracted_model" / "best.pt"
DEFAULT_META      = Path(__file__).resolve().parents[1] / "extracted_model" / "class_names.json"

_FALLBACK_NAMES = {0: "pothole", 1: "crack"}


def _configured_path(env_var: str, default: Path) -> Path:
    val = os.getenv(env_var)
    return Path(val).expanduser().resolve() if val else default


class YOLODetector:
    """
    YOLO inference via onnxruntime (no torch/ultralytics at runtime).
    Falls back to ultralytics .pt for local development when ONNX is absent.
    """

    def __init__(self) -> None:
        self._session = None
        self._input_name = ""
        self._input_hw = (640, 640)
        self._class_names: dict = {}
        self._ultralytics_model = None

    def load(self) -> None:
        if self._session is not None or self._ultralytics_model is not None:
            return

        onnx_path = _configured_path("YOLO_MODEL_PATH", DEFAULT_YOLO_ONNX)
        meta_path = _configured_path("YOLO_META_PATH", DEFAULT_META)

        # Load class names
        if meta_path.exists():
            raw = json.loads(meta_path.read_text())
            self._class_names = {int(k): v for k, v in raw.items()}
        else:
            self._class_names = _FALLBACK_NAMES

        # --- ONNX (production) ---
        if onnx_path.exists():
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
            shape = self._session.get_inputs()[0].shape
            self._input_hw = (int(shape[2]), int(shape[3]))  # H, W
            return

        # --- Ultralytics fallback (local dev) ---
        pt_path = _configured_path("YOLO_PT_PATH", DEFAULT_YOLO_PT)
        if pt_path.exists():
            try:
                from ultralytics import YOLO  # type: ignore
                self._ultralytics_model = YOLO(str(pt_path))
                self._class_names = getattr(self._ultralytics_model, "names", _FALLBACK_NAMES)
            except Exception as exc:
                raise RuntimeError(f"Could not load YOLO model: {exc}") from exc
            return

        raise FileNotFoundError(
            f"No YOLO model found. Tried ONNX at {onnx_path} and .pt at {pt_path}. "
            "Run convert_model.py to generate the ONNX file."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image: np.ndarray, confidence: Optional[float] = None) -> list[dict]:
        conf = confidence if confidence is not None else 0.25
        if self._session is not None:
            return self._infer_onnx(image, conf, iou=0.45)
        return self._infer_ultralytics(image, conf)

    def predict_with_annotation(
        self,
        image: np.ndarray,
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> tuple[list[dict], np.ndarray]:
        conf = confidence if confidence is not None else 0.25
        nms_iou = iou if iou is not None else 0.45

        if self._session is not None:
            detections = self._infer_onnx(image, conf, nms_iou)
            annotated = self._draw(image.copy(), detections)
            return detections, annotated

        detections = self._infer_ultralytics(image, conf, nms_iou)
        annotated = self._draw(image.copy(), detections)
        return detections, annotated

    # ------------------------------------------------------------------
    # ONNX inference + post-processing
    # ------------------------------------------------------------------

    def _infer_onnx(self, image: np.ndarray, conf_thr: float, iou: float) -> list[dict]:
        ih, iw = self._input_hw
        orig_h, orig_w = image.shape[:2]

        resized = cv2.resize(image, (iw, ih))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        blob = rgb.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

        raw = self._session.run(None, {self._input_name: blob})[0]  # (1, 4+classes, anchors)
        preds = raw[0].T  # (anchors, 4+classes)

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        mask = confidences >= conf_thr
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        # cx, cy, w, h → x1, y1, x2, y2 scaled to original image
        x1 = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2) * orig_w / iw
        y1 = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2) * orig_h / ih
        x2 = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2) * orig_w / iw
        y2 = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2) * orig_h / ih

        boxes_list = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(boxes_list, confidences.tolist(), conf_thr, iou)

        detections = []
        for idx in (indices.flatten() if hasattr(indices, "flatten") else indices):
            i = int(idx)
            detections.append({
                "class_id": int(class_ids[i]),
                "class_name": self._class_names.get(int(class_ids[i]), str(class_ids[i])),
                "confidence": round(float(confidences[i]), 6),
                "bbox": [round(float(x1[i]), 2), round(float(y1[i]), 2),
                         round(float(x2[i]), 2), round(float(y2[i]), 2)],
            })
        return detections

    # ------------------------------------------------------------------
    # Ultralytics fallback inference (local dev only)
    # ------------------------------------------------------------------

    def _infer_ultralytics(
        self, image: np.ndarray, conf: float, iou: float = 0.45
    ) -> list[dict]:
        results = self._ultralytics_model.predict(source=image, conf=conf, iou=iou, verbose=False)
        result = results[0]
        names = getattr(result, "names", self._class_names)
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        detections = []
        for bbox, confidence, cls_id in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy().astype(int),
        ):
            detections.append({
                "class_id": int(cls_id),
                "class_name": str(names.get(int(cls_id), cls_id)),
                "confidence": round(float(confidence), 6),
                "bbox": [round(float(v), 2) for v in bbox.tolist()],
            })
        return detections

    # ------------------------------------------------------------------
    # Annotation (cv2 — no ultralytics needed)
    # ------------------------------------------------------------------

    _COLORS = {0: (0, 100, 255), 1: (0, 220, 100)}  # pothole=orange-ish, crack=green

    def _draw(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det["bbox"])
            color = self._COLORS.get(det["class_id"], (200, 200, 200))
            label = f"{det['class_name']} {det['confidence']:.0%}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(image, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return image
