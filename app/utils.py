from __future__ import annotations

import base64
import os
from typing import Any

import cv2
import numpy as np
from fastapi import UploadFile


MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))

ACCEPTED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}


async def read_image_upload(file: UploadFile) -> np.ndarray:
    """Read an uploaded image into an OpenCV BGR numpy array.

    Accepts JPEG (camera captures), PNG, WebP, BMP, and TIFF.
    """
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type and content_type not in ACCEPTED_MIME_TYPES:
        raise ValueError(
            f"Unsupported file type '{content_type}'. "
            "Accepted formats: JPEG, PNG, WebP, BMP, TIFF."
        )

    contents = await file.read()

    if not contents:
        raise ValueError("Uploaded image is empty.")

    if len(contents) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Uploaded image is too large. Limit is {MAX_IMAGE_BYTES // (1024 * 1024)} MB."
        )

    image_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(
            "Could not decode the image. "
            "Make sure the file is a valid JPEG, PNG, WebP, BMP, or TIFF."
        )

    return image


def detections_from_result(result: Any, class_names: Any) -> list[dict[str, Any]]:
    """Convert an Ultralytics result object into API-safe JSON."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)

    detections: list[dict[str, Any]] = []
    for bbox, confidence, class_id in zip(xyxy, confidences, class_ids):
        detections.append(
            {
                "class_id": int(class_id),
                "class_name": _class_name(class_names, int(class_id)),
                "confidence": round(float(confidence), 6),
                "bbox": [round(float(value), 2) for value in bbox.tolist()],
            }
        )

    return detections


def encode_image_base64(image: np.ndarray, extension: str = ".jpg") -> str:
    """Encode an image array as a base64 string."""
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise ValueError("Annotated image could not be encoded.")

    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _class_name(class_names: Any, class_id: int) -> str:
    if isinstance(class_names, dict):
        return str(class_names.get(class_id, class_names.get(str(class_id), class_id)))

    if isinstance(class_names, (list, tuple)) and 0 <= class_id < len(class_names):
        return str(class_names[class_id])

    return str(class_id)
