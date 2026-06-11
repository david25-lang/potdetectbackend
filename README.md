# YOLO FastAPI Backend

Production-ready FastAPI backend for serving the extracted YOLO model at `extracted_model/best.pt`.

## Project Structure

```text
backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   └── utils.py
├── extracted_model/
│   └── best.pt
├── requirements.txt
└── README.md
```

## Setup

From this repository root:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The model is loaded once during application startup from:

```text
backend/extracted_model/best.pt
```

To use a different model path without changing code:

```bash
YOLO_MODEL_PATH=/absolute/path/to/best.pt uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Endpoints

### Health Check

```http
GET /
```

Response:

```json
{ "message": "YOLO API is running" }
```

### Predict

```http
POST /predict/
Content-Type: multipart/form-data
```

Form field:

```text
image: image file
```

Example:

```bash
curl -X POST "http://127.0.0.1:8000/predict/" \
  -F "image=@/path/to/test-image.jpg"
```

Optional confidence threshold:

```bash
curl -X POST "http://127.0.0.1:8000/predict/?confidence=0.4" \
  -F "image=@/path/to/test-image.jpg"
```

Response:

```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "pothole",
      "confidence": 0.87,
      "bbox": [120.5, 95.2, 240.1, 180.9]
    }
  ]
}
```

### Predict With Annotated Image

```http
POST /predict-annotated/
Content-Type: multipart/form-data
```

Example:

```bash
curl -X POST "http://127.0.0.1:8000/predict-annotated/" \
  -F "image=@/path/to/test-image.jpg"
```

Response includes the same `detections` array plus a base64-encoded JPEG:

```json
{
  "detections": [],
  "annotated_image": "/9j/4AAQSkZJRgABAQ...",
  "image_format": "jpg"
}
```

In Postman, create a `POST` request, choose `Body > form-data`, add a key named `image`, change its type to `File`, select an image, and send it to either prediction endpoint.

## Notes

- The model is not retrained or modified.
- Uploads are decoded with OpenCV and passed directly to Ultralytics YOLO.
- Default upload size limit is 10 MB. Override it with `MAX_IMAGE_BYTES`.
- CORS is enabled for all origins, methods, and headers for frontend integration.
# potdetectbackend
