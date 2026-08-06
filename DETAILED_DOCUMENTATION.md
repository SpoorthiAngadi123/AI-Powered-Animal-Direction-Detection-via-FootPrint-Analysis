### AI-Powered Animal Footprint Direction Detection

**Detailed Technical Documentation**  
_Last updated: 20 Nov 2025_

This document complements `README.md` and `QUICKSTART.md` by walking through every module, function, and HTTP route in the project. It explains how the backend, frontend, and training utilities interact so you can extend, debug, or audit the system with confidence.

---

## 1. System Overview

| Layer           | Location                               | Responsibilities                                                                       |
| --------------- | -------------------------------------- | -------------------------------------------------------------------------------------- |
| Frontend UI     | `frontend/src`                         | React/Vite single-page app with Home (prediction) and Metrics (telemetry) views        |
| REST API        | `api.py`                               | Serves YOLOv8 classification results, metrics, prediction history, and static assets   |
| Model Training  | `train_yolov8.py` + `data/`            | CLI pipeline for dataset verification, training, validation, visualization, and export |
| Assets & Output | `static/`, `runs/`, `predictions.json` | Overlay images, confusion matrix, YOLO training artifacts, persisted history           |

High‑level flow:

1. User uploads an image from the Home page.
2. Frontend sends `POST /predict`.
3. FastAPI loads YOLOv8 weights (singleton), preprocesses the image, performs inference, and returns probabilities plus a generated overlay path.
4. Prediction summary is stored in `predictions.json`.
5. Metrics page polls `/metrics` and `/predictions` to display validation accuracy, confusion matrix, and latest requests.

---

## 2. Backend (`api.py`)

### 2.1 Global Configuration

| Symbol                       | Description                                                               |
| ---------------------------- | ------------------------------------------------------------------------- |
| `API_BASE` (frontend only)   | Default `http://127.0.0.1:8001`; override via `VITE_API_BASE`.            |
| `MODEL_PATH`                 | `runs/classify/train/weights/best.pt`; YOLOv8 weights expected here.      |
| `STATIC_DIR` / `OVERLAY_DIR` | Folders for static assets and generated overlays.                         |
| `PREDICTIONS_DB`             | JSON file storing the latest 100 predictions.                             |
| `_model`, `_model_version`   | Lazy-loaded YOLO model reference and metadata string returned to clients. |

### 2.2 Helper Functions

1. `get_model()`

   - Implements a singleton loader for YOLO weights.
   - Validates `MODEL_PATH` and prints available class names on first load.
   - Subsequent calls reuse the in-memory model to keep inference fast.

2. `load_predictions()` / `save_predictions(predictions: List[dict])`

   - Serialize to/from `predictions.json`.
   - Gracefully handle missing/invalid files by returning an empty list.

3. `create_overlay_image(image_path, direction, confidence, all_probabilities=None)`
   - Reads the temp image (OpenCV with Pillow fallback).
   - Resizes large inputs to max 800px to keep overlays lightweight.
   - Builds a text box summarizing sorted probabilities (top line in green).
   - Saves annotated image in `static/overlays` and returns `/static/overlays/<uuid>.jpg`.

### 2.3 Routes

| Route          | Method | Summary                                                                    |
| -------------- | ------ | -------------------------------------------------------------------------- |
| `/predict`     | POST   | Accepts an image and returns YOLO direction prediction with overlay.       |
| `/metrics`     | GET    | Reads YOLO training CSV, returns final accuracy and confusion matrix path. |
| `/predictions` | GET    | Returns recent predictions (default limit 20).                             |
| `/static/*`    | GET    | Mounted directory for overlays and confusion matrix.                       |
| `/`            | GET    | Health/info endpoint.                                                      |

#### `/predict` detailed flow

1. Validate MIME type (`image/*`) and persist upload to `static/temp_<uuid>.<ext>`.
2. Run model inference (`model(str(temp_path), imgsz=224)`).
3. Extract `pred_idx`/`confidence` from YOLO `probs.top1`.
4. Resolve class names from `model.names` or fallback (`['North','South','East','West']`).
5. Build `all_probabilities` using raw tensor values, ensuring that each cardinal direction exists even if missing from the tensor.
6. **Reconcile final output:** choose the class with the highest probability and use that as the response `direction` and `confidence`.
7. Generate overlay (`create_overlay_image`).
8. Persist prediction metadata to `predictions.json` (prepend, cap at 100 entries).
9. Delete the temporary image and return JSON:

```json
{
  "direction": "South",
  "confidence": 0.987,
  "overlay": "/static/overlays/<uuid>.jpg",
  "model_version": "YOLOv8n-cls",
  "all_probabilities": {
    "North": 0.002,
    "South": 0.987,
    "East": 0.008,
    "West": 0.003
  }
}
```

#### `/metrics`

1. Load `runs/classify/train/results.csv`.
2. Extract final row’s `metrics/accuracy_top1` (default 0).
3. Attempt to locate a confusion matrix PNG in `runs/classify/val/` or `runs/classify/train/`; copy to `static/confusion_matrix.png`.
4. Return `{ "final": { "accuracy": <float> }, "class_names": [...], "confusion_matrix_image": "/static/confusion_matrix.png" }`.

#### `/predictions`

_Parameters:_ `limit` query param (default 20).  
_Behaviour:_ Loads `predictions.json`, slices first `limit` entries, and returns them. Each entry mirrors what `/predict` stored (timestamp ISO string, direction, confidence, overlay path, raw probabilities).

---

## 3. Training Pipeline (`train_yolov8.py`)

The script is a guided CLI workflow. Key functions:

| Function                    | Purpose                                                                                                                |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `install_dependencies()`    | Optional helper to install Ultralytics, PyTorch (CUDA wheels), Matplotlib, etc.                                        |
| `verify_environment()`      | Confirms Ultralytics/Torch availability, prints CUDA status, and validates `data/train` + `data/val` folder structure. |
| `train_model()`             | Runs `yolo classify train` with tuned hyperparameters (epochs=15, imgsz=224, batch=32, dropout=0.2, augment=False).    |
| `validate_model()`          | Executes `yolo classify val` to produce validation metrics and confusion matrix PNG.                                   |
| `predict_validation()`      | Iterates over each `data/val/<direction>` folder and saves YOLO predictions + overlays.                                |
| `visualize_metrics()`       | Parses `results.csv`/`results.json` to plot loss & accuracy curves (`runs/classify/train/accuracy_loss_curves.png`).   |
| `create_inference_script()` | Writes `infer.py` for quick CLI inference usage.                                                                       |
| `export_onnx()`             | Exports best weights to ONNX via `yolo export`.                                                                        |
| `main()`                    | Orchestrates the steps with interactive prompts, allowing users to skip or execute each stage sequentially.            |

Dataset expectations:

```
data/
  train/{North,South,East,West}/
  val/{North,South,East,West}/
```

Each folder contains JPEG/PNG/WebP images already oriented and labeled. Augmentations that change orientation are disabled to preserve directionality.

---

## 4. Frontend Application (`frontend/src`)

### 4.1 App Shell (`App.jsx`)

- Uses React Router for two routes:
  - `/` → `Home` page (prediction)
  - `/metrics` → `Metrics` dashboard
- Shared header with navigation highlights via `NavLink`.

### 4.2 Prediction Page (`pages/Home.jsx`)

State hooks:
| State | Description |
| --- | --- |
| `fileRef` | `<input type="file">` reference for resets. |
| `file` | Selected `File` object. |
| `loading` | Indicates prediction in-progress. |
| `result` | JSON returned from backend. |
| `error` | Friendly error message string. |

Key functions:

1. `onSubmit(e)`
   - Prevents default form submit.
   - Builds `FormData` with the selected file and posts to `${API_BASE}/predict`.
   - Parses JSON, stores in `result`, and handles HTTP/network errors.
2. `onReset()`
   - Clears file, result, error, and resets the input element.
3. `canSubmit` (`useMemo`)
   - Enables/disables the submit button based on file presence and loading state.

Rendering blocks:

1. Upload form.
2. Preview of the uploaded image.
3. Prediction section showing overlay image, predicted direction, confidence, and model version.

### 4.3 Metrics Dashboard (`pages/Metrics.jsx`)

Hooks and logic:

| Hook                                         | Purpose                                                                               |
| -------------------------------------------- | ------------------------------------------------------------------------------------- |
| `useEffect (metrics)`                        | Fetch `/metrics` once, handle loading/error states.                                   |
| `useCallback fetchPredictions` + `useEffect` | Poll `/predictions?limit=20` on load and every 500s (manual refresh button provided). |
| `useMemo aggregate`                          | Simplifies accuracy extraction from metrics payload.                                  |

UI sections:

1. **Model Metrics**
   - Displays accuracy card and confusion matrix image (if provided).
2. **Recent Predictions**
   - “Current Prediction” highlight card with overlay preview.
   - Table listing timestamp, filename, direction, confidence, and overlay link for each stored prediction.

Styling is driven by Tailwind utility classes configured in `tailwind.config.js`.

---

## 5. End-to-End Prediction Flow

1. **User action** – selects image locally and clicks “Predict direction”.
2. **Frontend** – sends multipart POST request to `/predict`.
3. **Backend** – saves temp image → YOLO inference → builds probability dictionary → selects max probability → generates overlay → logs prediction history → responds with JSON.
4. **Frontend** – renders overlay and cards; `Metrics` page auto-refreshes to reflect new history entry.
5. **Static assets** – overlay accessible via `/static/overlays/<uuid>.jpg`, enabling sharing/reference.

---

## 6. API Reference

| Endpoint       | Method | Payload                           | Response                                                       | Notes                                                      |
| -------------- | ------ | --------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------- |
| `/predict`     | POST   | `multipart/form-data` with `file` | JSON (see Section 2.3)                                         | Returns HTTP 400 for non-images, 500 for inference errors. |
| `/metrics`     | GET    | —                                 | `{ final: { accuracy }, class_names, confusion_matrix_image }` | Requires `runs/classify/train/results.csv`.                |
| `/predictions` | GET    | Optional `limit`                  | `{ predictions: [...] }`                                       | Entries sorted newest-first.                               |
| `/`            | GET    | —                                 | `{ message, version }`                                         | Health check.                                              |

Example cURL:

```bash
curl -X POST http://127.0.0.1:8001/predict \
     -F "file=@data/val/North/example.jpg"
```

---

## 7. Operations & Troubleshooting

1. **Port conflicts**: If `python api.py` fails with `OSError: [Errno 10048]`, stop the existing process (`netstat -ano | findstr :8001`, then `taskkill /PID <pid> /F`) or run Uvicorn on a different port.
2. **Model path missing**: Ensure `train_yolov8.py` has been executed or copy `best.pt` into `runs/classify/train/weights/`.
3. **Static assets stale**: Delete `static/overlays` if you want to reset; new predictions will repopulate it.
4. **Prediction history reset**: Remove `predictions.json` to clear history.
5. **Confusion matrix not showing**: Confirm `runs/classify/val/confusion_matrix.png` exists and rerun `/metrics` to regenerate copy in `static/`.

---

## 8. Extensibility Notes

- **Adding new classes**: retrain the YOLO model with additional folders (e.g., `Northeast`), update frontend copy, and ensure `class_names` fallback matches expanded labels.
- **Authentication**: Wrap FastAPI routes with dependencies (OAuth2, API keys) if needed.
- **Persistent database**: Replace `predictions.json` with PostgreSQL/Mongo; adjust `load_predictions`/`save_predictions` accordingly.
- **Batch inference**: Expose a new endpoint that accepts multiple images (ZIP or array of files) and iterates over `predict` logic.
- **Mobile/offline clients**: Convert YOLO weights via `export_onnx()` or TensorRT for on-device inference.

---

This documentation should equip developers and reviewers with a granular understanding of every moving part. For quick setup instructions, continue to use `README.md` and `QUICKSTART.md`; refer back here when modifying code paths, debugging, or presenting the project architecture.
