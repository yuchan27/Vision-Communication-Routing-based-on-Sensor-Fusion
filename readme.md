# Fire Vision Command Center

[English](readme.md) | [Traditional Chinese](README.zh-TW.md)

An operational prototype for smoke and fire screening with YOLO object detection, RGB-based temperature estimation, rule-based safety decisions, video telemetry, live camera monitoring, and a four-camera VCN evacuation workflow.

The repository is currently a local research/prototype system. It is useful for demonstrations, controlled experiments, and integration work; it is not a certified fire alarm or a substitute for a calibrated thermal camera and human safety procedures.

## What is included

- Image inference with annotated output, detections, confidence values, decision results, and explainability text.
- Video inference with optional decision overlays, processed MP4 output, a preview image, and risk/temperature telemetry.
- Live camera or local-video monitoring through an inference worker, Server-Sent Events, and latest-frame polling.
- A VCN pipeline for the bundled four-camera package and evacuation-map composition.
- A browser command center served by the FastAPI backend.
- An explicit temperature contract ready for radiometric thermal-sensor integration.

## Demo

The repository includes `out.gif`, a short demonstration of the current dashboard and inference workflow:

![Fire Vision Command Center demonstration](out.gif)

## Current status and important limitations

The current detector operates on RGB images and the bundled YOLO weights. `FireTemperatureEstimator` produces a visual estimate from the image; that value is not a calibrated physical measurement. The host/CPU temperature reported by the live worker is diagnostic metadata only and is never used as the scene temperature for a safety decision.

The HTTP API accepts an optional `sensor_temperature_celsius` value for image and video requests. When present, it is treated as the authoritative scene temperature and is marked `thermal_sensor` / calibrated in the response. This is an integration seam for a real radiometric sensor, not a hardware driver. A production deployment should send synchronized per-frame thermal measurements rather than a manually entered single value.

The decision engine is a screening aid. Alarm thresholds, persistence, calibration, sensor placement, emissivity, environmental conditions, false-positive handling, and emergency escalation still require domain validation.

## Repository layout

The checkout used for the commands below is:

```text
C:\Code\FIRE
```

The important paths are:

| Path | Purpose |
| --- | --- |
| `AppYOLO/app.py` | FastAPI application, upload validation, path boundaries, live worker, telemetry APIs, and static-file serving. |
| `AppYOLO/start.ps1` | Project startup script that creates or uses a dedicated Python 3.13 environment and launches Uvicorn. |
| `AppYOLO/src/inference/infer.py` | Facade that dispatches image/video work to the specialized inference classes. |
| `AppYOLO/src/inference/image.py` | Single-image YOLO inference, annotation, temperature estimation, and decision payload construction. |
| `AppYOLO/src/inference/video.py` | Frame-by-frame tracking, optional decision dashboard, video writing, and telemetry records. |
| `AppYOLO/src/inference/utils.py` | YOLO conversion helpers and the current RGB temperature estimator. |
| `AppYOLO/src/decision_engine.py` | Rule-based safety decision engine and explainability trace. |
| `AppYOLO/src/pipeline_service.py` | Service wrappers for the original main and VCN workflows. |
| `AppYOLO/src/api_contract.py` | Dependency-free upload, SSE, temperature-source, and validation helpers. |
| `AppYOLO/frontend/index.html` | English operational dashboard markup. |
| `AppYOLO/frontend/app.js` | Dashboard API calls, live updates, charting, previews, and status handling. |
| `AppYOLO/frontend/styles.css` | Responsive dark command-center visual system. |
| `AppYOLO/models/release11.pt` | Default model selected by the backend when present. |
| `AppYOLO/models/release26.pt` | Fallback/alternative bundled model. |
| `AppYOLO/test/` | Small local images, videos, and the VCN camera package used for controlled checks. |
| `AppYOLO/tests/` | Deterministic API-contract and Python-runtime baseline tests. |
| `data/` | Local YOLO dataset: train, validation, and test images plus normalized label files. |
| `data.example.yaml` | Portable local Ultralytics dataset configuration; the ignored `data.yaml` can use the same values. |
| `AppYOLO/outputs/images/` | Annotated image results. |
| `AppYOLO/outputs/videos/` | Processed video results. |
| `AppYOLO/outputs/live_logs/` | JSON Lines telemetry logs created by the live worker. |

Runtime uploads under `AppYOLO/outputs/uploads/` are temporary. Generated outputs and live logs can become large; review the file list before using the cleanup action.

## Requirements

- Windows PowerShell is the documented local workflow.
- Python 3.13.x is the project runtime baseline. The repository `.python-version` file records `3.13`, and the startup script enforces it for the backend modules.
- A working CPU or CUDA installation compatible with the installed PyTorch/Ultralytics stack. The repository does not force a CUDA installation.
- Python dependencies listed in `AppYOLO/requirements.txt`.
- A browser with JavaScript enabled. Chart.js is loaded from jsDelivr by the dashboard; an offline deployment should vendor or replace that dependency.

## Installation and startup (Python 3.13)

Open PowerShell and start from the backend directory. Starting from `AppYOLO` keeps the original inference code's relative output paths predictable.

The recommended project launcher is `AppYOLO/start.ps1`. It creates `.venv313` with the Python 3.13 Windows launcher when needed, installs dependencies on first run, and starts the backend with that environment:

```powershell
Set-Location C:\Code\FIRE\AppYOLO
.\start.ps1 -Install
```

Use `Ctrl+C` in that PowerShell window to stop the server. Add `-Reload` for development auto-reload.

```powershell
Set-Location C:\Code\FIRE\AppYOLO

py -3.13 -m venv .venv313
.\.venv313\Scripts\Activate.ps1

python --version
# Expected: Python 3.13.x

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

For development auto-reload, use the final command below instead. Reloading can initialize the YOLO model more than once, so it is not the default command for a constrained machine:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open the dashboard at [http://127.0.0.1:8000](http://127.0.0.1:8000).

Useful built-in pages:

- Dashboard: `http://127.0.0.1:8000/`
- Interactive API documentation: `http://127.0.0.1:8000/docs`
- ReDoc API documentation: `http://127.0.0.1:8000/redoc`
- Processed-video player: `http://127.0.0.1:8000/player?path=outputs/videos/<file>.mp4`

## Dashboard usage

1. Check the API status in the header. `READY` means the model loaded; `DEGRADED` means the HTTP server is running but inference is unavailable.
2. Use **Image inference** for a known image. Optionally enter a synchronized thermal reading in Celsius. Leave it blank to use the current RGB estimate.
3. Use **Video inference** for a local upload. Video processing is synchronous and can take time on CPU. The optional sensor temperature applies to the processed sequence as a temporary synchronized value.
4. Use **VCN pipeline** to run the bundled multi-camera example from `AppYOLO/WebCamPackage/`.
5. Use **Start live monitor** with `0` for the default webcam or a workspace-relative video path such as `test/dataset/forest1.avi`. Adjust confidence, frame skip, and maximum frame width in the performance controls.
6. Read the **Telemetry and decisions** section. The temperature source is shown beside the value so an RGB estimate is not confused with a calibrated sensor reading.
7. Preview generated images and videos in **Generated artifacts**. The **Clean old** button is destructive within the generated-output directory and asks for confirmation first.

## API reference

All paths in JSON request bodies are relative to `AppYOLO` and must resolve inside that workspace. Upload endpoints use multipart form data and generate random temporary names instead of trusting the client filename.

| Method and path | Use |
| --- | --- |
| `GET /api/health` | Model readiness, live-worker state, upload limits, and temperature contract. Returns `200` with `status: degraded` when the model is unavailable. |
| `GET /api/model/info` | Loaded model path and class mapping. |
| `POST /api/inference/local` | JSON image request: `image_path`, `save_annotated`, optional `sensor_temperature_celsius`. |
| `POST /api/inference/image` | Multipart image upload: `file`, `save_annotated`, optional `sensor_temperature_celsius`. |
| `POST /api/inference/video/local` | JSON video request: `video_path`, `output_video_path`, `with_decision`, optional `sensor_temperature_celsius`. |
| `POST /api/inference/video` | Multipart video upload: `file`, `with_decision`, optional `sensor_temperature_celsius`. |
| `POST /api/pipeline/main/run` | Runs the original image/video workflow through the service layer. |
| `POST /api/pipeline/vcn/run` | Runs the four-camera VCN and map-composition workflow. |
| `GET /api/generated/files?limit=48` | Lists generated output metadata and preview URLs. |
| `POST /api/generated/files/cleanup?keep_latest=40` | Removes older files under `AppYOLO/outputs/`; review the scope before calling it. |
| `POST /api/live/start` | JSON body with `source`, `conf`, `frame_skip`, and `max_frame_width`. File sources must stay inside `AppYOLO`. |
| `POST /api/live/stop` | Stops the live worker. |
| `GET /api/live/state` | Current state, latest metrics, thermal diagnostics, and bounded history arrays. |
| `GET /api/live/frame` | Latest annotated JPEG frame. |
| `GET /api/live/events` | SSE stream for state changes and new frames. |
| `GET /api/video/thumbnail?path=outputs/videos/<file>.mp4` | First-frame preview for a generated video. |
| `GET /api/video/mjpeg?path=outputs/videos/<file>.mp4&fps=15&loop=true` | Browser-friendly MJPEG preview for a generated video. |

### Image upload example

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/inference/image" `
  -F "file=@C:\Code\FIRE\AppYOLO\test\test_fire.jpg" `
  -F "save_annotated=true" `
  -F "sensor_temperature_celsius=83.4"
```

The response contains both the RGB estimate and the effective scene temperature:

```json
{
  "vision_temperature_celsius": 410.0,
  "scene_temperature_celsius": 83.4,
  "scene_temperature_source": "thermal_sensor",
  "scene_temperature_calibrated": true,
  "sensor_temperature_celsius": 83.4
}
```

The numeric values above are an illustrative response shape, not a claim about the sample image's actual temperature.

### Local image example

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/inference/local" `
  -H "Content-Type: application/json" `
  -d '{"image_path":"test/test_fire.jpg","save_annotated":true}'
```

### Live-monitor example

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/live/start" `
  -H "Content-Type: application/json" `
  -d '{"source":"test/dataset/forest1.avi","conf":0.25,"frame_skip":2,"max_frame_width":1280}'
```

Then inspect `GET /api/live/state`, view `GET /api/live/frame`, or subscribe to `GET /api/live/events`.

## Dataset

The local dataset follows the Ultralytics YOLO directory layout:

```text
data/
├── train/
│   ├── images/   14,122 files
│   └── labels/   14,122 files
├── val/
│   ├── images/    3,099 files
│   └── labels/    3,099 files
└── test/
    ├── images/    4,306 files
    └── labels/    4,306 files
```

Inventory summary: 21,527 images and 21,527 label files. The labels are normalized YOLO rows in the form `class_id x_center y_center width height`; the current class mapping is:

```text
0 = smoke
1 = fire
```

Some label files are empty, which represents a background/no-object sample. The repository's ignored `data.yaml` may contain a stale Kaggle absolute path from a previous environment (`/kaggle/working/D Fire Dataset`). For local work, use the tracked `data.example.yaml` or the equivalent paths relative to this checkout:

```yaml
path: .
train: data/train/images
val: data/val/images
test: data/test/images
names:
  0: smoke
  1: fire
```

The dataset was originally referenced from [Smoke/Fire Detection YOLO on Kaggle](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo?resource=download). The original sample-code reference is [johnmartinsson/fire-event-detection-dataset](https://github.com/johnmartinsson/fire-event-detection-dataset). The dataset is not downloaded or modified by the web service.

## VCN camera package

The VCN sample uses `AppYOLO/WebCamPackage/` and its camera map:

```json
{
  "left1.png": "100,100",
  "mid.jpg": "500,100",
  "right1.png": "850,100",
  "right2.jpg": "700,200"
}
```

Run it from the dashboard or call:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/pipeline/vcn/run" `
  -H "Content-Type: application/json" `
  -d '{}'
```

## Architecture

```text
Browser dashboard
    │  REST + SSE + JPEG/MJPEG
    ▼
FastAPI application (AppYOLO/app.py)
    ├── Path and upload boundary
    ├── Image/video/local/pipeline endpoints
    ├── Live worker + bounded telemetry history
    └── Static frontend and generated-output serving
    │
    ▼
YOLOInfer facade
    ├── ImageInfer ── YOLO + RGB temperature estimate
    └── VideoInfer ── tracking + optional decision overlay
    │
    ▼
SafetyDecisionEngine
    └── risk score + alarm flag + suggested action + explainability

Optional/future input
    Radiometric thermal sensor ── synchronized scene temperature / thermal frame
                                  │
                                  └── temperature contract ──► decision fusion
```

The live worker is intentionally bounded to one process-local worker and a fixed history window. It is suitable for a local prototype, not yet for multi-user scheduling, durable jobs, or horizontally scaled deployment.

## Temperature contract and thermal-sensor roadmap

### Current contract

Every decision-capable image/video result should distinguish these fields:

- `scene_temperature_celsius`: the value actually supplied to the decision engine.
- `scene_temperature_source`: `thermal_sensor`, `rgb_estimate`, or `ambient_fallback`.
- `scene_temperature_calibrated`: whether the value came from a calibrated radiometric source.
- `vision_temperature_celsius`: the current RGB-derived estimate, retained for comparison.
- `host_temperature_celsius` / `system_temperature_celsius`: host diagnostics only; never treat these as scene temperature.

### Recommended production design

The best next hardware step is a radiometric thermal-imaging sensor that exposes temperature data, not only a colorized preview. Keep the RGB camera for spatial/contextual detection and use the thermal frame for measurement:

1. Add a `ThermalSensorAdapter` that emits a raw/radiometric frame, calibration metadata, sensor ID, timestamp, unit, and valid temperature range.
2. Synchronize RGB and thermal frames with a shared `frame_id` and timestamp. Calibrate the sensor and define emissivity/environment assumptions before using values in an alarm rule.
3. Align the RGB and thermal images. Project each YOLO fire/smoke bounding box into the thermal frame and calculate region statistics such as max, P95, mean, area above threshold, and rate of rise.
4. Fuse visual confidence, thermal statistics, temporal persistence, and sensor health in the decision engine. Add hysteresis/debounce so one noisy frame does not trigger or clear an alarm.
5. Store the provenance with every decision: sensor ID, calibration version, thermal frame ID, RGB frame ID, region statistics, and decision trace.
6. Validate with controlled hot-object/fire/smoke scenarios, negative scenes, day/night conditions, reflective surfaces, occlusion, distance changes, and sensor-failure cases. Do not claim safety certification from model confidence alone.

A future per-frame envelope could look like this:

```json
{
  "frame_id": 1042,
  "timestamp": "2026-09-08T00:00:00Z",
  "rgb_image": "frames/rgb/1042.jpg",
  "thermal": {
    "frame_path": "frames/thermal/1042.radiometric",
    "temperature_unit": "celsius",
    "max_temperature_celsius": 83.4,
    "p95_temperature_celsius": 79.1,
    "calibrated": true,
    "sensor_id": "thermal-01"
  },
  "detections": [
    {"class_name": "fire", "confidence": 0.92, "bbox": [0.41, 0.34, 0.22, 0.31]}
  ]
}
```

The application should eventually accept that envelope or a sensor-stream adapter rather than a single global temperature field. The current optional form field is deliberately a small compatibility step for integration testing.

## Configuration

The backend reads these optional environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `FIRE_MAX_IMAGE_UPLOAD_BYTES` | `15728640` (15 MiB) | Maximum streamed image upload size. |
| `FIRE_MAX_VIDEO_UPLOAD_BYTES` | `262144000` (250 MiB) | Maximum streamed video upload size. |
| `FIRE_ALLOWED_ORIGINS` | `http://127.0.0.1:8000,http://localhost:8000` | Comma-separated CORS origins when the UI is served separately. |

Example:

```powershell
$env:FIRE_MAX_VIDEO_UPLOAD_BYTES = "524288000"
$env:FIRE_ALLOWED_ORIGINS = "http://127.0.0.1:8000"
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

## Verification commands

Run these from `C:\Code\FIRE\AppYOLO` with Python 3.13:

```powershell
py -3.13 -m unittest discover -s tests -v
py -3.13 -m py_compile app.py src/api_contract.py src/inference/infer.py src/inference/image.py src/inference/video.py
node --check frontend/app.js
```

The first two checks are deterministic source-level checks. A real inference run also depends on the local Python environment, model loading, available camera/video codecs, and hardware. A successful import or HTTP `200` response does not by itself prove that a production deployment or an external thermal sensor is connected.

## References

- [Smoke/Fire Detection YOLO dataset](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo?resource=download)
- [Fire event detection sample code](https://github.com/johnmartinsson/fire-event-detection-dataset)
- [VisiFire demo](http://signal.ee.bilkent.edu.tr/VisiFire/Demo/)
- [Ultralytics documentation](https://docs.ultralytics.com/)
