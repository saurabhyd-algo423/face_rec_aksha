# Aksha Face Recognition Service

This project is a gate-camera face recognition service built for staff identification, visitor tracking, attendance logging, and event streaming. It listens to Kafka frames, processes only the configured gate camera, detects and tracks faces, matches them against an enrolled database using embedding similarity, and writes recognition events to MongoDB and Kafka.

The service is designed to run as a real-time inference pod and also exposes a FastAPI interface for enrollment, roster management, attendance queries, and live event streams.

## Overview

The application does the following:

- Reads raw camera frames from the Kafka `raw_frame` topic.
- Filters for the configured gate camera (`GATE_CAM`).
- Uses RetinaFace for detection and ByteTrack for track management.
- Generates embeddings with ArcFace-style face embedding logic and matches against a FAISS index.
- Stores active faces in MongoDB with embeddings and metadata.
- Emits recognition events to the `face_events` topic and integration alerts to the shared `alerts` topic.
- Tracks first/last seen attendance per staff per day.
- Provides REST and WebSocket APIs for admin operations and live monitoring.
- Supports enrollment from video or RTSP streams.

## High-level architecture

- Ingestion layer: Kafka producer sends raw frames.
- Processing layer: `GateProcessor` in `service.py` detects/tracks/matches faces.
- Storage layer: MongoDB collections for face records, events, and attendance logs.
- Matching layer: FAISS vector index for face recognition.
- API layer: FastAPI app in `api.py` exposes endpoints for enrollment, records, attendance, and live event streaming.
- Outputs: snapshots, event records, alerts, and legacy monitor publishing.

## Project structure

```text
face_rec/
├── API_README.md
├── Readme.md
├── api.py
├── config.py
├── Dockerfile
├── localstore.py
├── main.py
├── models.py
├── requirements.txt
├── service.py
├── store.py
├── data/
│   ├── footages/
│   ├── new_data/
│   └── offline/
├── tools/
│   ├── calibrate_threshold.py
│   ├── cli_test.py
│   └── test_roi.py
└── README references and runtime outputs are generated under MEDIA_ROOT
```

## Main modules

- `main.py`: starts Kafka consumer loop and FastAPI server.
- `config.py`: environment-driven global configuration.
- `service.py`: enrollment flows, attendance tracker, event bus, and gate processing logic.
- `api.py`: FastAPI routes for enrollment, roster, events, attendance, and websocket streaming.
- `models.py`: face detection, embedding, FAISS index wrapper, and scoring utilities.
- `store.py`: MongoDB and local storage helpers.
- `localstore.py`: local/offline storage backend for file-based usage.
- `tools/calibrate_threshold.py`: threshold calibration helper.

## Features

### 1. Real-time gate recognition

The system consumes frames from Kafka and filters only the configured gate camera (`GATE_CAM`). It then runs:

- face detection
- track association
- quality checks
- embedding generation
- nearest-neighbor matching against the FAISS database
- staff/visitor classification
- event publishing

### 2. Enrollment

The service supports enrollment via:

- video upload flow through backend logic (`enroll_from_video`)
- RTSP direct stream capture (`enroll_from_rtsp`)
- raw image batches (`enroll_from_images`)

Each enrollment process:

- validates the face quality
- checks a minimum number of usable frames
- computes a reference embedding
- saves a best snapshot
- writes the data to MongoDB
- adds the face to the in-memory FAISS index

### 3. Attendance tracking

Attendance is tracked by staff_id and date:

- `first_seen_ts` is set on first observation for a date
- `last_seen_ts` is updated to the max timestamp seen
- daily attendance is reportable by staff or by day

### 4. Event and alert emission

The processing loop emits:

- face events to MongoDB and event bus
- live WebSocket notifications
- Kafka alerts to the `alerts` topic
- snapshot images on disk

### 5. WebSocket live stream

Clients can subscribe to live face recognition events using:

```text
WS /api/face-events/stream?token=<API_TOKEN>
```

The stream sends JSON messages of the form:

```json
{
  "type": "face_event",
  "payload": {
    "event_id": "fev_...",
    "ts": "2026-09-15T12:34:56+00:00",
    "camera_id": "gate_cam",
    "match_status": "matched",
    "face_id": "fc_...",
    "staff_id": "STAFF001",
    "confidence": 0.89
  }
}
```

## Requirements

### Runtime dependencies

The project relies on:

- Python 3.10+
- OpenCV
- NumPy
- PyMongo
- Kafka Python
- FastAPI
- Uvicorn
- FAISS
- requests
- scikit/other numeric dependencies used in face processing

See `requirements.txt` for the full dependency list.

### Infrastructure dependencies

- MongoDB instance
- Kafka broker
- Camera feed or RTSP source
- Optional local or network storage for media output

## Environment variables

The service reads most configuration values from environment variables. The defaults are defined in `config.py`.

| Variable | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://mongo:mongo@mongodb:27017/Aksha?authSource=admin&tls=false` | MongoDB connection string |
| `MONGO_DB` | `Aksha` | Mongo database name |
| `KAFKA_BOOTSTRAP_SERVERS` | empty | Kafka bootstrap servers |
| `INPUT_TOPIC` | `raw_frame` | Topic containing camera frames |
| `FACE_EVENTS_TOPIC` | `face_events` | Face event Kafka topic |
| `ALERTS_TOPIC` | `alerts` | Shared alerts topic |
| `GATE_CAM` | `gate_cam` | Gate camera identifier |
| `SITE_ID` | `site_default` | Site name or site ID |
| `SIMILARITY_THRESHOLD` | `0.8` | Matching threshold for face comparison |
| `REMATCH_COOLDOWN_SEC` | `2.5` | Cooldown to avoid repeated re-embeddings on same track |
| `FACE_MIN_SIZE` | `32` | Minimum accepted face box size |
| `DETECT_CONF` | `0.5` | Detection confidence threshold |
| `FRONTAL_MIN` | `0.35` | Front-facing score threshold |
| `SHARPNESS_MIN` | `60.0` | Minimum sharpness threshold |
| `BRIGHTNESS_MIN` | `40.0` | Minimum brightness threshold |
| `BRIGHTNESS_MAX` | `230.0` | Maximum brightness threshold |
| `EMBEDDING_DIM` | `512` | Embedding vector length |
| `MIN_GOOD_FRAMES` | `3` | Minimum valid frames for enrollment |
| `ENROLL_SAMPLE_FPS` | `2.0` | Sampling FPS for video enrollment |
| `UNKNOWN_FRAME_THRESHOLD` | `5` | Unknown frame threshold |
| `SAVE_UNKNOWN_IMAGES` | `true` | Save frames for unknown faces |
| `API_PORT` | `8000` | FastAPI listening port |
| `API_HOST` | `0.0.0.0` | FastAPI bind host |
| `API_TOKEN` | empty | Bearer token for protected APIs |
| `PROCESS_FPS` | `10` | Frame processing rate |
| `MONITOR_API` | `http://node_backend:5000/api/monitor/` | Legacy monitor endpoint |
| `AKSHA_PATH` | `./` | Base path for media output |
| `MEDIA_ROOT` | empty | Directory for media assets |
| `STORE_BACKEND` | `mongo` | Storage backend selector |
| `OUTPUT_SINK` | `kafka` | Output sink selector |
| `PUBLISH_MONITOR` | `true` | Enable legacy monitor publishing |
| `ROI_ENABLED` | `false` | Enable region-of-interest tracking |
| `ROI_X1` / `ROI_Y1` / `ROI_X2` / `ROI_Y2` | defaults | ROI rectangle coordinates |
| `ROI_IOU_THRESHOLD` | `0.8` | ROI overlap threshold |
| `VISITOR_REID_THRESHOLD` | `0.35` | Visitor re-identification threshold |

## Installation

From the project root:

```bash
cd /home/algo6/Saurabh_work/face_rec/face_rec
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you are running in a containerized setup, the project also includes a Dockerfile.

## Configuration example

Create a `.env` file or export the variables before running the service:

```bash
export MONGO_URI="mongodb://mongo:mongo@mongodb:27017/Aksha?authSource=admin&tls=false"
export KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
export INPUT_TOPIC="raw_frame"
export GATE_CAM="gate_cam"
export SITE_ID="site_default"
export API_PORT="8000"
export API_HOST="0.0.0.0"
export API_TOKEN="your-secret-token"
export SIMILARITY_THRESHOLD="0.8"
export PROCESS_FPS="10"
export MEDIA_ROOT="./faces_media"
```

## Running the service

Start the app in normal mode:

```bash
python main.py --fps 10
```

This starts:

- Kafka consumer in a background thread
- FastAPI server on `API_HOST:API_PORT`
- FAISS index build from MongoDB known faces
- event processing pipeline for the configured gate camera

## Docker usage

The repository includes a Dockerfile for container deployment.

Build:

```bash
docker build -t aksha-face-recognition .
```

Run:

```bash
docker run -d --name aksha-face-recognition \
  -p 8000:8000 \
  --env-file .env \
  aksha-face-recognition
```

## API documentation

The API is exposed through FastAPI and is listed in `api.py`.

### `GET /health`

Checks if the service is alive.

Response:

```json
{
  "status": "ok",
  "service": "face_recognition"
}
```

### `POST /api/faces/enroll-rtsp`

Enroll a person from a live RTSP stream.

Form fields:

- `rtsp_url` - required
- `staff_id` - required
- `name` - required
- `role` - optional, defaults to `staff`
- `site_id` - optional, defaults from config
- `category` - optional, defaults to `staff`
- `max_duration` - optional, defaults to `10`
- `sample_fps` - optional, defaults to `2.0`
- `min_frames` - optional, defaults to `3`

Example:

```bash
curl -X POST "http://localhost:8000/api/faces/enroll-rtsp" \
  -F "rtsp_url=rtsp://camera.example/live" \
  -F "staff_id=STAFF001" \
  -F "name=Rohit Sharma" \
  -F "role=staff" \
  -F "site_id=site_default" \
  -H "Authorization: Bearer your-secret-token"
```

### `GET /api/faces`

List active enrolled faces with optional filters.

Query parameters:

- `site_id`
- `role`
- `search`
- `page`
- `limit`

Example:

```bash
curl "http://localhost:8000/api/faces?site_id=site_default&page=1&limit=20" \
  -H "Authorization: Bearer your-secret-token"
```

### `DELETE /api/faces/{face_id}`

Soft deletes a registered face without removing historical records.

### `GET /api/face-events`

List face events with optional filters.

Query parameters include:

- `site_id`
- `camera_id`
- `zone`
- `match_status` (`matched`, `unknown`, `all`)
- `from`
- `to`
- `page`
- `limit`

Example:

```bash
curl "http://localhost:8000/api/face-events?site_id=site_default&match_status=matched&page=1&limit=20" \
  -H "Authorization: Bearer your-secret-token"
```

### `GET /api/face-events/{event_id}`

Fetch one event record by its ID.

### `GET /api/attendance/{staff_id}`

Get daily attendance rows for a staff member.

Example:

```bash
curl "http://localhost:8000/api/attendance/STAFF001?date=2026-09-15" \
  -H "Authorization: Bearer your-secret-token"
```

### `GET /api/attendance/report`

Get aggregated attendance data in JSON or CSV format.

Example:

```bash
curl "http://localhost:8000/api/attendance/report?site_id=site_default&start=2026-09-01&end=2026-09-15&group_by=staff&format=json" \
  -H "Authorization: Bearer your-secret-token"
```

### `WS /api/face-events/stream`

Open a WebSocket and subscribe to real-time events.

Example using Python:

```python
import asyncio
import websockets

async def main():
    uri = "ws://localhost:8000/api/face-events/stream?token=your-secret-token"
    async with websockets.connect(uri) as ws:
        while True:
            msg = await ws.recv()
            print(msg)

asyncio.run(main())
```

## MongoDB collections

The system expects these kinds of collections:

- `known_faces` - enrolled faces and embeddings
- `face_events` - recognition events, snapshots, and metadata
- `attendance_logs` - first/last seen attendance records

The project calls `ensure_indexes()` during startup to prepare the DB for efficient access.

## Data and output folders

The media root defaults to `AKSHA_PATH/faces_media` unless overridden using `MEDIA_ROOT`.

Typical output folders include:

- `snapshots/` - event and enrollment crops
- `videos/` - saved enrollment videos
- `local_store/` - local file-based storage output
- `local_out/` - local runtime outputs

## Operational notes

- If `API_TOKEN` is set, all protected endpoints require `Authorization: Bearer <API_TOKEN>`.
- If no token is set, the API runs without authentication.
- `SIMILARITY_THRESHOLD` and face-quality settings may need calibration per site and camera lighting conditions.
- The gate camera should match the actual producer camera name configured in Kafka headers.
- The system uses a track cooldown to avoid repeated re-embedding on the same person within a short window.

## Troubleshooting

### Kafka frames are not being processed

Check:

- `INPUT_TOPIC` value
- `KAFKA_BOOTSTRAP_SERVERS`
- the Kafka message header `camera_name`
- that the producer sends `GATE_CAM` exactly as configured

### Face recognition is too sensitive or too strict

Adjust:

- `SIMILARITY_THRESHOLD`
- `FRONTAL_MIN`
- `BRIGHTNESS_MIN` and `BRIGHTNESS_MAX`
- `SHARPNESS_MIN`
- `FACE_MIN_SIZE`

### Enrollment fails

Common causes:

- no usable face in the stream/video
- poor lighting or motion blur
- not enough good frames
- duplicated `staff_id` already active in the database

### API returns 401

Set a valid `API_TOKEN` or disable auth by leaving it empty.

## Development notes

- Adjust the camera filtering logic in `main.py` and `service.py` if you need to support more than one camera.
- The recognition logic is built around a single gate-camera processing path; extending to multiple cameras may require track isolation and camera-specific handling.
- The project includes helper tools in `tools/` for calibration and ROI testing.

## Useful commands

```bash
# Start app
python main.py --fps 10

# Health check
curl http://localhost:8000/health

# View active faces
curl "http://localhost:8000/api/faces?site_id=site_default" -H "Authorization: Bearer your-secret-token"

# Inspect events
curl "http://localhost:8000/api/face-events?site_id=site_default&page=1&limit=10" -H "Authorization: Bearer your-secret-token"
```

## Summary

This service is a real-time gate-facing face recognition pipeline designed for secure staff and visitor identification, event streaming, and attendance tracking. It combines Kafka ingestion, MongoDB persistence, FAISS-based matching, and FastAPI-admin tooling into a single operational deployment for a surveillance environment.

If you want, I can also create a second version of the README tailored for deployment, operations, or developer onboarding specifically.
