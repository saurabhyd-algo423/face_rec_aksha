# Aksha Face Recognition Pod (Branch A — Gate Camera)

Gate-only face recognition pod for the Jewelry Store AI Surveillance deployment.

## Scope

- Processes **only** the gate camera (`GATE_CAM=gate_cam`) for face recognition.
- **RTSP-only enrollment**: enroll faces from live camera streams via REST API.
- **Live recognition**: start/stop recognition on any RTSP stream via REST API.
- RetinaFace detect + ByteTrack person IDs + ArcFace embed/FAISS match with per-track cooldown.
- Emits `face_events` and alerts to Kafka topics + **saves to MongoDB**.
- Attendance tracking (first/last seen per staff per day) in Kolkata timezone (IST, UTC+5:30).
- Visitor Re-ID via FAISS embeddings (threshold=0.35).
- ROI-based anonymous visitor counting.
- WebSocket for real-time event streaming to frontend.
- Legacy behavior preserved: `workday.jpg` live image, alert/unauthorised snapshots.

## Project Structure

```
face_recognition/
├── config.py          # Centralised configuration (env vars)
├── models.py          # Face detection, recognition, quality gate, ByteTrack
├── store.py           # Storage abstraction (MongoDB / local JSONL + Kafka publisher)
├── localstore.py      # File-backed JSONL storage for offline/testing mode
├── service.py         # Enrollment pipelines, GateProcessor, AttendanceTracker
├── api.py             # FastAPI endpoints (REST + WebSocket)
├── main.py            # Entry point: Kafka consumer + uvicorn
├── tools/
│   ├── cli_test.py          # CLI test harness (offline)
│   ├── calibrate_threshold.py # Compute optimal similarity threshold
│   └── test_roi.py          # Interactive ROI coordinate tester
├── data/
│   ├── footages/      # Test CCTV footage
│   ├── video_enroll/  # Staff enrollment video clips
│   └── new_data/      # Additional staff images for enrollment
├── API_README.md      # Detailed API documentation
├── Dockerfile         # Container build
└── requirements.txt   # Python dependencies
```

## Configuration (env)

| Env | Default | Meaning |
|-----|---------|---------|
| `MONGO_URI` | `mongodb://mongo:mongo@mongodb:27017/Aksha?authSource=admin&tls=false` | MongoDB connection |
| `MONGO_DB` | `Aksha` | MongoDB database name |
| `KAFKA_BOOTSTRAP_SERVERS` | auto (broker:9092 / localhost:9092) | Kafka broker |
| `STORE_BACKEND` | `mongo` | `mongo` for production, `local` for offline |
| `OUTPUT_SINK` | `kafka` | `kafka` for production, `local` for offline |
| `GATE_CAM` | `gate_cam` | Camera name for face recognition |
| `SITE_ID` | `site_default` | Site identifier on events/alerts |
| `SIMILARITY_THRESHOLD` | `0.8` | ArcFace cosine match threshold |
| `REMATCH_COOLDOWN_SEC` | `2.5` | Min seconds between embeds of the same track |
| `FACE_MIN_SIZE` | `32` | Min face bbox size (px) |
| `DETECT_CONF` | `0.5` | Detection confidence threshold |
| `FRONTAL_MIN` | `0.35` | Min frontal face score |
| `SHARPNESS_MIN` | `60.0` | Min Laplacian variance |
| `BRIGHTNESS_MIN` / `BRIGHTNESS_MAX` | `40.0` / `230.0` | Brightness range |
| `MIN_GOOD_FRAMES` | `3` | Min quality frames for enrollment |
| `ENROLL_SAMPLE_FPS` | `2.0` | Frame sampling rate during enrollment |
| `PROCESS_FPS` | `10` | Frame processing rate |
| `VISITOR_REID_THRESHOLD` | `0.35` | FAISS re-id threshold for visitors |
| `ROI_ENABLED` | `false` | Enable ROI-based visitor counting |
| `ROI_X1` / `ROI_Y1` / `ROI_X2` / `ROI_Y2` | `0.4` / `0.2` / `0.55` / `0.5` | ROI coordinates (normalized 0-1) |
| `ROI_IOU_THRESHOLD` | `0.8` | Min overlap % to count face inside ROI |
| `API_PORT` / `API_HOST` | `8000` / `0.0.0.0` | REST API bind |
| `API_TOKEN` | empty | If set, requires `Authorization: Bearer <token>` |
| `MONITOR_API` | `http://node_backend:5000/api/monitor/` | Legacy monitor publish endpoint |
| `MEDIA_ROOT` | `<AKSHA_PATH>/faces_media` | Media output root |

## MongoDB Collections

All collections are prefixed with `face_rec_` under the `Aksha` database:

```
Aksha (database)
├── face_rec_known_faces        # Enrolled staff/visitors
├── face_rec_face_events        # Every face detection event
├── face_rec_attendance_logs    # Staff first/last seen per day
├── face_rec_visitor_logs       # Visitor Re-ID embeddings
├── face_rec_visitor_counts     # Daily visitor head counts
├── face_rec_alerts             # Alerts (unknown, vip, blacklist)
└── face_rec_facemeta_gate_cam  # Per-camera legacy face metadata
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Run (production with MongoDB + Kafka)

```bash
export MONGO_URI="mongodb://mongo:mongo@<MONGODB_IP>:27017/Aksha?authSource=admin&tls=false"
export KAFKA_BOOTSTRAP_SERVERS="<KAFKA_IP>:9092"
python main.py --fps 10
```

Starts the Kafka consumer (background thread) + FastAPI (uvicorn) on `:8000`.

### 3. Run (offline test mode)

```bash
# Reset local store
python tools/cli_test.py reset

# Enroll staff from video
python tools/cli_test.py enroll data/video_enroll/face1.mp4 --staff_id S01 --name "Rahul"

# Enroll staff from images
python tools/cli_test.py enroll-images data/new_data/amar --staff_id STAFF004 --name "Amar"

# Enroll staff from RTSP
python tools/cli_test.py enroll-rtsp --rtsp-url "rtsp://user:pass@ip:port/stream" --staff-id S05 --name "Test"

# List enrolled faces
python tools/cli_test.py list

# Run recognition on video file
python tools/cli_test.py run data/footages/gate_cam1.mp4 --camera gate_cam --thresh 0.7 --roi --roi-x1 0.35 --roi-y1 0.15 --roi-x2 0.65 --roi-y2 0.6

# Live recognition from RTSP
python tools/cli_test.py recognize-rtsp --rtsp-url "rtsp://user:pass@ip:port/stream" --camera gate_cam
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `POST /api/faces/enroll-rtsp` | POST | Enroll a face from a live RTSP stream |
| `GET /api/faces` | GET | List active enrolled faces (paginated) |
| `DELETE /api/faces/{face_id}` | DELETE | Soft-delete a face |
| `GET /api/face-events` | GET | Query recognition events (filterable) |
| `GET /api/face-events/{event_id}` | GET | Single event detail |
| `GET /api/attendance/{staff_id}` | GET | Per-staff daily first/last seen |
| `GET /api/attendance/report` | GET | Attendance report (JSON/CSV) |
| `POST /api/recognize/start` | POST | Start live recognition on RTSP stream |
| `POST /api/recognize/stop` | POST | Stop live recognition for a camera |
| `GET /api/recognize/status` | GET | List active recognition streams + stats |
| `WS /api/face-events/stream` | WS | Live face-event feed (`?token=...`) |
| `GET /health` | GET | Liveness probe |

For detailed API documentation, see [API_README.md](API_README.md).

## Usage Examples

### Enroll via REST API

```bash
curl -X POST http://localhost:8000/api/faces/enroll-rtsp \
  -F "rtsp_url=rtsp://algocam1:algo1234@192.168.1.151:554/stream1" \
  -F "staff_id=STAFF001" \
  -F "name=Rahul" \
  -F "role=staff"
```

### Start live recognition

```bash
curl -X POST http://localhost:8000/api/recognize/start \
  -H "Content-Type: application/json" \
  -d '{"rtsp_url": "rtsp://algocam1:algo1234@192.168.1.151:554/stream1", "camera_id": "gate_cam"}'
```

### Check recognition status

```bash
curl http://localhost:8000/api/recognize/status
```

### Stop recognition

```bash
curl -X POST http://localhost:8000/api/recognize/stop \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "gate_cam"}'
```

### WebSocket (live events)

```javascript
const ws = new WebSocket("ws://localhost:8000/api/face-events/stream?token=YOUR_TOKEN");
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "face_event") {
    console.log(`${data.payload.match_status}: ${data.payload.name || data.payload.face_id}`);
  }
};
```

## Data Flow

```
RTSP Stream ──> GateProcessor ──> MongoDB (face_rec_face_events)
                      │          ──> MongoDB (face_rec_alerts)
                      │          ──> MongoDB (face_rec_attendance_logs)
                      │          ──> MongoDB (face_rec_visitor_logs)
                      │          ──> Kafka (face_events topic)
                      │          ──> Kafka (alerts topic)
                      │          ──> WebSocket (real-time to frontend)
                      │
                      └──> Legacy DB (face_rec_facemeta_<camera>)
                           Monitor API (workday.jpg)
```

## Module Map

| Module | Purpose |
|--------|---------|
| `config.py` | Centralised configuration from environment variables |
| `models.py` | Face detection (RetinaFace), recognition (ArcFace), quality gate, ByteTrack, FAISS index |
| `store.py` | Storage abstraction layer (MongoDB / local JSONL) + Kafka publisher |
| `localstore.py` | File-backed JSONL storage for offline/testing mode |
| `service.py` | Enrollment pipelines (video/images/RTSP), GateProcessor, AttendanceTracker, EventBus |
| `api.py` | FastAPI endpoints (REST + WebSocket) + RecognitionManager |
| `main.py` | Entry point: Kafka consumer thread + uvicorn server |

## Tools

| Tool | Purpose |
|------|---------|
| `tools/cli_test.py` | Offline CLI test harness (reset, enroll, list, run, recognize-rtsp) |
| `tools/calibrate_threshold.py` | Compute optimal SIMILARITY_THRESHOLD from known/unknown face sets |
| `tools/test_roi.py` | Interactive ROI coordinate tester |

## Alert Schema

Alerts are saved to MongoDB (`face_rec_alerts`) and published to the Kafka `alerts` topic:

```json
{
  "alert_id": "alt_abc123",
  "type": "face_unknown_gate",
  "severity": "MEDIUM",
  "camera_id": "gate_cam",
  "site_id": "site_default",
  "ts": "2026-09-09T10:30:15+00:00",
  "snapshot_url": "/path/to/snapshot.jpg",
  "status": "open",
  "source": "face",
  "meta": {"track_id": 1, "similarity": null}
}
```

Alert types: `face_unknown_gate`, `face_vip_match`, `face_list_match`.
