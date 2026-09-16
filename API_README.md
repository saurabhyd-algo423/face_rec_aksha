# Aksha Face Recognition API

## Overview
FastAPI service for gate camera face recognition, enrollment, attendance tracking, and event streaming.

## Authentication
All endpoints require `Authorization: Bearer <API_TOKEN>` header. Set `API_TOKEN` env var to enable. If unset, auth is disabled.

## Endpoints

### Health
```
GET /health
```
**Response:**
```json
{ "status": "ok", "service": "face_recognition" }
```

---

### Enrollment (RTSP only)
```
POST /api/faces/enroll-rtsp
Content-Type: multipart/form-data
```
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `rtsp_url` | string | yes | - | RTSP stream URL |
| `staff_id` | string | yes | - | Unique staff identifier |
| `name` | string | yes | - | Person's name |
| `role` | string | no | `staff` | `staff` or `visitor` |
| `site_id` | string | no | config default | Site identifier |
| `category` | string | no | `staff` | Category label |
| `max_duration` | int | no | `10` | Max enrollment time (sec) |
| `sample_fps` | float | no | `2.0` | Frame sampling rate |
| `min_frames` | int | no | `3` | Minimum good frames to enroll |

**Response (201):**
```json
{ "face_id": "face_abc123", "staff_id": "STAFF001", "name": "Rahul", "status": "enrolled" }
```

**Errors:**
- `422` - Quality gate failed (code: `QUALITY_GATE_FAILED`)
- `422` - No face detected (code: `NO_FACE`)
- `500` - Stream connection error

---

### List Faces
```
GET /api/faces?site_id={site_id}&role={role}&search={query}&page={page}&limit={limit}
```
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `site_id` | string | config default | Filter by site |
| `role` | string | - | Filter by role |
| `search` | string | - | Search name/staff_id (case-insensitive) |
| `page` | int | 1 | Page number |
| `limit` | int | 20 | Items per page (max 200) |

**Response:**
```json
{
  "data": [
    {
      "face_id": "face_abc123",
      "staff_id": "STAFF001",
      "name": "Rahul",
      "role": "staff",
      "site_id": "site_default",
      "status": "active",
      "enrolled_at": "2026-09-09T10:30:00+05:30"
    }
  ],
  "page": 1,
  "total": 21,
  "has_more": false
}
```

---

### Delete Face (Soft Delete)
```
DELETE /api/faces/{face_id}
```
**Response:** `204 No Content`

**Error:** `404` if face not found.

---

### List Face Events
```
GET /api/face-events?site_id={site_id}&camera_id={camera_id}&zone={zone}&match_status={status}&from={iso_ts}&to={iso_ts}&page={page}&limit={limit}
```
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `site_id` | string | config default | Filter by site |
| `camera_id` | string | - | Filter by camera |
| `zone` | string | - | Filter by zone |
| `match_status` | string | - | `matched`, `unknown`, or `all` |
| `from` | string | - | ISO timestamp (start) |
| `to` | string | - | ISO timestamp (end) |
| `page` | int | 1 | Page number |
| `limit` | int | 20 | Items per page |

**Response:**
```json
{
  "data": [
    {
      "event_id": "fev_abc123",
      "ts": "2026-09-09T10:30:15+05:30",
      "camera_id": "gate_cam",
      "zone": "gate",
      "match_status": "matched",
      "face_id": "face_abc123",
      "staff_id": "STAFF001",
      "confidence": 0.85,
      "snapshot_url": "/snapshots/gate_cam/fev_abc123.jpg"
    }
  ],
  "page": 1,
  "total": 150,
  "has_more": true
}
```

---

### Face Event Detail
```
GET /api/face-events/{event_id}
```
**Response:** Single face event object.

**Error:** `404` if event not found.

---

### Staff Attendance
```
GET /api/attendance/{staff_id}?date={YYYY-MM-DD}
```
**Response:**
```json
{
  "staff_id": "STAFF001",
  "rows": [
    {
      "date": "2026-09-09",
      "first_seen_ts": "2026-09-09T09:15:00+05:30",
      "last_seen_ts": "2026-09-09T18:45:00+05:30",
      "total_hours": 9.5
    }
  ]
}
```

---

### Attendance Report
```
GET /api/attendance/report?site_id={site_id}&start={YYYY-MM-DD}&end={YYYY-MM-DD}&group_by={staff|day}&format={json|csv}
```
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `site_id` | string | config default | Filter by site |
| `start` | string | - | Start date (YYYY-MM-DD) |
| `end` | string | - | End date (YYYY-MM-DD) |
| `group_by` | string | `staff` | Group by `staff` or `day` |
| `format` | string | `json` | `json` or `csv` |

**Response (JSON):**
```json
{
  "data": [
    { "staff_id": "STAFF001", "date": "2026-09-09", "first_seen_ts": "...", "last_seen_ts": "...", "total_hours": 9.5 }
  ]
}
```

**Response (CSV):** Returns `text/csv` with headers.

---

### WebSocket - Live Face Events
```
WS /api/face-events/stream?token={API_TOKEN}
```
**Messages received:**
```json
{
  "type": "face_event",
  "payload": {
    "event_id": "fev_xyz",
    "ts": "...",
    "camera_id": "gate_cam",
    "match_status": "matched|unknown",
    "face_id": "...",
    "staff_id": "...",
    "confidence": 0.85
  }
}
```
Keepalive pings every 30s:
```json
{ "type": "ping", "ts": "..." }
```

**Close codes:** `4401` - auth expired.

---

### Start Live Recognition
```
POST /api/recognize/start
Content-Type: application/json
```
```json
{
  "rtsp_url": "rtsp://user:pass@ip:port/stream",
  "camera_id": "gate_cam",
  "site_id": "site_default",
  "thresh": 0.7,
  "cooldown": 2.5,
  "roi": false,
  "roi_x1": 0.35,
  "roi_y1": 0.15,
  "roi_x2": 0.65,
  "roi_y2": 0.6
}
```
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `rtsp_url` | string | yes | - | RTSP stream URL |
| `camera_id` | string | no | `gate_cam` | Camera identifier |
| `site_id` | string | no | config default | Site identifier |
| `thresh` | float | no | `0.7` | Similarity threshold |
| `cooldown` | float | no | `2.5` | Rematch cooldown (sec) |
| `roi` | bool | no | `false` | Enable ROI visitor counting |
| `roi_x1` | float | no | - | ROI top-left X (0-1) |
| `roi_y1` | float | no | - | ROI top-left Y (0-1) |
| `roi_x2` | float | no | - | ROI bottom-right X (0-1) |
| `roi_y2` | float | no | - | ROI bottom-right Y (0-1) |

**Response (201):**
```json
{
  "task_id": "task_abc123",
  "camera_id": "gate_cam",
  "rtsp_url": "rtsp://...",
  "status": "running",
  "started_at": "2026-09-09T10:30:00+00:00"
}
```

**Errors:**
- `409` - Already running on this camera

---

### Stop Live Recognition
```
POST /api/recognize/stop
Content-Type: application/json
```
```json
{ "camera_id": "gate_cam" }
```

**Response:**
```json
{ "camera_id": "gate_cam", "status": "stopping" }
```

**Error:** `404` if not running.

---

### Recognition Status
```
GET /api/recognize/status
```
**Response:**
```json
{
  "active_streams": [
    {
      "task_id": "task_abc123",
      "camera_id": "gate_cam",
      "rtsp_url": "rtsp://...",
      "status": "running",
      "started_at": "2026-09-09T10:30:00+00:00",
      "frames_processed": 1500,
      "matched": 45,
      "unknown": 12
    }
  ]
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGO_URI` | `mongodb://mongo:mongo@mongodb:27017/Aksha?authSource=admin&tls=false` | MongoDB connection |
| `MONGO_DB` | `Aksha` | Database name |
| `KAFKA_BOOTSTRAP_SERVERS` | - | Kafka broker (comma-separated) |
| `FACE_EVENTS_TOPIC` | `face_events` | Kafka topic for face events |
| `ALERTS_TOPIC` | `alerts` | Kafka topic for alerts |
| `GATE_CAM` | `gate_cam` | Gate camera identifier |
| `SITE_ID` | `site_default` | Site identifier |
| `SIMILARITY_THRESHOLD` | `0.8` | Face match threshold |
| `REMATCH_COOLDOWN_SEC` | `2.5` | Cooldown between same-person alerts |
| `FACE_MIN_SIZE` | `32` | Min face bbox size (px) |
| `DETECT_CONF` | `0.5` | Detection confidence threshold |
| `FRONTAL_MIN` | `0.35` | Min frontal face score |
| `SHARPNESS_MIN` | `60.0` | Min Laplacian variance |
| `BRIGHTNESS_MIN` | `40.0` | Min brightness |
| `BRIGHTNESS_MAX` | `230.0` | Max brightness |
| `VISITOR_REID_THRESHOLD` | `0.35` | FAISS re-id threshold |
| `ROI_ENABLED` | `false` | Enable ROI-based visitor counting |
| `ROI_X1` | `0.4` | ROI top-left X (normalized 0-1) |
| `ROI_Y1` | `0.2` | ROI top-left Y (normalized 0-1) |
| `ROI_X2` | `0.55` | ROI bottom-right X (normalized 0-1) |
| `ROI_Y2` | `0.5` | ROI bottom-right Y (normalized 0-1) |
| `ROI_IOU_THRESHOLD` | `0.8` | Min overlap % to count face inside ROI |
| `API_PORT` | `8000` | API server port |
| `API_HOST` | `0.0.0.0` | API bind address |
| `API_TOKEN` | - | Bearer token (empty = no auth) |
| `PROCESS_FPS` | `10` | Frame processing rate |
| `STORE_BACKEND` | `mongo` | `mongo` or `local` |
| `OUTPUT_SINK` | `kafka` | `kafka` or `local` |
| `EMBEDDING_DIM` | `512` | Face embedding dimension |
| `MIN_GOOD_FRAMES` | `3` | Min quality frames for enrollment |
| `ENROLL_SAMPLE_FPS` | `2.0` | Frame sampling rate during enrollment |
| `UNKNOWN_FRAME_THRESHOLD` | `5` | Frames before saving unknown image |
| `SAVE_UNKNOWN_IMAGES` | `true` | Save unknown face images to disk |
| `PUBLISH_MONITOR` | `true` | Publish frames to monitor API |
| `MONITOR_API` | `http://node_backend:5000/api/monitor/` | Monitor API endpoint |
| `MEDIA_ROOT` | `<AKSHA_PATH>/faces_media` | Media output root |
| `LOCAL_STORE_DIR` | `<MEDIA_ROOT>/local_store` | Local JSONL store dir |
| `LOCAL_OUT_DIR` | `<MEDIA_ROOT>/local_out` | Local output dir |
