"""FastAPI application for the Aksha face-recognition service.

Endpoints:
  POST /api/faces/enroll-rtsp     – enroll a person from a live RTSP stream
  GET  /api/faces                 – list enrolled faces (paginated)
  DELETE /api/faces/{face_id}     – soft-delete a face
  GET  /api/face-events           – query face events (paginated, filterable)
  GET  /api/face-events/{id}      – single event detail
  GET  /api/attendance/{id}       – staff attendance for a date range
  GET  /api/attendance/report     – aggregated report (JSON or CSV)
  WS   /api/face-events/stream    – live event stream (WebSocket)
  POST /api/recognize/start       – start live recognition on RTSP stream
  POST /api/recognize/stop        – stop live recognition for a camera
  GET  /api/recognize/status      – list active recognition streams
  GET  /health                    – liveness probe
"""
import asyncio
import datetime as dt
import os
import threading
import time
import uuid

import cv2
from fastapi import FastAPI, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import UploadFile, File, Form
from pymongo import ASCENDING, DESCENDING
from pydantic import BaseModel

from config import config
from models import FACE_INDEX, build_index_from_db
from service import AttendanceTracker, EnrollmentError, GateProcessor, delete_face, enroll_from_rtsp, event_bus
from store import alerts_collection, face_events_collection, known_faces_collection

app = FastAPI(title="Aksha Face Recognition API", version="1.1.0")
security = HTTPBearer(auto_error=False)

attendance = AttendanceTracker()


def require_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency that validates the Bearer token against ``config.API_TOKEN``."""
    if not config.API_TOKEN:
        return True
    if credentials is None or credentials.credentials != config.API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return True


@app.get("/health")
def health():
    """Liveness probe endpoint."""
    return {"status": "ok", "service": "face_recognition"}


# --- Enrollment from images/VIDEO COMMENTED OUT ---
# --- RTSP enrollment only ---

@app.post("/api/faces/enroll-rtsp", status_code=201, dependencies=[Depends(require_auth)])
async def enroll_rtsp(
    rtsp_url: str = Form(...),
    staff_id: str = Form(...),
    name: str = Form(...),
    role: str = Form("staff"),
    site_id: str = Form(config.SITE_ID),
    category: str = Form("staff"),
    max_duration: int = Form(10),
    sample_fps: float = Form(2.0),
    min_frames: int = Form(3),
):
    """Enroll a person from a live RTSP stream.

    Captures frames for up to *max_duration* seconds, applies quality gates,
    and stores the aggregated embedding.  Returns the new face_id on success.
    """
    try:
        face_id = enroll_from_rtsp(rtsp_url, staff_id, name, role, site_id, category, max_duration, sample_fps, min_frames)
    except EnrollmentError as e:
        raise HTTPException(status_code=422, detail={"code": e.code, "message": str(e)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"face_id": face_id, "staff_id": staff_id, "name": name, "status": "enrolled"}


@app.get("/api/faces", dependencies=[Depends(require_auth)])
def list_faces(
    site_id: str = Query(config.SITE_ID),
    role: str = Query(None),
    search: str = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200),
):
    """List active enrolled faces with optional role/search filters."""
    q = {"site_id": site_id, "status": "active"}
    if role:
        q["role"] = role
    if search:
        import re
        q["$or"] = [
            {"name": {"$regex": search, "$options": "i"}},
            {"staff_id": {"$regex": search, "$options": "i"}},
        ]
    coll = known_faces_collection()
    total = coll.count_documents(q)
    docs = coll.find(q, {"embedding": 0}).sort("enrolled_at", DESCENDING).skip((page - 1) * limit).limit(limit)
    data = []
    for d in docs:
        d.pop("_id", None)
        data.append(d)
    return {"data": data, "page": page, "total": total, "has_more": page * limit < total}


@app.delete("/api/faces/{face_id}", status_code=204, dependencies=[Depends(require_auth)])
def remove_face(face_id: str):
    """Soft-delete a face (sets status to 'inactive', removes from FAISS index)."""
    removed = delete_face(face_id)
    if not removed:
        raise HTTPException(status_code=404, detail={"code": "FACE_NOT_FOUND"})
    return None

@app.get("/api/face-events", dependencies=[Depends(require_auth)])
def list_face_events(
    site_id: str = Query(config.SITE_ID),
    camera_id: str = Query(None),
    zone: str = Query(None),
    match_status: str = Query(None, pattern="^(matched|unknown|all|)$"),
    from_ts: str = Query(None, alias="from"),
    to_ts: str = Query(None, alias="to"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200),
):
    """Query face events with optional camera, zone, status, and time filters."""
    q = {"site_id": site_id}
    if camera_id:
        q["camera_id"] = camera_id
    if zone:
        q["zone"] = zone
    if match_status in ("matched", "unknown"):
        q["match_status"] = match_status
    if from_ts or to_ts:
        q["ts"] = {}
        if from_ts:
            q["ts"]["$gte"] = dt.datetime.fromisoformat(from_ts)
        if to_ts:
            q["ts"]["$lte"] = dt.datetime.fromisoformat(to_ts)

    coll = face_events_collection()
    total = coll.count_documents(q)
    docs = coll.find(q).sort("ts", DESCENDING).skip((page - 1) * limit).limit(limit)
    data = []
    for d in docs:
        d.pop("_id", None)
        d["ts"] = d["ts"].isoformat() if isinstance(d.get("ts"), dt.datetime) else d.get("ts")
        data.append(d)
    return {"data": data, "page": page, "total": total, "has_more": page * limit < total}


@app.get("/api/face-events/{event_id}", dependencies=[Depends(require_auth)])
def face_event_detail(event_id: str):
    """Return a single face event by its event_id."""
    doc = face_events_collection().find_one({"event_id": event_id})
    if not doc:
        raise HTTPException(status_code=404, detail={"code": "EVENT_NOT_FOUND"})
    doc.pop("_id", None)
    if isinstance(doc.get("ts"), dt.datetime):
        doc["ts"] = doc["ts"].isoformat()
    return doc


@app.get("/api/attendance/{staff_id}", dependencies=[Depends(require_auth)])
def staff_attendance(staff_id: str, date: str = Query(None, description="YYYY-MM-DD")):
    """Return attendance rows for a specific staff member."""
    rows = attendance.report_for(staff_id, day=date)
    return {"staff_id": staff_id, "rows": rows}


@app.get("/api/attendance/report", dependencies=[Depends(require_auth)])
def attendance_report(
    site_id: str = Query(config.SITE_ID),
    start: str = Query(None),
    end: str = Query(None),
    group_by: str = Query("staff", pattern="^(staff|day)$"),
    format: str = Query("json", pattern="^(json|csv)$"),
):
    """Aggregated attendance report grouped by staff or day, JSON or CSV."""
    rows = attendance.report(site_id=site_id, start=start, end=end, group_by=group_by)
    if format == "csv":
        import csv
        import io
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["staff_id", "date", "first_seen_ts", "last_seen_ts", "total_hours"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(buf.getvalue(), media_type="text/csv")
    return {"data": rows}


# ---------------------------------------------------------------------------
# Live Recognition Manager
# ---------------------------------------------------------------------------

class RecognitionManager:
    """Manages background RTSP recognition tasks.

    Each camera gets its own thread that reads frames from the RTSP stream,
    runs them through ``GateProcessor``, and publishes events/alerts to
    MongoDB / Kafka / WebSocket.
    """

    def __init__(self):
        self._tasks = {}
        self._lock = threading.Lock()

    def start(self, rtsp_url, camera_id=None, site_id=None,
              thresh=0.7, cooldown=2.5, roi=False,
              roi_x1=None, roi_y1=None, roi_x2=None, roi_y2=None):
        camera_id = camera_id or config.GATE_CAM
        site_id = site_id or config.SITE_ID

        with self._lock:
            if camera_id in self._tasks and self._tasks[camera_id]["status"] == "running":
                return None

            task_id = f"task_{uuid.uuid4().hex[:10]}"
            stop_event = threading.Event()
            state = {
                "task_id": task_id,
                "camera_id": camera_id,
                "rtsp_url": rtsp_url,
                "site_id": site_id,
                "status": "running",
                "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "frames_processed": 0,
                "matched": 0,
                "unknown": 0,
                "stop_event": stop_event,
            }
            self._tasks[camera_id] = state

            t = threading.Thread(
                target=self._run,
                args=(camera_id, rtsp_url, thresh, cooldown,
                      roi, roi_x1, roi_y1, roi_x2, roi_y2, stop_event),
                daemon=True,
            )
            t.start()
            return {k: v for k, v in state.items() if k != "stop_event"}

    def stop(self, camera_id):
        with self._lock:
            task = self._tasks.get(camera_id)
            if not task or task["status"] != "running":
                return None
            task["stop_event"].set()
            task["status"] = "stopping"
            return {"camera_id": camera_id, "status": "stopping"}

    def status(self):
        with self._lock:
            return [
                {k: v for k, v in t.items() if k != "stop_event"}
                for t in self._tasks.values()
            ]

    def _run(self, camera_id, rtsp_url, thresh, cooldown,
             roi, roi_x1, roi_y1, roi_x2, roi_y2, stop_event):
        try:
            os.environ["SIMILARITY_THRESHOLD"] = str(thresh)
            os.environ["REMATCH_COOLDOWN_SEC"] = str(cooldown)
            os.environ["STORE_BACKEND"] = "mongo"
            os.environ["OUTPUT_SINK"] = "kafka"

            if roi:
                os.environ["ROI_ENABLED"] = "true"
                if roi_x1 is not None: os.environ["ROI_X1"] = str(roi_x1)
                if roi_y1 is not None: os.environ["ROI_Y1"] = str(roi_y1)
                if roi_x2 is not None: os.environ["ROI_X2"] = str(roi_x2)
                if roi_y2 is not None: os.environ["ROI_Y2"] = str(roi_y2)

            build_index_from_db(known_faces_collection())
            print(f"[{camera_id}] FAISS index: {len(FACE_INDEX.face_ids())} face(s)")

            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                print(f"[{camera_id}] ERROR: Cannot connect to {rtsp_url}")
                with self._lock:
                    self._tasks[camera_id]["status"] = "failed"
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            process_fps = config.PROCESS_FPS
            frame_interval = 1.0 / process_fps

            processor = GateProcessor()
            processor.load_daily_state()
            base_ts = dt.datetime.now(dt.timezone.utc)
            frame_idx = 0

            print(f"[{camera_id}] Live recognition started from {rtsp_url}")

            while not stop_event.is_set():
                t_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    print(f"[{camera_id}] Stream disconnected.")
                    break

                ts = base_ts + dt.timedelta(seconds=frame_idx / fps)
                events = processor.process_frame(frame, camera_id, ts)

                with self._lock:
                    task = self._tasks.get(camera_id, {})
                    task["frames_processed"] = task.get("frames_processed", 0) + 1
                    for e in events:
                        if e.get("match_status") == "matched":
                            task["matched"] = task.get("matched", 0) + 1
                        else:
                            task["unknown"] = task.get("unknown", 0) + 1

                frame_idx += 1
                elapsed = time.time() - t_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            cap.release()
            processor.persist_daily_counts()
            with self._lock:
                self._tasks[camera_id]["status"] = "stopped"
            print(f"[{camera_id}] Stopped. Processed {frame_idx} frames.")

        except Exception as e:
            print(f"[{camera_id}] Recognition error: {e}")
            with self._lock:
                if camera_id in self._tasks:
                    self._tasks[camera_id]["status"] = "failed"


recognition_manager = RecognitionManager()


# ---------------------------------------------------------------------------
# Live Recognition Endpoints
# ---------------------------------------------------------------------------

class RecognizeStartRequest(BaseModel):
    rtsp_url: str
    camera_id: str = None
    site_id: str = None
    thresh: float = 0.7
    cooldown: float = 2.5
    roi: bool = False
    roi_x1: float = None
    roi_y1: float = None
    roi_x2: float = None
    roi_y2: float = None


class RecognizeStopRequest(BaseModel):
    camera_id: str


@app.post("/api/recognize/start", status_code=201, dependencies=[Depends(require_auth)])
def start_recognition(req: RecognizeStartRequest):
    """Start live face recognition on an RTSP stream.

    Runs a background thread that captures frames, detects faces,
    matches against the known-face index, records attendance, and
    publishes events/alerts to MongoDB + Kafka.
    """
    result = recognition_manager.start(
        rtsp_url=req.rtsp_url,
        camera_id=req.camera_id,
        site_id=req.site_id,
        thresh=req.thresh,
        cooldown=req.cooldown,
        roi=req.roi,
        roi_x1=req.roi_x1,
        roi_y1=req.roi_y1,
        roi_x2=req.roi_x2,
        roi_y2=req.roi_y2,
    )
    if result is None:
        raise HTTPException(status_code=409, detail={"code": "ALREADY_RUNNING", "message": "Recognition already running on this camera"})
    return result


@app.post("/api/recognize/stop", dependencies=[Depends(require_auth)])
def stop_recognition(req: RecognizeStopRequest):
    """Stop live recognition for a specific camera."""
    result = recognition_manager.stop(req.camera_id)
    if result is None:
        raise HTTPException(status_code=404, detail={"code": "NOT_RUNNING", "message": "No recognition task running on this camera"})
    return result


@app.get("/api/recognize/status", dependencies=[Depends(require_auth)])
def recognition_status():
    """List all active recognition streams."""
    return {"active_streams": recognition_manager.status()}


@app.websocket("/api/face-events/stream")
async def face_events_stream(websocket: WebSocket):
    """WebSocket endpoint that streams face events in real time.

    Send ``?token=<API_TOKEN>`` as a query param for auth.
    Pings keepalive every 30 s; closes with 4401 on auth failure.
    """
    token = websocket.query_params.get("token", "")
    if config.API_TOKEN and token != config.API_TOKEN:
        await websocket.close(code=4401, reason="auth expired")
        return
    await websocket.accept()
    last_id = None
    last_ping = asyncio.get_event_loop().time()
    try:
        while True:
            events = event_bus.drain(after_id=last_id)
            for e in events:
                await websocket.send_json({"type": "face_event", "payload": e})
                last_id = e.get("event_id")
            now = asyncio.get_event_loop().time()
            if now - last_ping >= 30:
                await websocket.send_json({"type": "ping", "ts": dt.datetime.now(dt.timezone.utc).isoformat()})
                last_ping = now
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
