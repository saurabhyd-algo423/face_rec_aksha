"""FastAPI application for the Aksha face-recognition service.

Endpoints:
  POST /api/faces/enroll-rtsp  – enroll a person from a live RTSP stream
  GET  /api/faces              – list enrolled faces (paginated)
  DELETE /api/faces/{face_id}  – soft-delete a face
  GET  /api/face-events        – query face events (paginated, filterable)
  GET  /api/face-events/{id}   – single event detail
  GET  /api/attendance/{id}    – staff attendance for a date range
  GET  /api/attendance/report  – aggregated report (JSON or CSV)
  WS   /api/face-events/stream – live event stream (WebSocket)
  GET  /health                 – liveness probe
"""
import datetime as dt

from fastapi import FastAPI, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import UploadFile, File, Form
from pymongo import ASCENDING, DESCENDING

from config import config
from service import AttendanceTracker, EnrollmentError, delete_face, enroll_from_rtsp, event_bus
from store import face_events_collection, known_faces_collection

app = FastAPI(title="Aksha Face Recognition API", version="1.0.0")
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
