"""Core face-recognition service layer.

Contains the enrollment pipelines (video, images, RTSP), the
``GateProcessor`` that drives the real-time gate-camera loop, and
supporting utilities for attendance tracking, event publishing,
visitor Re-ID, and snapshot management.
"""
import collections
import datetime as dt
import json
import os
import threading
import uuid

import cv2
import faiss
import numpy as np
import requests

from config import config
from models import FACE_INDEX, ByteTracker, assess_face_scores, detect_faces, embed_face, iou
from store import attendance_logs_collection, face_events_collection, known_faces_collection, load_database, publisher


class EnrollmentError(Exception):
    """Raised when an enrollment attempt fails (e.g. no face, low quality)."""
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def _save_video(data, staff_id):
    """Persist raw video bytes and return the file path."""
    os.makedirs(os.path.join(config.media_root, "videos"), exist_ok=True)
    name = f"{staff_id}_{uuid.uuid4().hex[:8]}.mp4"
    path = os.path.join(config.media_root, "videos", name)
    with open(path, "wb") as f:
        f.write(data)
    return path


def _save_enrollment_snapshot(frame, bbox):
    """Crop the best face frame and save as a JPEG snapshot."""
    os.makedirs(os.path.join(config.media_root, "snapshots"), exist_ok=True)
    name = f"{uuid.uuid4().hex[:8]}.jpg"
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    path = os.path.join(config.media_root, "snapshots", name)
    cv2.imwrite(path, crop)
    return path


def _sample_frames(video_path, sample_fps):
    """Yield frames from *video_path* at the requested sample rate."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps / max(sample_fps, 0.5))))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames.append(frame)
            idx += 1
    finally:
        cap.release()
    return frames


def _aggregate(embeddings):
    """Average-normalise a list of embeddings into a single reference vector."""
    mean = np.mean(np.asarray(embeddings, dtype='float32'), axis=0)
    norm = np.linalg.norm(mean)
    if norm == 0:
        return mean
    return mean / norm


def enroll_from_video(video_data, staff_id, name, role, site_id, category="staff"):
    """Enroll a person from raw video bytes.

    Samples frames, applies the quality gate, aggregates embeddings, and
    stores the reference in the known_faces collection + FAISS index.
    """
    coll = known_faces_collection()
    existing = coll.find_one({"staff_id": staff_id, "status": "active"})
    if existing:
        raise EnrollmentError("STAFF_ALREADY_ENROLLED", f"Staff {staff_id} already enrolled")

    video_path = _save_video(video_data, staff_id)
    frames = _sample_frames(video_path, config.ENROLL_SAMPLE_FPS)
    if not frames:
        raise EnrollmentError("FACE_NOT_DETECTED", "Video could not be decoded or is empty")

    embeddings = []
    best = None
    for frame in frames:
        faces = detect_faces(frame)
        if not faces:
            continue
        largest = max(faces, key=lambda f: abs(f.bbox[2] - f.bbox[0]) * abs(f.bbox[3] - f.bbox[1]))
        scores = assess_face_scores(frame, largest.bbox, largest.landmarks)
        if not scores["ok"]:
            continue
        try:
            emb = embed_face(frame, largest)
        except Exception:
            continue
        embeddings.append(emb)
        if best is None or scores["sharpness"] > best["scores"]["sharpness"]:
            best = {"frame": frame, "bbox": largest.bbox, "scores": scores}

    if not embeddings:
        raise EnrollmentError("FACE_NOT_DETECTED", "No usable face detected in the video")
    if len(embeddings) < config.MIN_GOOD_FRAMES:
        raise EnrollmentError(
            "LOW_IMAGE_QUALITY",
            f"Only {len(embeddings)} usable frame(s); need at least {config.MIN_GOOD_FRAMES}",
        )

    ref_embedding = _aggregate(embeddings).astype('float32').tolist()
    face_id = f"fc_{uuid.uuid4().hex[:10]}"
    photo_url = None
    if best is not None:
        photo_url = _save_enrollment_snapshot(best["frame"], best["bbox"])

    doc = {
        "face_id": face_id,
        "staff_id": staff_id,
        "name": name,
        "role": role,
        "site_id": site_id,
        "category": category,
        "embedding": ref_embedding,
        "enrolled_via": "video",
        "video_url": video_path,
        "photo_url": photo_url,
        "status": "active",
        "enrolled_at": dt.datetime.now(dt.timezone.utc),
        "updated_at": dt.datetime.now(dt.timezone.utc),
    }
    coll.insert_one(doc)
    FACE_INDEX.add(np.asarray(ref_embedding, dtype='float32'), face_id)
    return face_id


def enroll_from_images(image_data_list, staff_id, name, role, site_id, category="staff"):
    """Enroll a person from a list of raw image bytes (JPEG/PNG).

    Decodes each image, applies the quality gate, aggregates embeddings,
    and stores the reference in the known_faces collection + FAISS index.
    """
    coll = known_faces_collection()
    existing = coll.find_one({"staff_id": staff_id, "status": "active"})
    if existing:
        raise EnrollmentError("STAFF_ALREADY_ENROLLED", f"Staff {staff_id} already enrolled")

    embeddings = []
    best = None
    for img_data in image_data_list:
        buf = np.frombuffer(img_data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        faces = detect_faces(frame)
        if not faces:
            continue
        largest = max(faces, key=lambda f: abs(f.bbox[2] - f.bbox[0]) * abs(f.bbox[3] - f.bbox[1]))
        scores = assess_face_scores(frame, largest.bbox, largest.landmarks)
        if not scores["ok"]:
            continue
        try:
            emb = embed_face(frame, largest)
        except Exception:
            continue
        embeddings.append(emb)
        if best is None or scores["sharpness"] > best["scores"]["sharpness"]:
            best = {"frame": frame, "bbox": largest.bbox, "scores": scores}

    if not embeddings:
        raise EnrollmentError("FACE_NOT_DETECTED", "No usable face detected in the images")

    ref_embedding = _aggregate(embeddings).astype('float32').tolist()
    face_id = f"fc_{uuid.uuid4().hex[:10]}"
    photo_url = None
    if best is not None:
        photo_url = _save_enrollment_snapshot(best["frame"], best["bbox"])

    doc = {
        "face_id": face_id,
        "staff_id": staff_id,
        "name": name,
        "role": role,
        "site_id": site_id,
        "category": category,
        "embedding": ref_embedding,
        "enrolled_via": "images",
        "photo_url": photo_url,
        "status": "active",
        "enrolled_at": dt.datetime.now(dt.timezone.utc),
        "updated_at": dt.datetime.now(dt.timezone.utc),
    }
    coll.insert_one(doc)
    FACE_INDEX.add(np.asarray(ref_embedding, dtype='float32'), face_id)
    return face_id


def enroll_from_rtsp(rtsp_url, staff_id, name, role, site_id, category="staff",
                     max_duration=10, sample_fps=2.0, min_good_frames=3,
                     sharpness_min=60.0, on_frame=None):
    """Enroll a person from a live RTSP stream.

    Captures frames for up to *max_duration* seconds, applies the quality
    gate, aggregates embeddings, and stores the reference.  Optionally
    invokes *on_frame(frame, face_bbox, frame_ok, count, needed)* for
    live preview feedback.
    """
    coll = known_faces_collection()
    existing = coll.find_one({"staff_id": staff_id, "status": "active"})
    if existing:
        raise EnrollmentError("STAFF_ALREADY_ENROLLED", f"Staff {staff_id} already enrolled")

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise EnrollmentError("RTSP_CONNECT_FAILED", f"Cannot connect to RTSP: {rtsp_url}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / max(sample_fps, 0.5))))
    embeddings = []
    best = None
    frame_idx = 0
    max_frames = int(max_duration * fps)
    good_frames_needed = max(min_good_frames, config.MIN_GOOD_FRAMES)

    try:
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                faces = detect_faces(frame)
                frame_ok = False
                face_bbox = None

                if faces:
                    largest = max(faces, key=lambda f: abs(f.bbox[2] - f.bbox[0]) * abs(f.bbox[3] - f.bbox[1]))
                    scores = assess_face_scores(frame, largest.bbox, largest.landmarks)
                    if scores["ok"]:
                        try:
                            emb = embed_face(frame, largest)
                            embeddings.append(emb)
                            frame_ok = True
                            face_bbox = largest.bbox
                            if best is None or scores["sharpness"] > best["scores"]["sharpness"]:
                                best = {"frame": frame, "bbox": largest.bbox, "scores": scores}
                        except Exception:
                            pass

                if on_frame:
                    on_frame(frame, face_bbox, frame_ok, len(embeddings), good_frames_needed)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

                if len(embeddings) >= good_frames_needed:
                    break

            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not embeddings:
        raise EnrollmentError("FACE_NOT_DETECTED", "No usable face detected in RTSP stream")
    if len(embeddings) < good_frames_needed:
        raise EnrollmentError(
            "LOW_IMAGE_QUALITY",
            f"Only {len(embeddings)} usable frame(s); need at least {good_frames_needed}",
        )

    ref_embedding = _aggregate(embeddings).astype('float32').tolist()
    face_id = f"fc_{uuid.uuid4().hex[:10]}"
    photo_url = None
    if best is not None:
        photo_url = _save_enrollment_snapshot(best["frame"], best["bbox"])

    doc = {
        "face_id": face_id,
        "staff_id": staff_id,
        "name": name,
        "role": role,
        "site_id": site_id,
        "category": category,
        "embedding": ref_embedding,
        "enrolled_via": "rtsp",
        "photo_url": photo_url,
        "status": "active",
        "enrolled_at": dt.datetime.now(dt.timezone.utc),
        "updated_at": dt.datetime.now(dt.timezone.utc),
    }
    coll.insert_one(doc)
    FACE_INDEX.add(np.asarray(ref_embedding, dtype='float32'), face_id)
    return face_id


def delete_face(face_id):
    """Soft-delete a face by setting its status to 'inactive' and removing from FAISS."""
    coll = known_faces_collection()
    doc = coll.find_one({"face_id": face_id})
    if not doc:
        return False
    coll.update_one(
        {"face_id": face_id},
        {"$set": {"status": "inactive", "updated_at": dt.datetime.now(dt.timezone.utc)}},
    )
    FACE_INDEX.remove(face_id)
    return True


def _date_str(ts):
    """Extract YYYY-MM-DD string from a datetime (Kolkata or UTC)."""
    return ts.strftime("%Y-%m-%d")


class AttendanceTracker:
    """Records first/last seen timestamps per staff member per day (Kolkata TZ).

    Uses ``$setOnInsert`` for ``first_seen_ts`` and ``$max`` for
    ``last_seen_ts`` so subsequent visits only extend the window.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._last = {}

    def record(self, staff_id, ts):
        date = _date_str(ts)
        with self._lock:
            self._last[staff_id] = ts
        coll = attendance_logs_collection()
        coll.update_one(
            {"staff_id": staff_id, "date": date},
            {
                "$setOnInsert": {
                    "staff_id": staff_id,
                    "date": date,
                    "first_seen_ts": ts,
                    "created_at": dt.datetime.now(dt.timezone.utc),
                },
                # Only update last_seen_ts (never first_seen_ts)
                "$max": {"last_seen_ts": ts},
                "$set": {"updated_at": dt.datetime.now(dt.timezone.utc)},
            },
            upsert=True,
        )

    def finalize(self, staff_id, ts):
        date = _date_str(ts)
        coll = attendance_logs_collection()
        coll.update_one(
            {"staff_id": staff_id, "date": date},
            {"$max": {"last_seen_ts": ts}},
        )

    def report_for(self, staff_id, day=None):
        q = {"staff_id": staff_id}
        if day:
            q["date"] = day
        docs = list(attendance_logs_collection().find(q).sort("date", 1))
        return self._to_report(docs)

    def report(self, site_id=None, start=None, end=None, group_by="staff"):
        coll = attendance_logs_collection()
        q = {}
        if start or end:
            q["date"] = {}
            if start:
                q["date"]["$gte"] = start
            if end:
                q["date"]["$lte"] = end
        docs = list(coll.find(q).sort("date", 1))
        return self._to_report(docs, group_by=group_by)

    @staticmethod
    def _to_report(docs, group_by="staff"):
        rows = []
        for doc in docs:
            first = doc.get("first_seen_ts")
            last = doc.get("last_seen_ts")
            hours = 0.0
            if first and last:
                delta = last - first
                hours = max(0.0, round(delta.total_seconds() / 3600, 2))
            row = {
                "staff_id": doc.get("staff_id"),
                "date": doc.get("date"),
                "first_seen_ts": first,
                "last_seen_ts": last,
                "total_hours": hours,
            }
            rows.append(row)
        return rows


class EventBus:
    """In-process bounded queue for streaming face events to WebSocket clients."""
    def __init__(self, maxlen=500):
        self._q = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def put(self, event):
        with self._lock:
            self._q.append(event)

    def drain(self, after_id=None):
        with self._lock:
            items = list(self._q)
        if after_id is None:
            return items
        for i, e in enumerate(items):
            if e.get("event_id") == after_id:
                return items[i + 1:]
        return items

    def latest_id(self):
        with self._lock:
            return self._q[-1].get("event_id") if self._q else None


event_bus = EventBus()


class GateProcessor:
    """Main gate-camera processing pipeline.

    On each frame: detect faces, track with ByteTrack, match against the
    known-face index, emit events/alerts, record attendance, count
    visitors via ROI + Re-ID, and publish snapshots.
    """
    def __init__(self, logger=None):
        self.logger = logger
        self.tracker = ByteTracker(
            track_thresh=config.DETECT_CONF,
            match_thresh=0.5,
            low_thresh=0.1,
            track_buffer=30,
        )
        self.attendance = AttendanceTracker()
        self.last_embed = {}
        self.track_labels = {}
        self.track_staff = {}
        self.unknown_counter = 0
        self._face_meta = {}
        self._lock = threading.RLock()
        # ROI for visitor counting (normalized 0-1)
        self._roi_x1 = config.ROI_X1
        self._roi_y1 = config.ROI_Y1
        self._roi_x2 = config.ROI_X2
        self._roi_y2 = config.ROI_Y2
        self._roi_iou_threshold = config.ROI_IOU_THRESHOLD
        self._roi_entered_tracks = set()
        self._roi_counted_tracks = set()
        # Counts
        self.staff_count = 0
        self.visitor_count = 0
        self._seen_staff_ids = set()
        self._visitor_embeddings = []
        self._visitor_index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
        self._visitor_map = {}
        # Kolkata timezone for attendance
        self._kolkata_tz = dt.timezone(dt.timedelta(hours=5, minutes=30))

    def log(self, msg, level="info"):
        if self.logger is not None:
            getattr(self.logger, level)(msg)

    def _emit(self, event, cam_name, ts, face):
        event.setdefault("event_id", f"fev_{os.urandom(5).hex()}")
        event.setdefault("camera_id", cam_name)
        event.setdefault("ts", ts.isoformat())
        event.setdefault("site_id", config.SITE_ID)
        event.setdefault("zone", "gate")

        snapshot_url = self._save_snapshot(ts, face)
        event["snapshot_url"] = snapshot_url

        publisher.publish_face_event(event)
        event_bus.put(event)
        self._log_event_to_db(event)
        return event

    def _save_snapshot(self, ts, face):
        try:
            os.makedirs(os.path.join(config.media_root, "snapshots"), exist_ok=True)
            meta = self._face_meta.get(id(face), {})
            frame = meta.get("frame")
            if frame is None:
                return None
            bbox = face.bbox
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            name = f"fev_{ts.strftime('%Y%m%d_%H%M%S')}_{meta.get('track_id')}.jpg"
            path = os.path.join(config.media_root, "snapshots", name)
            cv2.imwrite(path, crop)
            return path
        except Exception as e:
            self.log(f"Snapshot save error: {e}", "error")
            return None

    def _log_event_to_db(self, event):
        try:
            doc = dict(event)
            doc["ts"] = dt.datetime.fromisoformat(doc["ts"])
            face_events_collection().insert_one(doc)
        except Exception as e:
            self.log(f"face_events DB error: {e}", "error")

    @staticmethod
    def _parse_embedding(emb):
        """Parse a stored embedding into a flat float32 numpy array."""
        if emb is None:
            return None
        try:
            if isinstance(emb, dict) and "$vector" in emb:
                emb_list = emb["$vector"]
            elif isinstance(emb, str):
                try:
                    emb_list = json.loads(emb)
                except (ValueError, TypeError):
                    emb_list = np.fromstring(emb.strip("[]"), sep=" ").tolist()
            else:
                emb_list = list(emb)
            arr = np.asarray(emb_list, dtype='float32')
            return arr.flatten()
        except Exception:
            return None

    def _rebuild_visitor_index(self):
        """Rebuild the visitor FAISS index from today's visitor_logs."""
        try:
            from store import visitor_logs_collection
            docs = list(visitor_logs_collection().find({"date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")}))
            embeddings = []
            visitor_ids = []
            for doc in docs:
                emb_arr = self._parse_embedding(doc.get("embedding"))
                if emb_arr is None:
                    continue
                emb_arr = emb_arr.reshape(1, -1)
                embeddings.append(emb_arr)
                visitor_ids.append(doc.get("visitor_id", "unknown"))
            if embeddings:
                self._visitor_index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
                all_embs = np.vstack(embeddings)
                faiss.normalize_L2(all_embs)
                self._visitor_index.add(all_embs)
                self._visitor_map = {str(i): vid for i, vid in enumerate(visitor_ids)}
            else:
                self._visitor_index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
                self._visitor_map = {}
            self._visitor_embeddings = embeddings
        except Exception as e:
            self.log(f"Visitor index rebuild error: {e}", "error")
            self._visitor_index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
            self._visitor_map = {}

    def load_daily_state(self):
        """Load today's cumulative staff/visitor counts + visitor embeddings from the store."""
        try:
            from store import attendance_logs_collection, visitor_logs_collection
            today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

            # Visitors: load embeddings + count
            vdocs = list(visitor_logs_collection().find({"date": today}))
            self._visitor_embeddings = []
            visitor_ids = []
            for d in vdocs:
                emb_arr = self._parse_embedding(d.get("embedding"))
                if emb_arr is None:
                    continue
                self._visitor_embeddings.append((emb_arr, d.get("visitor_id")))
                visitor_ids.append(d.get("visitor_id"))
            if self._visitor_embeddings:
                self._visitor_index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
                all_embs = np.vstack([e[0].reshape(1, -1) for e in self._visitor_embeddings])
                faiss.normalize_L2(all_embs)
                self._visitor_index.add(all_embs)
                self._visitor_map = {str(i): vid for i, vid in enumerate(visitor_ids)}
            else:
                self._visitor_index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
                self._visitor_map = {}
            self.visitor_count = len(set(visitor_ids))

            # Staff: load unique staff_ids seen today
            adocs = list(attendance_logs_collection().find({"date": today}))
            self._seen_staff_ids = set(d.get("staff_id") for d in adocs if d.get("staff_id"))
            self.staff_count = len(self._seen_staff_ids)
        except Exception as e:
            self.log(f"Load daily state error: {e}", "error")
            self._visitor_index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
            self._visitor_map = {}
            self._seen_staff_ids = set()

    def _update_visitor_visit_count(self, visitor_id, ts):
        """Increment visit_count for a returning visitor."""
        try:
            from store import visitor_logs_collection
            visitor_logs_collection().update_one(
                {"visitor_id": visitor_id},
                {
                    "$set": {"last_seen_ts": ts},
                    "$inc": {"visit_count": 1},
                },
                upsert=True,
            )
        except Exception as e:
            self.log(f"Visitor visit count update error: {e}", "error")

    def _persist_visitor_embedding(self, embedding, ts):
        """Store a new visitor's embedding for today's Re-ID."""
        try:
            from store import visitor_logs_collection
            today = ts.strftime("%Y-%m-%d") if ts.tzinfo else ts.astimezone(dt.timezone.utc).strftime("%Y-%m-%d")
            visitor_id = f"vis_{ts.strftime('%H%M%S')}_{os.urandom(3).hex()}"
            emb_flat = np.asarray(embedding, dtype="float32").flatten().tolist()
            visitor_logs_collection().insert_one({
                "visitor_id": visitor_id,
                "date": today,
                "embedding": emb_flat,
                "visit_count": 1,
                "first_seen_ts": ts,
                "last_seen_ts": ts,
            })
            # Add to in-memory index
            emb_arr = np.asarray(embedding, dtype="float32").reshape(1, -1)
            faiss.normalize_L2(emb_arr)
            self._visitor_index.add(emb_arr)
        except Exception as e:
            self.log(f"Visitor embedding persist error: {e}", "error")

    def persist_daily_counts(self):
        """Persist the daily staff/visitor head counts to the store."""
        try:
            from store import visitor_counts_collection
            today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
            visitor_counts_collection().update_one(
                {"date": today, "site_id": config.SITE_ID},
                {
                    "$set": {
                        "staff_count": self.staff_count,
                        "visitor_count": self.visitor_count,
                        "updated_at": dt.datetime.now(dt.timezone.utc),
                    },
                    "$setOnInsert": {
                        "date": today,
                        "site_id": config.SITE_ID,
                        "created_at": dt.datetime.now(dt.timezone.utc),
                    },
                },
                upsert=True,
            )
        except Exception as e:
            self.log(f"Persist daily counts error: {e}", "error")

    def _handle_alert(self, event, cam_name, ts, frame):
        if event["match_status"] == "unknown":
            alert_id = publisher.publish_alert(
                "face_unknown_gate", "MEDIUM", cam_name,
                snapshot_url=event.get("snapshot_url"),
                meta={"track_id": event.get("track_id"), "similarity": event.get("similarity")},
            )
            event["alert_id"] = alert_id
            self.unknown_counter += 1
            if config.SAVE_UNKNOWN_IMAGES and self.unknown_counter >= config.UNKNOWN_FRAME_THRESHOLD:
                self._save_unauth_image(frame, ts, cam_name)
                self.unknown_counter = 0
            return alert_id

        if event.get("category") in ("vip", "blacklist"):
            alert_id = publisher.publish_alert(
                "face_list_match" if event.get("category") == "blacklist" else "face_vip_match",
                "INFO", cam_name,
                snapshot_url=event.get("snapshot_url"),
                meta={
                    "staff_id": event.get("staff_id"),
                    "face_id": event.get("face_id"),
                    "category": event.get("category"),
                },
            )
            event["alert_id"] = alert_id
            return alert_id
        return None

    def _save_unauth_image(self, frame, ts, cam_name):
        try:
            base = os.path.join(config.AKSHA_PATH, cam_name, "unauthorised", ts.strftime("%Y-%m-%d"))
            os.makedirs(base, exist_ok=True)
            path = os.path.join(base, f"unauth_{ts.strftime('%H%M%S')}.jpg")
            cv2.imwrite(path, frame)
            self.log(f"Unauthorised image saved: {path}", "warning")
        except Exception as e:
            self.log(f"Unauth image error: {e}", "error")

    def _save_alert_image(self, frame, ts, cam_name):
        try:
            base = os.path.join(config.AKSHA_PATH, cam_name, "alerts", ts.strftime("%Y-%m-%d"))
            os.makedirs(base, exist_ok=True)
            path = os.path.join(base, f"{ts.strftime('%Y-%m-%d_%H-%M-%S')}_alert.jpg")
            cv2.imwrite(path, frame)
            self.log(f"Alert image saved: {path}", "warning")
        except Exception as e:
            self.log(f"Alert image error: {e}", "error")

    def _publish_monitor(self, session, cam_name, ts, frame):
        try:
            live_dir = os.path.join(config.AKSHA_PATH, cam_name, "live")
            os.makedirs(live_dir, exist_ok=True)
            image_path = os.path.join(live_dir, "workday.jpg")
            cv2.imwrite(image_path, frame)
            data = {
                "camera_name": cam_name,
                "timestamp": ts.isoformat(),
                "image_type": "workday",
            }
            with open(image_path, "rb") as fh:
                resp = session.post(
                    config.MONITOR_API,
                    files={"image": fh},
                    data=data,
                    timeout=(3, 3),
                )
            self.log(f"Monitor API published (status={resp.status_code})")
        except Exception as e:
            self.log(f"Monitor API error: {e}", "error")

    def _log_legacy_db(self, cam_name, ts, detections, alert_triggered):
        try:
            coll = load_database(cam_name)
            if coll is not None:
                coll.insert_one({
                    "Timestamp": ts,
                    "CameraName": cam_name,
                    "FaceRecognitionResults": detections,
                    "AlertTriggered": alert_triggered,
                })
        except Exception as e:
            self.log(f"Legacy DB error: {e}", "error")

    def process_frame(self, frame, cam_name, ts):
        faces = detect_faces(frame)
        self._face_meta = {id(f): {"frame": frame, "track_id": None} for f in faces}

        dets = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(float)
            dets.append([x1, y1, x2, y2, float(f.confidence)])

        dets_arr = np.asarray(dets, dtype=np.float64).reshape(-1, 5) if dets else np.empty((0, 5))
        tracked = self.tracker.update(dets_arr)

        events = []
        now = ts
        frame_h, frame_w = frame.shape[:2]
        roi_x1_px = self._roi_x1 * frame_w
        roi_y1_px = self._roi_y1 * frame_h
        roi_x2_px = self._roi_x2 * frame_w
        roi_y2_px = self._roi_y2 * frame_h

        for row in tracked:
            x1, y1, x2, y2, track_id = map(float, row)
            track_id = int(track_id)
            face = self._match_face(faces, (x1, y1, x2, y2))
            if face is None:
                continue
            self._face_meta[id(face)]["track_id"] = track_id

            last = self.last_embed.get(track_id)
            cooldown = config.REMATCH_COOLDOWN_SEC
            if last is not None and (now - last).total_seconds() < cooldown:
                continue

            try:
                embedding = embed_face(frame, face)
            except Exception as e:
                self.log(f"Embed error: {e}", "error")
                continue
            self.last_embed[track_id] = now

            match = FACE_INDEX.search(embedding)
            if match is None:
                event = {
                    "track_id": track_id,
                    "face_id": None,
                    "staff_id": None,
                    "category": None,
                    "match_status": "unknown",
                    "similarity": None,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            else:
                face_id, similarity = match
                if similarity < config.SIMILARITY_THRESHOLD:
                    event = {
                        "track_id": track_id,
                        "face_id": None,
                        "staff_id": None,
                        "category": None,
                        "match_status": "unknown",
                        "similarity": round(similarity, 4),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    }
                else:
                    known = self._known_doc(face_id)
                    event = {
                        "track_id": track_id,
                        "face_id": face_id,
                        "staff_id": known.get("staff_id") if known else None,
                        "name": known.get("name") if known else None,
                        "category": known.get("category") if known else None,
                        "match_status": "matched",
                        "similarity": round(similarity, 4),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    }

            # --- ROI-based visitor counting with Re-ID dedup ---
            if event["match_status"] == "unknown" and config.ROI_ENABLED:
                fx1, fy1, fx2, fy2 = float(x1), float(y1), float(x2), float(y2)
                overlap_x1 = max(fx1, roi_x1_px)
                overlap_y1 = max(fy1, roi_y1_px)
                overlap_x2 = min(fx2, roi_x2_px)
                overlap_y2 = min(fy2, roi_y2_px)
                overlap_area = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)
                face_area = (fx2 - fx1) * (fy2 - fy1)
                overlap_pct = overlap_area / face_area if face_area > 0 else 0
                if overlap_pct >= self._roi_iou_threshold:
                    # Re-ID dedup: check embedding against known visitors today
                    is_returning = False
                    if embedding is not None and self._visitor_index.ntotal > 0:
                        try:
                            q = embedding.reshape(1, -1).astype("float32")
                            faiss.normalize_L2(q)
                            D, I = self._visitor_index.search(q, 1)
                            if D[0][0] >= config.VISITOR_REID_THRESHOLD:
                                is_returning = True
                        except Exception:
                            pass
                    if not is_returning:
                        # Only count if this track_id hasn't been counted yet
                        if track_id not in self._roi_counted_tracks:
                            self._roi_counted_tracks.add(track_id)
                            self.visitor_count += 1
                            # Persist new visitor embedding
                            if embedding is not None:
                                self._persist_visitor_embedding(embedding, now)
            # --- End ROI tracking ---

            event = self._emit(event, cam_name, ts, face)
            alert_id = self._handle_alert(event, cam_name, ts, frame)
            if alert_id:
                self._save_alert_image(frame, ts, cam_name)

            if event["match_status"] == "matched" and event.get("staff_id"):
                kolkata_ts = now.astimezone(self._kolkata_tz)
                self.attendance.record(event["staff_id"], kolkata_ts)
                self.track_staff[track_id] = event["staff_id"]
                if event["staff_id"] not in self._seen_staff_ids:
                    self.staff_count += 1
                    self._seen_staff_ids.add(event["staff_id"])

            label = event.get("name") or event["face_id"] or "Unknown"
            if event["match_status"] == "unknown":
                self.track_labels[track_id] = ("Unknown", 0.0, (0, 0, 255))
            else:
                self.track_labels[track_id] = (label, event.get("similarity", 0.0), (0, 255, 0))
            events.append(event)

        active_track_ids = {int(r[4]) for r in tracked}
        for track_id, staff_id in list(self.track_staff.items()):
            if track_id not in active_track_ids:
                self.attendance.finalize(staff_id, now.astimezone(self._kolkata_tz))
                self.track_staff.pop(track_id, None)

        if not any(e.get("match_status") == "unknown" for e in events):
            self.unknown_counter = 0

        display = self._draw(frame, faces)
        self._publish_monitor_legacy(display, cam_name, now)

        any_unknown = any(e.get("match_status") == "unknown" for e in events)
        any_alert = any(e.get("alert_id") for e in events)
        legacy_dets = [
            {
                "box": e.get("bbox"),
                "label": e.get("name") or e.get("face_id") or "Unknown",
                "Score": round(e.get("similarity") or 0.0, 2),
            }
            for e in events
        ]
        self._log_legacy_db(cam_name, now, legacy_dets, any_alert)
        return events

    def _match_face(self, faces, tlbr):
        best, best_iou = None, 0.0
        for f in faces:
            b = f.bbox.astype(float)
            v = iou(tuple(b), tuple(tlbr))
            if v > best_iou:
                best, best_iou = f, v
        return best if best_iou >= 0.3 else None

    def _known_doc(self, face_id):
        try:
            return known_faces_collection().find_one({"face_id": face_id})
        except Exception:
            return None

    def _draw(self, frame, faces):
        for f in faces:
            track_id = self._face_meta.get(id(f), {}).get("track_id")
            if track_id is None:
                continue
            label, score, color = self.track_labels.get(track_id, ("Person", 0.0, (255, 255, 0)))
            x1, y1, x2, y2 = map(int, f.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} ({score})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        return frame

    def _publish_monitor_legacy(self, frame, cam_name, ts):
        if config.is_local_sink or not config.PUBLISH_MONITOR:
            return
        try:
            with requests.Session() as session:
                self._publish_monitor(session, cam_name, ts, frame)
        except Exception as e:
            self.log(f"Monitor publish error: {e}", "error")