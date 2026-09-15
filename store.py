"""Storage abstraction layer for face-recognition data.

Routes reads/writes to either MongoDB (production) or local JSONL files
(offline/testing) based on the ``STORE_BACKEND`` and ``OUTPUT_SINK``
config settings.  Also provides a Kafka-based ``FaceEventsPublisher``
for streaming events and alerts.
"""
import json
import os
import threading
import uuid

import pymongo
from kafka import KafkaProducer

from config import config
from localstore import (
    local_attendance_logs,
    local_face_events,
    local_known_faces,
    local_visitor_logs,
    local_visitor_counts,
    get_local_store,
)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = pymongo.MongoClient(config.MONGO_URI, directConnection=True)
    return _client


def get_database():
    return _get_client()[config.MONGO_DB]


def known_faces_collection():
    if config.is_local:
        return local_known_faces()
    return get_database()["known_faces"]


def face_events_collection():
    if config.is_local:
        return local_face_events()
    return get_database()["face_events"]


def attendance_logs_collection():
    if config.is_local:
        return local_attendance_logs()
    return get_database()["attendance_logs"]


def visitor_logs_collection():
    if config.is_local:
        return local_visitor_logs()
    return get_database()["visitor_logs"]


def visitor_counts_collection():
    if config.is_local:
        return local_visitor_counts()
    return get_database()["visitor_counts"]


def ensure_indexes():
    """Create MongoDB indexes for known_faces, face_events, and attendance_logs.

    No-op when running in local/file mode.
    """
    if config.is_local:
        return
    known_faces_collection().create_index([("staff_id", 1)], unique=True, sparse=True)
    known_faces_collection().create_index([("face_id", 1)], unique=True)
    known_faces_collection().create_index([("status", 1)])
    face_events_collection().create_index([("camera_id", 1), ("ts", -1)])
    face_events_collection().create_index([("event_id", 1)], unique=True)
    attendance_logs_collection().create_index([("staff_id", 1), ("date", 1)], unique=True)


def load_database(camera_name):
    """Return the per-camera face-metadata collection.

    In local mode this is a ``FileCollection`` backed by JSONL; in
    production it is a MongoDB collection named ``facemeta_<camera_name>``.
    """
    try:
        if config.is_local:
            return get_local_store().collection(f"facemeta_{camera_name}")
        return get_database()[f"facemeta_{camera_name}"]
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None


def _ts():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _short_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class FaceEventsPublisher:
    """Publishes face events and alerts to Kafka (or local JSONL in offline mode).

    Thread-safe; lazily initialises the ``KafkaProducer`` on first use.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._producer = None

    def _get_producer(self):
        if self._producer is None:
            self._producer = KafkaProducer(
                bootstrap_servers=[config.kafka_servers()],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            )
        return self._producer

    def _write_local(self, kind, payload):
        out_dir = config.local_out_dir
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{kind}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")

    def publish_face_event(self, event):
        event.setdefault("event_id", _short_id("fev"))
        event.setdefault("ts", _ts())
        event.setdefault("site_id", config.SITE_ID)
        event.setdefault("zone", "gate")
        if config.is_local_sink:
            self._write_local("face_events", event)
            return
        with self._lock:
            try:
                self._get_producer().send(config.FACE_EVENTS_TOPIC, event)
            except Exception as e:
                print(f"Kafka face_events publish error: {e}")

    def publish_alert(self, alert_type, severity, camera_id, snapshot_url=None, meta=None):
        alert = {
            "alert_id": _short_id("alt"),
            "type": alert_type,
            "severity": severity,
            "camera_id": camera_id,
            "site_id": config.SITE_ID,
            "ts": _ts(),
            "snapshot_url": snapshot_url,
            "status": "open",
            "source": "face",
            "meta": meta or {},
        }
        if config.is_local_sink:
            self._write_local("alerts", alert)
            return alert["alert_id"]
        with self._lock:
            try:
                self._get_producer().send(config.ALERTS_TOPIC, alert)
            except Exception as e:
                print(f"Kafka alerts publish error: {e}")
        return alert["alert_id"]

    def close(self):
        with self._lock:
            if self._producer is not None:
                try:
                    self._producer.flush()
                    self._producer.close()
                except Exception:
                    pass
                self._producer = None


publisher = FaceEventsPublisher()