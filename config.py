"""Centralised configuration for the Aksha face-recognition service.

All settings are read from environment variables with sensible defaults.
The singleton ``config`` at module level is the canonical source of truth
used by every other module in this package.
"""

import os
import datetime as dt


def _env_bool(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class Config:
    """Application-wide configuration loaded from environment variables."""
    MONGO_URI = os.environ.get(
        "MONGO_URI",
        "mongodb://mongo:mongo@mongodb:27017/Aksha?authSource=admin&tls=false",
    )
    MONGO_DB = os.environ.get("MONGO_DB", "Aksha")

    KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "")
    INPUT_TOPIC = os.environ.get("INPUT_TOPIC", "raw_frame")
    FACE_EVENTS_TOPIC = os.environ.get("FACE_EVENTS_TOPIC", "face_events")
    ALERTS_TOPIC = os.environ.get("ALERTS_TOPIC", "alerts")

    GATE_CAM = os.environ.get("GATE_CAM", "gate_cam")
    SITE_ID = os.environ.get("SITE_ID", "site_default")

    SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.8"))
    REMATCH_COOLDOWN_SEC = float(os.environ.get("REMATCH_COOLDOWN_SEC", "2.5"))
    FACE_MIN_SIZE = int(os.environ.get("FACE_MIN_SIZE", "32"))
    DETECT_CONF = float(os.environ.get("DETECT_CONF", "0.5"))
    FRONTAL_MIN = float(os.environ.get("FRONTAL_MIN", "0.35"))
    SHARPNESS_MIN = float(os.environ.get("SHARPNESS_MIN", "60.0"))
    BRIGHTNESS_MIN = float(os.environ.get("BRIGHTNESS_MIN", "40.0"))
    BRIGHTNESS_MAX = float(os.environ.get("BRIGHTNESS_MAX", "230.0"))
    EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "512"))
    MIN_GOOD_FRAMES = int(os.environ.get("MIN_GOOD_FRAMES", "3"))
    ENROLL_SAMPLE_FPS = float(os.environ.get("ENROLL_SAMPLE_FPS", "2.0"))
    UNKNOWN_FRAME_THRESHOLD = int(os.environ.get("UNKNOWN_FRAME_THRESHOLD", "5"))
    SAVE_UNKNOWN_IMAGES = _env_bool("SAVE_UNKNOWN_IMAGES", True)
    ROI_ENABLED = _env_bool("ROI_ENABLED", False)
    ROI_X1 = float(os.environ.get("ROI_X1", "0.4"))
    ROI_Y1 = float(os.environ.get("ROI_Y1", "0.2"))
    ROI_X2 = float(os.environ.get("ROI_X2", "0.55"))
    ROI_Y2 = float(os.environ.get("ROI_Y2", "0.5"))
    ROI_IOU_THRESHOLD = float(os.environ.get("ROI_IOU_THRESHOLD", "0.8"))
    VISITOR_REID_THRESHOLD = float(os.environ.get("VISITOR_REID_THRESHOLD", "0.35"))
    API_PORT = int(os.environ.get("API_PORT", "8000"))
    API_HOST = os.environ.get("API_HOST", "0.0.0.0")
    API_TOKEN = os.environ.get("API_TOKEN", "1234")
    PROCESS_FPS = int(os.environ.get("PROCESS_FPS", "10"))
    MONITOR_API = os.environ.get("MONITOR_API", "http://node_backend:5000/api/monitor/")

    # Date-based folder for organizing outputs
    _CURRENT_DATE = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

    @property
    def date_folder(self):
        return self._CURRENT_DATE

    AKSHA_PATH = os.environ.get("AKSHA_PATH", "./")
    MEDIA_ROOT = os.environ.get("MEDIA_ROOT", "")

    STORE_BACKEND = os.environ.get("STORE_BACKEND", "mongo")
    LOCAL_STORE_DIR = os.environ.get("LOCAL_STORE_DIR", "")
    OUTPUT_SINK = os.environ.get("OUTPUT_SINK", "kafka")
    LOCAL_OUT_DIR = os.environ.get("LOCAL_OUT_DIR", "")
    PUBLISH_MONITOR = _env_bool("PUBLISH_MONITOR", True)

    @property
    def local_store_dir(self):
        if self.LOCAL_STORE_DIR:
            return self.LOCAL_STORE_DIR
        return os.path.join(self.media_root, "local_store")

    @property
    def local_out_dir(self):
        if self.LOCAL_OUT_DIR:
            return self.LOCAL_OUT_DIR
        return os.path.join(self.media_root, "local_out")

    @property
    def is_local(self):
        return self.STORE_BACKEND.lower() in ("local", "file", "offline")

    @property
    def is_local_sink(self):
        return self.OUTPUT_SINK.lower() in ("local", "file", "offline")

    @property
    def media_root(self):
        if self.MEDIA_ROOT:
            return self.MEDIA_ROOT
        return os.path.join(self.AKSHA_PATH, "faces_media")

    def kafka_servers(self):
        if self.KAFKA_BOOTSTRAP_SERVERS:
            return self.KAFKA_BOOTSTRAP_SERVERS
        if os.path.exists("/.dockerenv"):
            return "broker:9092"
        return "localhost:9092"


config = Config()
