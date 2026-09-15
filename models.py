"""Face detection, recognition, quality assessment, and object tracking.

Provides the core ML components used by the gate processor:
- ``FaceIndex``: FAISS-backed vector index for known-face lookup.
- ``ByteTracker``: ByteTrack-style multi-object tracker for face bboxes.
- ``assess_face`` / ``assess_face_scores``: quality gate (size, sharpness,
  brightness, frontal angle) used during enrollment and live processing.
- ``detect_faces``, ``embed_face``, ``analyze_frame``: thin wrappers around
  the uniface ``FaceAnalyzer``.
"""

import math
import os
import pickle
import threading
from enum import IntEnum

import cv2
import faiss
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from uniface import FaceAnalyzer
from uniface.detection import RetinaFace
from uniface.recognition import ArcFace

from config import config

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EMBEDDING_DIM = config.EMBEDDING_DIM
LEGACY_DB_PATH = './faiss_database.pkl'


class FaceIndex:
    """Thread-safe FAISS inner-product index mapping face_id -> embedding.

    Supports add, remove (soft-delete via rebuild), and top-1 search.
    Embeddings are L2-normalised before insertion so dot-product equals cosine.
    """
    def __init__(self, dim=EMBEDDING_DIM):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self._id_to_face = {}
        self._face_to_ids = {}
        self._lock = threading.RLock()

    def _normalize(self, embedding):
        v = np.asarray(embedding, dtype='float32').reshape(1, -1)
        faiss.normalize_L2(v)
        return v

    @property
    def size(self):
        with self._lock:
            return self.index.ntotal

    def add(self, embedding, face_id):
        with self._lock:
            v = self._normalize(embedding)
            faiss_id = self.index.ntotal
            self.index.add(v)
            self._id_to_face[faiss_id] = face_id
            self._face_to_ids.setdefault(face_id, []).append(faiss_id)

    def add_many(self, embeddings, face_ids):
        if not embeddings:
            return
        with self._lock:
            vs = np.asarray(embeddings, dtype='float32').reshape(len(embeddings), -1)
            faiss.normalize_L2(vs)
            start = self.index.ntotal
            self.index.add(vs)
            for offset, face_id in enumerate(face_ids):
                faiss_id = start + offset
                self._id_to_face[faiss_id] = face_id
                self._face_to_ids.setdefault(face_id, []).append(faiss_id)

    def remove(self, face_id):
        with self._lock:
            ids = self._face_to_ids.pop(face_id, [])
            if not ids:
                return False
            remove_set = set(ids)
            keep = [self._id_to_face[i] for i in range(self.index.ntotal) if i not in remove_set]
            self._rebuild_from(keep)
            return True

    def clear(self):
        with self._lock:
            self.index = faiss.IndexFlatIP(self.dim)
            self._id_to_face = {}
            self._face_to_ids = {}

    def rebuild(self, known_faces):
        with self._lock:
            self.clear()
            embeddings, face_ids = [], []
            for doc in known_faces:
                emb = doc.get("embedding")
                if emb is None:
                    continue
                embeddings.append(emb)
                face_ids.append(doc["face_id"])
            self.add_many(embeddings, face_ids)

    def _rebuild_from(self, face_ids):
        keep_vectors = []
        for face_id in face_ids:
            for faiss_id in self._face_to_ids.get(face_id, []):
                keep_vectors.append((face_id, self.index.reconstruct(faiss_id).reshape(-1)))
        new_index = faiss.IndexFlatIP(self.dim)
        if keep_vectors:
            new_index.add(np.asarray([v for _, v in keep_vectors], dtype='float32'))
        self.index = new_index
        self._id_to_face = {}
        self._face_to_ids = {}
        for pos, (face_id, _) in enumerate(keep_vectors):
            self._id_to_face[pos] = face_id
            self._face_to_ids.setdefault(face_id, []).append(pos)

    def search(self, embedding, k=1):
        with self._lock:
            if self.index.ntotal == 0:
                return None
            v = self._normalize(embedding)
            D, I = self.index.search(v, k)
        best = int(I[0][0])
        if best == -1 or best not in self._id_to_face:
            return None
        return self._id_to_face[best], float(D[0][0])

    def face_ids(self):
        with self._lock:
            return list(self._face_to_ids.keys())


FACE_INDEX = FaceIndex()


def bbox_size_ok(bbox):
    """Return True if the bounding box meets the minimum pixel size."""
    try:
        x1, y1, x2, y2 = map(int, bbox)
    except (TypeError, ValueError):
        return False
    return (x2 - x1) >= config.FACE_MIN_SIZE and (y2 - y1) >= config.FACE_MIN_SIZE


def _safe_crop(frame, bbox):
    """Crop a bounding-box region from *frame*, clamped to image bounds."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def sharpness_score(frame, bbox):
    """Laplacian variance of the cropped face region (higher = sharper)."""
    crop = _safe_crop(frame, bbox)
    if crop is None:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(frame, bbox):
    """Mean pixel intensity of the cropped face region."""
    crop = _safe_crop(frame, bbox)
    if crop is None:
        return 0.0
    return float(crop.mean())


def frontal_score(landmarks):
    """Estimate how frontal a face is from eye landmarks (0..1)."""
    if landmarks is None:
        return 0.5
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.shape[0] < 2:
        return 0.5
    left_eye, right_eye = lm[0], lm[1]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    if abs(dx) < 1e-6:
        return 0.0
    angle = abs(math.atan2(dy, dx))
    return max(0.0, min(1.0, 1.0 - angle / 0.6))


def assess_face(frame, bbox, landmarks=None):
    """Quality gate: returns (ok: bool, reason: str)."""
    if not bbox_size_ok(bbox):
        return False, "small_face"

    sharp = sharpness_score(frame, bbox)
    if sharp < config.SHARPNESS_MIN:
        return False, "blurry"

    bright = brightness_score(frame, bbox)
    if bright < config.BRIGHTNESS_MIN or bright > config.BRIGHTNESS_MAX:
        return False, "bad_lighting"

    frontal = frontal_score(landmarks)
    if frontal < config.FRONTAL_MIN:
        return False, "not_frontal"

    return True, "ok"


def assess_face_scores(frame, bbox, landmarks=None):
    """Return a dict with individual quality scores for diagnostics."""
    ok, reason = assess_face(frame, bbox, landmarks)
    return {
        "ok": ok,
        "reason": reason,
        "sharpness": round(sharpness_score(frame, bbox), 2),
        "brightness": round(brightness_score(frame, bbox), 2),
        "frontal": round(frontal_score(landmarks), 2),
    }


class TrackState(IntEnum):
    """Lifecycle states for a tracked object."""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """Single-object track state (bbox, velocity, lifecycle)."""
    _count = 0

    def __init__(self, tlbr, score):
        self.tlbr = np.asarray(tlbr, dtype=np.float64).copy()
        self.score = float(score)
        self.state = TrackState.New
        self.track_id = -1
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0
        self.is_activated = False
        self.last_embed_ts = None
        self.last_match = None
        self.velocity = np.zeros(4)
        self.lost_frames = 0

    def activate(self, track_id, frame_id):
        self.track_id = track_id
        self.state = TrackState.Tracked
        self.start_frame = frame_id
        self.frame_id = frame_id
        self.is_activated = True

    def reactivate(self, frame_id):
        self.state = TrackState.Tracked
        self.is_activated = True
        self.lost_frames = 0
        self.frame_id = frame_id

    def mark_lost(self):
        self.state = TrackState.Lost
        self.lost_frames = 0

    def mark_removed(self):
        self.state = TrackState.Removed

    def predict(self):
        self.tlbr = self.tlbr + self.velocity
        self.lost_frames += 1

    def update(self, tlbr, score, frame_id):
        if self.tlbr.shape == tlbr.shape:
            self.velocity = 0.6 * self.velocity + 0.4 * (tlbr - self.tlbr)
        self.tlbr = np.asarray(tlbr, dtype=np.float64).copy()
        self.score = float(score)
        self.state = TrackState.Tracked
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.lost_frames = 0


def iou(a, b):
    """Intersection-over-Union between two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def iou_matrix(tlbrs_a, tlbrs_b):
    """Pairwise IoU matrix between two sets of boxes."""
    if len(tlbrs_a) == 0 or len(tlbrs_b) == 0:
        return np.zeros((len(tlbrs_a), len(tlbrs_b)))
    out = np.zeros((len(tlbrs_a), len(tlbrs_b)))
    for i, a in enumerate(tlbrs_a):
        for j, b in enumerate(tlbrs_b):
            out[i, j] = iou(a, b)
    return out


class ByteTracker:
    """ByteTrack-style multi-object tracker using IoU + Hungarian matching.

    Maintains three pools: *tracked*, *lost*, and *removed*.  High-confidence
    detections are matched first; remaining high-confidence tracks are matched
    against low-confidence detections as a second stage.
    """
    def __init__(self, track_thresh=0.5, match_thresh=0.6, low_thresh=0.1,
                 track_buffer=30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.low_thresh = low_thresh
        self.track_buffer = track_buffer
        self.frame_id = 0
        self.tracked = []
        self.lost = []
        self.removed = []
        self._next_id = 1

    def reset(self):
        self.frame_id = 0
        self.tracked = []
        self.lost = []
        self.removed = []
        self._next_id = 1

    def _new_id(self):
        tid = self._next_id
        self._next_id += 1
        return tid

    def _associate(self, tracks, detections, thresh):
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        dists = iou_matrix(
            [t.tlbr for t in tracks],
            [d.tlbr for d in detections],
        )
        cost = 1.0 - dists
        row, col = linear_sum_assignment(cost)
        matches, unmatched_t, unmatched_d = [], set(range(len(tracks))), set(range(len(detections)))
        for r, c in zip(row, col):
            if dists[r, c] >= thresh:
                matches.append((r, c))
                unmatched_t.discard(r)
                unmatched_d.discard(c)
        return matches, sorted(unmatched_t), sorted(unmatched_d)

    def update(self, dets):
        self.frame_id += 1
        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 5)
        if dets.size == 0:
            dets = np.empty((0, 5))

        scores = dets[:, 4]
        high_inds = scores >= self.track_thresh
        low_inds = (scores > self.low_thresh) & (scores < self.track_thresh)

        dets_high = [STrack(dets[i, :4], dets[i, 4]) for i in np.where(high_inds)[0]]
        dets_low = [STrack(dets[i, :4], dets[i, 4]) for i in np.where(low_inds)[0]]

        for t in self.tracked + self.lost:
            t.predict()

        pool = self.tracked + self.lost
        matched_pool = set()
        matches, u_pool, u_high = self._associate(pool, dets_high, self.match_thresh)
        for ip, ih in matches:
            track = pool[ip]
            if track.state == TrackState.Lost:
                track.reactivate(self.frame_id)
            track.update(dets_high[ih].tlbr, dets_high[ih].score, self.frame_id)
            matched_pool.add(ip)

        matched_second = set()
        if u_pool:
            candidates = [(i, pool[i]) for i in u_pool if pool[i].state == TrackState.Tracked]
            if candidates:
                m2, u2, _ = self._associate(
                    [t for _, t in candidates], dets_low, 0.5
                )
                for ic, il in m2:
                    track = candidates[ic][1]
                    track.update(dets_low[il].tlbr, dets_low[il].score, self.frame_id)
                    matched_second.add(candidates[ic][0])

        for i, track in enumerate(self.tracked):
            if i not in matched_pool and i not in matched_second:
                if track.state == TrackState.Tracked:
                    track.mark_lost()

        for i in u_high:
            d = dets_high[i]
            if d.score < self.track_thresh:
                continue
            d.activate(self._new_id(), self.frame_id)
            self.tracked.append(d)

        self.lost = [t for t in self.lost if self.frame_id - t.start_frame <= self.track_buffer]

        self.tracked = [t for t in self.tracked + self.lost if t.state == TrackState.Tracked]
        self.lost = [t for t in self.lost if t.state == TrackState.Lost]

        outputs = []
        for t in self.tracked:
            if t.is_activated:
                outputs.append(np.concatenate([t.tlbr, [t.track_id]]))
        return np.asarray(outputs).reshape(-1, 5) if outputs else np.empty((0, 5))

    def get_track(self, track_id):
        for t in self.tracked:
            if t.track_id == track_id:
                return t
        for t in self.lost:
            if t.track_id == track_id:
                return t
        return None


ANALYZER = FaceAnalyzer(
    detector=RetinaFace(confidence_threshold=config.DETECT_CONF),
    recognizer=ArcFace(),
)


def detect_faces(frame):
    """Run RetinaFace detection on *frame*, return list of Face objects."""
    return ANALYZER.detector.detect(frame)


def embed_face(frame, face):
    """Compute a 512-d ArcFace embedding for a detected face."""
    return ANALYZER.recognizer.get_normalized_embedding(frame, face.landmarks)


def analyze_frame(frame):
    """Detect faces and compute embeddings in a single pass."""
    faces = detect_faces(frame)
    for face in faces:
        try:
            face.embedding = embed_face(frame, face)
        except Exception:
            face.embedding = None
    return faces


def load_legacy_faiss_db(analyzer, reference_dir, db_path):
    """Load or build a legacy FAISS pickle database from a reference directory."""
    if os.path.exists(db_path):
        with open(db_path, 'rb') as f:
            data = pickle.load(f)
        index = faiss.deserialize_index(data['index'])
        id_to_name = data['id_to_name']
        return index, id_to_name

    embeddings_list = []
    id_to_name = []
    if not os.path.exists(reference_dir):
        return None, []

    for person_name in os.listdir(reference_dir):
        person_path = os.path.join(reference_dir, person_name)
        if not os.path.isdir(person_path):
            continue
        images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_name in images:
            img = cv2.imread(os.path.join(person_path, img_name))
            if img is None:
                continue
            faces = analyzer.analyze(img)
            if faces:
                emb = faces[0].embedding.flatten().astype('float32')
                embeddings_list.append(emb)
                id_to_name.append(person_name)
        print(f"  [SUCCESS] Enrolled {person_name}")

    embeddings_np = np.array(embeddings_list).astype('float32')
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    if embeddings_np.size:
        index.add(embeddings_np)
    with open(db_path, 'wb') as f:
        pickle.dump({'index': faiss.serialize_index(index), 'id_to_name': id_to_name}, f)
    return index, id_to_name


def process_single_frame_for_api(frame):
    """Process a single frame and return a list of detection dicts for the API."""
    if frame is None:
        return []
    current_faces = analyze_frame(frame)
    results = []
    if not current_faces:
        return results

    for i, face in enumerate(current_faces):
        try:
            emb = face.embedding
            if emb is None:
                continue
            match = FACE_INDEX.search(emb, k=1)
            if match is None:
                label, best_score = "Unknown", 0.0
            else:
                face_id, best_score = match
                if best_score >= config.SIMILARITY_THRESHOLD:
                    label = face_id
                else:
                    label = "Unknown"
            x_min, y_min, x_max, y_max = face.bbox.astype(int)
            results.append({
                "box": [int(x_min), int(y_min), int(x_max), int(y_max)],
                "label": label,
                "Score": round(best_score, 2),
            })
        except Exception as e:
            print(f"Error processing face {i}: {e}")
            bbox = face.bbox.astype(int)
            results.append({
                "box": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                "label": "ERROR",
                "Score": 0.0,
            })
    return results


def build_index_from_db(known_faces_coll):
    """Rebuild the global FACE_INDEX from all active docs in *known_faces_coll*."""
    docs = list(known_faces_coll.find({"status": "active"}))
    FACE_INDEX.rebuild(docs)
    return FACE_INDEX