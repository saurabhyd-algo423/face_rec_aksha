"""Local file-backed storage for offline / testing mode.

Provides ``FileCollection`` (a JSONL-backed drop-in for pymongo
``Collection``) and ``LocalStore`` which organises collections into
date-based subfolders under ``<media_root>/local_store/<YYYY-MM-DD>/``.
"""

import datetime as dt
import json
import os
import re
import threading

from config import config
import datetime as dt

_OPS = ("$gte", "$lte", "$gt", "$lt")


def _enc(o):
    """Encode a Python object for JSON serialisation (handles datetimes)."""
    if isinstance(o, (dt.datetime, dt.date)):
        return {"$iso": o.isoformat()}
    if isinstance(o, dict):
        return {k: _enc(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_enc(x) for x in o]
    return o


def _dec(o):
    """Decode a JSON-parsed object back to native Python types (restores datetimes)."""
    if isinstance(o, dict):
        if set(o.keys()) == {"$iso"}:
            try:
                return dt.datetime.fromisoformat(o["$iso"])
            except (TypeError, ValueError):
                return o["$iso"]
        return {k: _dec(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_dec(x) for x in o]
    return o


def _contains_op(v):
    """Return True if *v* is a MongoDB-style operator dict ($gte, $lte, etc.)."""
    return isinstance(v, dict) and any(op in v for op in _OPS)


def _value_ok(dv, cond):
    """Evaluate a single MongoDB-style condition ($gte, $lte, $gt, $lt)."""
    op = "$gte"
    if "$gte" in cond:
        result = dv >= cond["$gte"]
    if "$lte" in cond:
        result = result and dv <= cond["$lte"]
    if "$gt" in cond:
        result = result and dv > cond["$gt"]
    if "$lt" in cond:
        result = result and dv < cond["$lt"]
    return result


def _match(doc, query):
    """Evaluate a MongoDB-style query dict against a document."""
    for key, cond in query.items():
        if key == "$or":
            if not any(_match(doc, sub) for sub in cond):
                return False
            continue
        if isinstance(cond, dict) and "$regex" in cond:
            val = doc.get(key)
            if val is None or not re.search(cond["$regex"], str(val), re.IGNORECASE if cond.get("$options") == "i" else 0):
                return False
            continue
        if _contains_op(cond):
            if not _value_ok(doc.get(key), cond):
                return False
            continue
        if doc.get(key) != cond:
            return False
    return True


def _apply_update(doc, update):
    """Apply a MongoDB-style update ($set, $setOnInsert, $max, $min) to *doc*."""
    for op, fields in update.items():
        if op == "$set":
            for k, v in fields.items():
                doc[k] = v
        elif op == "$setOnInsert":
            for k, v in fields.items():
                doc.setdefault(k, v)
        elif op == "$max":
            for k, v in fields.items():
                if doc.get(k) is None or v > doc.get(k):
                    doc[k] = v
        elif op == "$min":
            for k, v in fields.items():
                if doc.get(k) is None or v < doc.get(k):
                    doc[k] = v


class Cursor:
    """Minimal pymongo-compatible cursor over an in-memory list of dicts."""
    def __init__(self, docs):
        self._docs = docs

    def sort(self, field, direction=1):
        rev = direction < 0
        self._docs = sorted(
            self._docs,
            key=lambda d: (d.get(field) is None, str(d.get(field))),
            reverse=rev,
        )
        return self

    def skip(self, n):
        self._docs = self._docs[n:]
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    def __iter__(self):
        return iter(self._docs)

    def __len__(self):
        return len(self._docs)


class FileCollection:
    """JSONL-backed collection with a pymongo-compatible subset of methods.

    Thread-safe; serialises datetimes via ``_enc``/``_dec`` helpers.
    """
    def __init__(self, path):
        self.path = path
        self._lock = threading.RLock()
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write("")

    def _load(self):
        docs = []
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        docs.append(_dec(json.loads(line)))
                    except ValueError:
                        continue
        return docs

    def _save(self, docs):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(_enc(doc), default=str) + "\n")

    def create_index(self, *args, **kwargs):
        return None

    def insert_one(self, doc):
        with self._lock:
            docs = self._load()
            docs.append(dict(doc))
            self._save(docs)
        return None

    def find_one(self, query=None, projection=None):
        q = query or {}
        for doc in self._find_many(q, projection):
            return doc
        return None

    def find(self, query=None, projection=None):
        return Cursor(self._find_many(query or {}, projection))

    def _find_many(self, query, projection=None):
        docs = self._load()
        out = [d for d in docs if _match(d, query)]
        if projection is not None:
            out = [self._project(d, projection) for d in out]
        return out

    @staticmethod
    def _project(doc, projection):
        result = dict(doc)
        for k, v in projection.items():
            if v == 0:
                result.pop(k, None)
        return result

    def count_documents(self, query=None):
        return len([d for d in self._load() if _match(d, query or {})])

    def update_one(self, query, update, upsert=False):
        with self._lock:
            docs = self._load()
            for doc in docs:
                if _match(doc, query):
                    _apply_update(doc, update)
                    self._save(docs)
                    return True
            if upsert:
                new = dict(query)
                self._strip_match_ops(new)
                _apply_update(new, update)
                docs.append(new)
                self._save(docs)
                return True
            return False

    @staticmethod
    def _strip_match_ops(doc):
        for k in list(doc.keys()):
            if isinstance(doc[k], dict) and (_contains_op(doc[k]) or "$regex" in doc[k]):
                doc.pop(k, None)


class LocalStore:
    """Manages named ``FileCollection`` instances under a date-stamped base directory."""
    def __init__(self, base_dir=None):
        self.base_dir = (base_dir or config.local_store_dir)
        # Add date-based subfolder for organizing outputs
        date_folder = getattr(config, 'date_folder', dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"))
        self.base_dir = os.path.join(self.base_dir, date_folder)
        os.makedirs(self.base_dir, exist_ok=True)
        self._collections = {}

    def collection(self, name):
        if name not in self._collections:
            self._collections[name] = FileCollection(os.path.join(self.base_dir, f"{name}.jsonl"))
        return self._collections[name]


_local = LocalStore()


def get_local_store():
    return _local


def local_known_faces():
    return _local.collection("known_faces")


def local_face_events():
    return _local.collection("face_events")


def local_attendance_logs():
    return _local.collection("attendance_logs")


def local_visitor_logs():
    return _local.collection("visitor_logs")


def local_visitor_counts():
    return _local.collection("visitor_counts")