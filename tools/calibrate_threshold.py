import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ANALYZER, FACE_INDEX, detect_faces, embed_face  # noqa: E402
from config import config  # noqa: E402


def embed_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    faces = detect_faces(img)
    if not faces:
        return None
    largest = max(faces, key=lambda f: abs(f.bbox[2] - f.bbox[0]) * abs(f.bbox[3] - f.bbox[1]))
    return embed_face(img, largest)


def collect_embeddings(root):
    out = []
    if not os.path.isdir(root):
        return out
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        for f in os.listdir(p):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            emb = embed_image(os.path.join(p, f))
            if emb is not None:
                out.append((name, emb))
    return out


def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/calibrate_threshold.py <known_dir> <unknown_dir>")
        print("  known_dir:  folders of staff faces")
        print("  unknown_dir: folders of customer/unknown faces")
        sys.exit(1)

    known = collect_embeddings(sys.argv[1])
    unknown = collect_embeddings(sys.argv[2])
    print(f"Known samples: {len(known)}, unknown samples: {len(unknown)}")

    if not known or not unknown:
        print("Need at least one known and one unknown sample")
        sys.exit(1)

    from models import FACE_INDEX
    from store import known_faces_collection

    docs = list(known_faces_collection().find({"status": "active"}))
    FACE_INDEX.rebuild(docs)
    if FACE_INDEX.size == 0:
        print("FAISS index is empty; enroll staff first")
        sys.exit(1)

    gen_scores, imp_scores = [], []
    for _, emb in known:
        m = FACE_INDEX.search(emb)
        gen_scores.append(m[1] if m else 0.0)
    for _, emb in unknown:
        m = FACE_INDEX.search(emb)
        imp_scores.append(m[1] if m else 0.0)

    gen = np.array(gen_scores)
    imp = np.array(imp_scores)
    print(f"Known  (genuine) similarity: mean={gen.mean():.3f} min={gen.min():.3f} max={gen.max():.3f}")
    print(f"Unknown (impostor) similarity: mean={imp.mean():.3f} min={imp.min():.3f} max={imp.max():.3f}")

    for thresh in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        far = float((imp >= thresh).mean())
        frr = float((gen < thresh).mean())
        print(f"threshold={thresh:.2f}  FAR={far:.3f}  FRR={frr:.3f}")

    best = None
    for thresh in np.arange(0.20, 0.60, 0.01):
        far = (imp >= thresh).mean()
        frr = (gen < thresh).mean()
        score = far + frr
        if best is None or score < best[1]:
            best = (round(float(thresh), 2), float(score))
    print(f"Suggested threshold (min FAR+FRR): {best[0]}")


if __name__ == "__main__":
    main()
