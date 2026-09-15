#!/usr/bin/env python
"""
Test script for interactive ROI coordinate selection.

Usage:
    python test_roi.py <video_path> [--x1 X1] [--y1 Y1] [--x2 X2] [--y2 Y2]

Displays a frame with ROI rectangle showing what region will be monitored.
Helps you choose the right region for visitor counting.

The script shows:
  - ROI normalized coords (0-1) and pixel coords
  - Percentage of total frame area covered by ROI
  - First detected face overlap with ROI (if any)

After running, ROI params are saved to a JSON file for reference.
"""

import argparse
import cv2
import sys
import os
import json
import numpy as np


def _parse():
    parser = argparse.ArgumentParser(
        prog="test_roi",
        description="Test and visualize ROI coordinates on a video frame.",
    )
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument(
        "--x1", type=float, default=0.05,
        help="ROI top-left X (normalized 0-1, default: 0.05)"
    )
    parser.add_argument(
        "--y1", type=float, default=0.0,
        help="ROI top-left Y (normalized 0-1, default: 0.0)"
    )
    parser.add_argument(
        "--x2", type=float, default=0.95,
        help="ROI bottom-right X (normalized 0-1, default: 0.95)"
    )
    parser.add_argument(
        "--y2", type=float, default=1.0,
        help="ROI bottom-right Y (normalized 0-1, default: 1.0)"
    )
    parser.add_argument(
        "--frame", type=int, default=0,
        help="Frame number to display (default: 0)"
    )
    return parser.parse_args()


def main():
    args = _parse()

    # Open video and seek to frame
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, args.frame))
    ret, frame = cap.read()
    if not ret:
        print(f"ERROR: Cannot read frame {args.frame}")
        sys.exit(1)

    # Calculate pixel coords from normalized values
    px1 = int(args.x1 * w)
    py1 = int(args.y1 * h)
    px2 = int(args.x2 * w)
    py2 = int(args.y2 * h)

    # Draw ROI on frame
    output = frame.copy()
    overlay = output.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (255, 180, 0), -1)
    output = cv2.addWeighted(overlay, 0.35, output, 0.65, 0)
    cv2.rectangle(output, (px1, py1), (px2, py2), (255, 180, 0), 3)

    # Add labels
    label = f"ROI: ({args.x1:.3f},{args.y1:.3f})-({args.x2:.3f},{args.y2:.3f})"
    cv2.putText(output, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)

    # Calculate area coverage
    roi_px_area = (px2 - px1) * (py2 - py1)
    frame_px_area = w * h
    pct_area = roi_px_area / frame_px_area * 100

    # Try to detect a face and show overlap
    face_overlap_pct = 0
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            fx, fy, fw, fh = faces[0]
            fnx1, fny1 = fx / w, fy / h
            fnx2, fny2 = (fx + fw) / w, (fy + fh) / h

            overlap_x1 = max(fnx1, args.x1)
            overlap_y1 = max(fny1, args.y1)
            overlap_x2 = min(fnx2, args.x2)
            overlap_y2 = min(fny2, args.y2)
            ow = max(0, overlap_x2 - overlap_x1)
            oh = max(0, overlap_y2 - overlap_y1)
            oa = ow * oh
            fa = fw * fh
            face_overlap_pct = oa / fa if fa > 0 else 0

            cv2.rectangle(output, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
            cv2.putText(output, f"Face overlap: {face_overlap_pct:.1%}", (fx, fy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    except Exception:
        pass

    # Display info
    info = f"Frame {args.frame} | {w}x{h} | ROI pixels: ({px1},{py1})-({px2},{py2})"
    cv2.putText(output, info, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Print analysis
    print(f"""
=== ROI Analysis ===
Video: {os.path.basename(args.video)}
Frame: {args.frame} | Resolution: {w}x{h}
ROI Normalized: ({args.x1:.3f},{args.y1:.3f})-({args.x2:.3f},{args.y2:.3f})
ROI Pixels: ({px1},{py1})-({px2},{py2}) | Size: {(px2-px1)}x{(py2-py1)}
ROI Area: {roi_px_area:,} px² | Frame Area: {frame_px_area:,} px²
ROI covers {pct_area:.1f}% of total frame
Face overlap: {face_overlap_pct:.1%} of detected face

Recommended for gate camera:
  - Narrow ROI (30-50%): Best for Re-ID deduplication
  - Medium ROI (50-70%): Good balance of coverage and dedup
  - Wide ROI (70%+): Catches more entries but may include edge faces

Saved params to: {os.path.dirname(args.video)}/_roi_params.json

Press any key to close window (or 'q' to quit)...

""")

    # Show window
    try:
        cv2.imshow("ROI Tester", output)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("Quitting without saving.")
        cv2.destroyAllWindows()
    except cv2.error:
        print("GUI not available, params already saved.")


if __name__ == "__main__":
    main()