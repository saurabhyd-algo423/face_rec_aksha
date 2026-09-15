import argparse
import datetime as dt
import os
import sys


def _enabled_env():
    SysPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, SysPath)


def _parse():
    parser = argparse.ArgumentParser(
        prog="cli_test",
        description="Offline face-recognition test harness (no Kafka/Mongo/RTSP).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_enroll = sub.add_parser("enroll", help="Enroll a face from a video clip.")
    p_enroll.add_argument("video", help="Path to the enrollment video clip")
    p_enroll.add_argument("--staff_id", required=True)
    p_enroll.add_argument("--name", required=True)
    p_enroll.add_argument("--role", default="staff")
    p_enroll.add_argument("--category", default="staff")
    p_enroll.add_argument("--sharpness", type=float, default=25.0, help="SHARPNESS_MIN for quality gate")

    p_enroll_img = sub.add_parser("enroll-images", help="Enroll a face from a directory of images.")
    p_enroll_img.add_argument("image_dir", help="Path to directory containing face images")
    p_enroll_img.add_argument("--staff_id", required=True)
    p_enroll_img.add_argument("--name", required=True)
    p_enroll_img.add_argument("--role", default="staff")
    p_enroll_img.add_argument("--category", default="staff")

    p_enroll_rtsp = sub.add_parser("enroll-rtsp", help="Enroll a face from live RTSP webcam.")
    p_enroll_rtsp.add_argument("--rtsp-url", required=True, help="RTSP stream URL (rtsp://user:pass@ip:port/stream)")
    p_enroll_rtsp.add_argument("--staff-id", required=True)
    p_enroll_rtsp.add_argument("--name", required=True)
    p_enroll_rtsp.add_argument("--role", default="staff")
    p_enroll_rtsp.add_argument("--category", default="staff")
    p_enroll_rtsp.add_argument("--max-duration", type=int, default=10, help="Max capture duration in seconds")
    p_enroll_rtsp.add_argument("--sample-fps", type=float, default=2.0, help="Frames per second to sample")
    p_enroll_rtsp.add_argument("--min-frames", type=int, default=3, help="Min good frames needed")

    p_list = sub.add_parser("list", help="List enrolled faces")

    p_run = sub.add_parser("run", help="Run recognition on sample footage.")
    p_run.add_argument("footage", help="Path to the sample CCTV footage video")
    p_run.add_argument("--camera", default=None)
    p_run.add_argument("--thresh", type=float, default=0.7, help="SIMILARITY_THRESHOLD")
    p_run.add_argument("--cooldown", type=float, default=0, help="REMATCH_COOLDOWN_SEC")
    p_run.add_argument("--max_frames", type=int, default=0, help="Limit frames to process (0 = all)")
    p_run.add_argument("--out", default=None, help="Output dir for annotated video + alerts")
    p_run.add_argument("--sharpness", type=float, default=10.0, help="SHARPNESS_MIN for quality gate")
    p_run.add_argument("--roi", action="store_true", help="Enable ROI-based visitor counting")
    p_run.add_argument("--roi-x1", type=float, default=0.0, help="ROI top-left X (0-1, normalized)")
    p_run.add_argument("--roi-y1", type=float, default=0.0, help="ROI top-left Y (0-1, normalized)")
    p_run.add_argument("--roi-x2", type=float, default=1.0, help="ROI bottom-right X (0-1, normalized)")
    p_run.add_argument("--roi-y2", type=float, default=1.0, help="ROI bottom-right Y (0-1, normalized)")
    p_run.add_argument("--roi-iou", type=float, default=0.8, help="ROI overlap threshold (0-1)")
    p_run.add_argument("--reid-thresh", type=float, default=0.35, help="VISITOR_REID_THRESHOLD")

    p_recognize_rtsp = sub.add_parser("recognize-rtsp", help="Live face recognition from RTSP stream.")
    p_recognize_rtsp.add_argument("--rtsp-url", required=True, help="RTSP stream URL")
    p_recognize_rtsp.add_argument("--camera", default=None)
    p_recognize_rtsp.add_argument("--thresh", type=float, default=0.7, help="SIMILARITY_THRESHOLD")
    p_recognize_rtsp.add_argument("--cooldown", type=float, default=2.5, help="REMATCH_COOLDOWN_SEC")
    p_recognize_rtsp.add_argument("--sharpness", type=float, default=10.0, help="SHARPNESS_MIN for quality gate")
    p_recognize_rtsp.add_argument("--roi", action="store_true", help="Enable ROI-based visitor counting")
    p_recognize_rtsp.add_argument("--roi-x1", type=float, default=0.0)
    p_recognize_rtsp.add_argument("--roi-y1", type=float, default=0.0)
    p_recognize_rtsp.add_argument("--roi-x2", type=float, default=1.0)
    p_recognize_rtsp.add_argument("--roi-y2", type=float, default=1.0)
    p_recognize_rtsp.add_argument("--roi-iou", type=float, default=0.8)
    p_recognize_rtsp.add_argument("--reid-thresh", type=float, default=0.35)

    p_reset = sub.add_parser("reset", help="Wipe local store + output dirs.")

    return parser.parse_args()


def _setup_local(out_dir=None):
    if not out_dir:
        out_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "data", "offline")
    os.environ["STORE_BACKEND"] = "local"
    os.environ["OUTPUT_SINK"] = "local"
    os.environ["PUBLISH_MONITOR"] = "false"
    os.environ["LOCAL_STORE_DIR"] = os.path.join(out_dir, "store")
    os.environ["LOCAL_OUT_DIR"] = os.path.join(out_dir, "out")
    os.environ["MEDIA_ROOT"] = os.path.join(out_dir, "media")
    os.environ["AKSHA_PATH"] = os.path.join(out_dir, "images")
    return out_dir


def main():
    args = _parse()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "data", "offline")
    out_dir = getattr(args, "out", None) or base_dir
    _setup_local(out_dir)

    if getattr(args, "thresh", None) is not None:
        os.environ["SIMILARITY_THRESHOLD"] = str(args.thresh)
    if getattr(args, "cooldown", None) is not None:
        os.environ["REMATCH_COOLDOWN_SEC"] = str(args.cooldown)
    if getattr(args, "sharpness", None) is not None:
        os.environ["SHARPNESS_MIN"] = str(args.sharpness)
    if getattr(args, "roi", False):
        os.environ["ROI_ENABLED"] = "true"
    os.environ["ROI_X1"] = str(getattr(args, "roi_x1", 0.0))
    os.environ["ROI_Y1"] = str(getattr(args, "roi_y1", 0.0))
    os.environ["ROI_X2"] = str(getattr(args, "roi_x2", 1.0))
    os.environ["ROI_Y2"] = str(getattr(args, "roi_y2", 1.0))
    os.environ["ROI_IOU_THRESHOLD"] = str(getattr(args, "roi_iou", 0.8))
    os.environ["VISITOR_REID_THRESHOLD"] = str(getattr(args, "reid_thresh", 0.35))

    if args.cmd == "reset":
        import shutil
        for d in (os.environ["LOCAL_STORE_DIR"], os.environ["LOCAL_OUT_DIR"], os.environ["MEDIA_ROOT"], os.environ["AKSHA_PATH"]):
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
        print(f"Reset local dirs under {out_dir}")
        return

    from models import FACE_INDEX, build_index_from_db
    from store import known_faces_collection

    if args.cmd in ("enroll",):
        return _do_enroll(args, known_faces_collection)

    if args.cmd == "enroll-images":
        return _do_enroll_images(args, known_faces_collection)

    if args.cmd == "enroll-rtsp":
        return _do_enroll_rtsp(args, known_faces_collection)

    if args.cmd == "list":
        return _do_list(known_faces_collection)

    if args.cmd == "run":
        return _do_run(args, known_faces_collection, FACE_INDEX, build_index_from_db)

    if args.cmd == "recognize-rtsp":
        return _do_recognize_rtsp(args, known_faces_collection, FACE_INDEX, build_index_from_db)

    parser_exit(f"Unknown command: {args.cmd}")


def parser_exit(msg):
    print(msg)
    sys.exit(2)


def _do_enroll(args, known_faces_collection):
    import cv2
    from service import EnrollmentError, enroll_from_video

    with open(args.video, "rb") as f:
        video_data = f.read()
    try:
        face_id = enroll_from_video(
            video_data, args.staff_id, args.name, args.role,
            site_id="offline", category=args.category,
        )
        print(f"ENROLLED face_id={face_id} staff_id={args.staff_id} name={args.name} (via video)")
        return face_id
    except EnrollmentError as e:
        print(f"ENROLL_ERROR {e.code}: {e}")
        sys.exit(1)


def _do_enroll_images(args, known_faces_collection):
    from service import EnrollmentError, enroll_from_images

    image_dir = args.image_dir
    if not os.path.isdir(image_dir):
        print(f"Not a directory: {image_dir}")
        sys.exit(1)

    image_data_list = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            fpath = os.path.join(image_dir, fname)
            with open(fpath, "rb") as f:
                image_data_list.append(f.read())

    if not image_data_list:
        print(f"No images found in {image_dir}")
        sys.exit(1)

    print(f"Found {len(image_data_list)} images, enrolling {args.name}...")
    try:
        face_id = enroll_from_images(
            image_data_list, args.staff_id, args.name, args.role,
            site_id="offline", category=args.category,
        )
        print(f"ENROLLED face_id={face_id} staff_id={args.staff_id} name={args.name} (via {len(image_data_list)} images)")
        return face_id
    except EnrollmentError as e:
        print(f"ENROLL_ERROR {e.code}: {e}")
        sys.exit(1)


def _do_enroll_rtsp(args, known_faces_collection):
    import cv2
    import time
    from service import EnrollmentError, enroll_from_rtsp

    print(f"Connecting to RTSP: {args.rtsp_url}")
    print(f"Enrolling: {args.name} (staff_id={args.staff_id})")
    print(f"Max duration: {args.max_duration}s | Sample FPS: {args.sample_fps} | Min frames: {args.min_frames}")
    print("Press 'q' to abort early\n")

    def on_frame(frame, bbox, frame_ok, collected, needed):
        display = frame.copy()
        if bbox is not None and frame_ok:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            status_text = f"GOOD [{collected}/{needed}]"
            color = (0, 255, 0)
        elif bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
            status_text = f"LOW QUALITY [{collected}/{needed}]"
            color = (0, 0, 255)
        else:
            status_text = f"NO FACE [{collected}/{needed}]"
            color = (0, 0, 255)

        cv2.putText(display, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display, f"Name: {args.name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("RTSP Enrollment - Press 'q' to abort", display)

    try:
        face_id = enroll_from_rtsp(
            args.rtsp_url, args.staff_id, args.name, args.role,
            site_id="offline", category=args.category,
            max_duration=args.max_duration, sample_fps=args.sample_fps,
            min_good_frames=args.min_frames, on_frame=on_frame,
        )
        cv2.destroyAllWindows()
        print(f"\nENROLLED face_id={face_id} staff_id={args.staff_id} name={args.name} (via RTSP)")
        return face_id
    except EnrollmentError as e:
        cv2.destroyAllWindows()
        print(f"\nENROLL_ERROR {e.code}: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\nAborted by user.")
        sys.exit(1)


def _do_list(known_faces_collection):
    docs = list(known_faces_collection().find({"status": "active"}))
    if not docs:
        print("No enrolled faces.")
        return
    for d in docs:
        print(f"  face_id={d.get('face_id')} staff_id={d.get('staff_id')} "
              f"name={d.get('name')} role={d.get('role')} cat={d.get('category')} "
              f"via={d.get('enrolled_via')}")
    print(f"Total: {len(docs)} face(s)")


def _do_run(args, known_faces_collection, FACE_INDEX, build_index_from_db):
    import cv2
    import numpy as np
    from service import GateProcessor
    from config import config

    build_index_from_db(known_faces_collection())
    print(f"FAISS index: {len(FACE_INDEX.face_ids())} known face(s), "
          f"SIMILARITY_THRESHOLD={config.SIMILARITY_THRESHOLD}, cooldown={config.REMATCH_COOLDOWN_SEC}s")

    cam = args.camera or config.GATE_CAM
    cap = cv2.VideoCapture(args.footage)
    if not cap.isOpened():
        print(f"Cannot open footage: {args.footage}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    base_name = os.path.basename(args.footage).split('.')[0]

    # --- ROI visualization frame ---
    os.makedirs(config.local_out_dir, exist_ok=True)
    roi_frame_path = os.path.join(config.local_out_dir, f"{base_name}_roi_overlay.png")
    ret_sample, sample_frame = cap.read()
    if ret_sample:
        roi_img = sample_frame.copy()
        if config.ROI_ENABLED:
            rx1 = int(config.ROI_X1 * w)
            ry1 = int(config.ROI_Y1 * h)
            rx2 = int(config.ROI_X2 * w)
            ry2 = int(config.ROI_Y2 * h)
            overlay = roi_img.copy()
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (255, 180, 0), -1)
            roi_img = cv2.addWeighted(overlay, 0.25, roi_img, 0.75, 0)
            cv2.rectangle(roi_img, (rx1, ry1), (rx2, ry2), (255, 180, 0), 3)
            cv2.putText(roi_img, "ROI", (rx1 + 5, ry1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)
            roi_info = f"ROI: ({config.ROI_X1},{config.ROI_Y1})-({config.ROI_X2},{config.ROI_Y2}) iou={config.ROI_IOU_THRESHOLD}"
            cv2.putText(roi_img, roi_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
        else:
            cv2.putText(roi_img, "ROI disabled", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(roi_frame_path, roi_img)
        print(f"ROI overlay frame saved: {roi_frame_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- SSIM setup ---
    try:
        from skimage.metrics import structural_similarity as ssim_func
        has_ssim = True
    except ImportError:
        has_ssim = False
    prev_gray = None
    ssim_values = []

    # Create visitor_frame subfolder before the loop
    visitor_frame_dir = os.path.join(config.local_out_dir, "visitor_frame")
    os.makedirs(visitor_frame_dir, exist_ok=True)

    processor = GateProcessor()
    processor.load_daily_state()
    base_ts = dt.datetime.now(dt.timezone.utc)
    matched = unknown = 0
    frame_idx = 0
    total = 0
    video_frames = []
    saved_visitor_tracks = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = base_ts + dt.timedelta(seconds=frame_idx / fps)
        events = processor.process_frame(frame, cam, ts)

        # --- SSIM calculation ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and has_ssim:
            try:
                ssim_val = ssim_func(prev_gray, gray, full=False)
                ssim_values.append((frame_idx, ssim_val))
            except Exception:
                pass
        prev_gray = gray

        # --- Draw annotations + per-face saving ---
        annotated = frame.copy()
        if config.ROI_ENABLED:
            rx1 = int(config.ROI_X1 * w)
            ry1 = int(config.ROI_Y1 * h)
            rx2 = int(config.ROI_X2 * w)
            ry2 = int(config.ROI_Y2 * h)
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 180, 0), 2)

        for e in events:
            status = e.get("match_status", "unknown")
            x1, y1, x2, y2 = e.get("bbox", [0, 0, 0, 0])
            label = e.get("name") or "Unknown"
            color = (0, 255, 0) if status == "matched" else (0, 0, 255)
            sim = e.get("similarity")
            display = f"{label} ({sim:.2f})" if sim else label

            # Draw on annotated frame
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, display, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Count
            if status == "matched":
                matched += 1
            else:
                unknown += 1
            total += 1

            # Save per-face image
            track_id = e.get("track_id", 0)
            out_name = f"frame_{frame_idx:06d}_track{track_id:03d}_{status}.jpg"
            out_path = os.path.join(config.local_out_dir, out_name)
            cv2.imwrite(out_path, annotated)

            # Save visitor frames to subfolder (first frame per unique track only)
            if status == "unknown" and track_id not in saved_visitor_tracks:
                saved_visitor_tracks.add(track_id)
                visitor_name = f"visitor_{frame_idx:06d}_track{track_id:03d}.jpg"
                cv2.imwrite(os.path.join(visitor_frame_dir, visitor_name), annotated)

        hud = f"Frame {frame_idx} | Staff: {processor.staff_count} | Visitors: {processor.visitor_count}"
        cv2.putText(annotated, hud, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        video_frames.append(annotated)

        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break

    cap.release()

    # --- Write annotated video using imageio ---
    out_video = None
    if video_frames:
        out_video = os.path.join(config.local_out_dir, f"{base_name}_annotated.mp4")
        try:
            import imageio
            writer = imageio.get_writer(out_video, fps=fps, codec="libx264", macro_block_size=1)
            for vf in video_frames:
                writer.append_data(cv2.cvtColor(vf, cv2.COLOR_BGR2RGB))
            writer.close()
            print(f"Annotated video saved: {out_video}")
        except Exception as e:
            print(f"WARN: imageio write failed: {e}")
            out_video = None

    processor.persist_daily_counts()

    # --- Save SSIM summary ---
    ssim_path = None
    if ssim_values:
        import json as json_mod
        ssim_vals_only = [s[1] for s in ssim_values]
        ssim_summary = {
            "total_frames": len(ssim_values),
            "mean_ssim": round(float(np.mean(ssim_vals_only)), 4),
            "min_ssim": round(float(np.min(ssim_vals_only)), 4),
            "max_ssim": round(float(np.max(ssim_vals_only)), 4),
            "std_ssim": round(float(np.std(ssim_vals_only)), 4),
            "low_ssim_frames": [{"frame": int(f), "ssim": round(float(s), 4)} for f, s in ssim_values if s < 0.9],
            "per_frame": [{"frame": int(f), "ssim": round(float(s), 4)} for f, s in ssim_values],
        }
        ssim_path = os.path.join(config.local_out_dir, f"{base_name}_ssim_summary.json")
        with open(ssim_path, "w") as f:
            json_mod.dump(ssim_summary, f, indent=2)
        print(f"SSIM: mean={ssim_summary['mean_ssim']} min={ssim_summary['min_ssim']} max={ssim_summary['max_ssim']}")

    print(f"Processed {frame_idx} frames on camera '{cam}'")
    print(f"Face events: {total} total | {matched} matched | {unknown} unknown")
    print(f"Unique staff: {processor.staff_count}")
    print(f"Unique visitors: {processor.visitor_count}")
    if config.ROI_ENABLED:
        print(f"ROI: ({config.ROI_X1},{config.ROI_Y1})-({config.ROI_X2},{config.ROI_Y2}) iou_thresh={config.ROI_IOU_THRESHOLD} reid_thresh={config.VISITOR_REID_THRESHOLD}")
    print(f"ROI overlay frame: {roi_frame_path}")
    print(f"Annotated video: {out_video}")
    print(f"SSIM summary: {ssim_path}")
    print(f"Local output: {config.local_out_dir}")


def _do_recognize_rtsp(args, known_faces_collection, FACE_INDEX, build_index_from_db):
    import cv2
    import time
    import datetime as dt
    from service import GateProcessor
    from config import config

    build_index_from_db(known_faces_collection())
    print(f"FAISS index: {len(FACE_INDEX.face_ids())} known face(s), SIMILARITY_THRESHOLD={config.SIMILARITY_THRESHOLD}")

    cam = args.camera or config.GATE_CAM
    process_fps = config.PROCESS_FPS
    frame_interval = 1.0 / process_fps

    print(f"Connecting to RTSP: {args.rtsp_url}")
    cap = cv2.VideoCapture(args.rtsp_url)
    if not cap.isOpened():
        print(f"ERROR: Cannot connect to RTSP stream: {args.rtsp_url}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Stream: {w}x{h} @ {fps:.1f} FPS | Processing: {process_fps} FPS")

    processor = GateProcessor()
    processor.load_daily_state()
    base_ts = dt.datetime.now(dt.timezone.utc)
    matched = unknown = 0
    frame_idx = 0
    total = 0

    print("Live recognition started. Press 'q' or 'Esc' to stop.\n")

    try:
        while True:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Stream disconnected.")
                break

            ts = base_ts + dt.timedelta(seconds=frame_idx / fps)
            events = processor.process_frame(frame, cam, ts)

            # Draw annotations
            annotated = frame.copy()
            if config.ROI_ENABLED:
                rx1 = int(config.ROI_X1 * w)
                ry1 = int(config.ROI_Y1 * h)
                rx2 = int(config.ROI_X2 * w)
                ry2 = int(config.ROI_Y2 * h)
                cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 180, 0), 2)

            for e in events:
                status = e.get("match_status", "unknown")
                x1, y1, x2, y2 = e.get("bbox", [0, 0, 0, 0])
                label = e.get("name") or "Unknown"
                color = (0, 255, 0) if status == "matched" else (0, 0, 255)
                sim = e.get("similarity")
                display = f"{label} ({sim:.2f})" if sim else label
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, display, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if status == "matched":
                    matched += 1
                else:
                    unknown += 1
                total += 1

            hud = f"Frame {frame_idx} | Staff: {processor.staff_count} | Visitors: {processor.visitor_count} | Matched: {matched} | Unknown: {unknown}"
            cv2.putText(annotated, hud, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow("Live Recognition - Press 'q' to quit", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\nStopped by user.")
                break

            frame_idx += 1
            elapsed = time.time() - t_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        processor.persist_daily_counts()

    print(f"\n--- Summary ---")
    print(f"Processed {frame_idx} frames on camera '{cam}'")
    print(f"Face events: {total} total | {matched} matched | {unknown} unknown")
    print(f"Unique staff: {processor.staff_count}")
    print(f"Unique visitors: {processor.visitor_count}")
    print(f"Local output: {config.local_out_dir}")


if __name__ == "__main__":
    main()