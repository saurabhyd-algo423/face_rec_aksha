import argparse
import datetime as dt
import threading

import cv2
import numpy as np
import uvicorn
from kafka import KafkaConsumer

import config as cfg
from api import app
from models import FACE_INDEX, build_index_from_db
from service import GateProcessor
from store import ensure_indexes, known_faces_collection

INPUT_TOPIC = cfg.config.INPUT_TOPIC
KAFKA_SERVER = cfg.config.kafka_servers()


def parse_args():
    parser = argparse.ArgumentParser(description="Aksha Face Recognition Surveillance Pod")
    parser.add_argument("--fps", type=int, default=cfg.config.PROCESS_FPS, help="Processing FPS")
    parser.add_argument("--old_camera_name", default=None, help="Old camera name (if renamed)")
    parser.add_argument("--update_camera_name", default=None, help="Updated camera name")
    return parser.parse_args()


def kafka_loop(args):
    consumer = KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=[KAFKA_SERVER],
        group_id="face_rec_group",
    )
    processor = GateProcessor()
    print("Kafka consumer created; waiting for frames...", flush=True)
    for msg in consumer:
        try:
            headers = {k: v.decode('utf-8') if v else None for k, v in msg.headers}
            camera_name = headers.get('camera_name')
            frame_id = headers.get('frame_id')
            timestamp_str = headers.get('timestamp_str')
            if not camera_name:
                continue
            if timestamp_str:
                try:
                    ts = dt.datetime.fromisoformat(timestamp_str)
                except ValueError:
                    ts = dt.datetime.now()
            else:
                ts = dt.datetime.now()

            frame_bytes = msg.value
            frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            if camera_name == cfg.config.GATE_CAM:
                processor.process_frame(frame, camera_name, ts)
        except Exception as e:
            print(f"Error processing message: {e}", flush=True)


def main():
    args = parse_args()
    print(f"KAFKA_SERVER: {KAFKA_SERVER}", flush=True)
    print(f"GATE_CAM: {cfg.config.GATE_CAM}", flush=True)

    ensure_indexes()
    try:
        build_index_from_db(known_faces_collection())
        print(f"FAISS index built: {len(FACE_INDEX.face_ids())} face(s)", flush=True)
    except Exception as e:
        print(f"FAISS build warning: {e}", flush=True)

    consumer_thread = threading.Thread(target=kafka_loop, args=(args,), daemon=True)
    consumer_thread.start()
    uvicorn.run(app, host=cfg.config.API_HOST, port=cfg.config.API_PORT)


if __name__ == "__main__":
    main()
