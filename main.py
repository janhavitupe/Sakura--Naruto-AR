"""Chakra-Based Multi-Modal Gesture Interaction System.

Pipeline per frame:
  capture (thread) -> mirror -> [hands, face] inference -> features
  -> per-frame gesture rules -> temporal logic -> jutsu engine + state
  -> AR renderer -> display
Capture, inference and rendering run on separate threads (see InferenceWorker).

Keys: q/Esc quit | d debug overlay | c clear cooldowns | r refill chakra
"""
import os

# Must run before numpy/cv2 are imported. OpenBLAS otherwise reserves a
# buffer per CPU core at import, which fails with "Memory allocation still
# failed" on machines low on free memory. Our numpy work is tiny arrays, so
# one thread costs nothing.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

from config import settings as S
from engine.jutsu_engine import JutsuEngine
from engine.renderer import Renderer
from modules.face_tracker import FaceTracker
from modules.feature_extractor import HandFeatures
from modules.gesture_classifier import GestureClassifier
from modules.hand_tracker import HandTracker
from modules.state_manager import StateManager
from modules.temporal import TemporalLogic
from utils.helpers import FPSCounter, ThreadedCamera


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=S.CAMERA_INDEX)
    ap.add_argument("--demo", action="store_true",
                    help=f"short cooldowns ({S.DEMO_STANDARD_COOLDOWN_S}s / "
                         f"{S.DEMO_BYAKUGOU_COOLDOWN_S}s) for testing")
    ap.add_argument("--debug", action="store_true", help="start with debug overlay on")
    ap.add_argument("--sync", action="store_true",
                    help="run inference on the main thread (no pipelining)")
    return ap.parse_args()


class Pipeline:
    def __init__(self, demo=False):
        self.hands = HandTracker()
        self.face = FaceTracker()
        self.classifier = GestureClassifier()
        self.temporal = TemporalLogic()
        self.state = StateManager(demo=demo)
        self.engine = JutsuEngine(self.state)
        self.renderer = Renderer()
        self.frame_no = 0
        self.last_face = None
        self.last_ms = -1
        self.pool = ThreadPoolExecutor(max_workers=1)

    def infer(self, frame, t):
        """Stage 1: model inference. Hands and face run concurrently (the
        MediaPipe calls release the GIL). Returns (t, hands, face)."""
        # MediaPipe VIDEO mode requires strictly increasing integer ms.
        ms = max(self.last_ms + 1, int(t * 1000))
        self.last_ms = ms
        t = ms / 1000.0

        small = frame
        if S.INFERENCE_SCALE < 1.0:
            small = cv2.resize(frame, None, fx=S.INFERENCE_SCALE, fy=S.INFERENCE_SCALE,
                               interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        run_face = self.frame_no % S.FACE_EVERY_N_FRAMES == 0
        self.frame_no += 1
        face_job = self.pool.submit(self.face.process, rgb, frame.shape, t) if run_face else None
        hands = self.hands.process(rgb, frame.shape, t)
        if face_job is not None:
            self.last_face = face_job.result()
        return t, hands, self.last_face

    def update(self, frame, t, hands, face, fps=0.0, debug=False):
        """Stage 2: features -> gestures -> temporal -> jutsu -> render."""
        feats = {h.label: HandFeatures(h) for h in hands}
        raw = self.classifier.classify(list(feats.values()), face)
        ts = self.temporal.update(raw, face, t)
        active = self.engine.update(ts, feats, face, t)

        self.renderer.render(frame, active, feats, face, self.state, ts, fps, t, debug)
        self.renderer.draw_messages(frame, self.engine.messages)
        return frame

    def step(self, frame, t, fps=0.0, debug=False):
        t, hands, face = self.infer(frame, t)
        return self.update(frame, t, hands, face, fps, debug)

    def close(self):
        self.pool.shutdown()
        self.hands.close()
        self.face.close()


class InferenceWorker:
    """Runs capture + inference on its own thread, so inference for frame N
    overlaps logic/rendering of frame N-1. Frame time becomes
    max(inference, render) instead of their sum. Each result carries its own
    frame, so overlays stay aligned with the image they were computed on."""

    def __init__(self, cam, pipe, t0):
        self.cam, self.pipe, self.t0 = cam, pipe, t0
        self.out = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        fid = -1
        while self.running:
            fid, frame = self.cam.read(fid)
            if frame is None:
                self.out.put(None)
                return
            frame = cv2.flip(frame, 1)
            result = (frame,) + self.pipe.infer(frame, time.perf_counter() - self.t0)
            while self.running:
                try:
                    self.out.put(result, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def get(self):
        return self.out.get()

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)


def main():
    args = parse_args()
    cv2.setNumThreads(S.OPENCV_THREADS)
    cam = ThreadedCamera(args.camera, S.FRAME_WIDTH, S.FRAME_HEIGHT, S.CAMERA_FPS)
    if not cam.running:
        raise SystemExit(f"Could not open camera {args.camera}")
    pipe = Pipeline(demo=args.demo)
    fps = FPSCounter()
    debug = args.debug
    t0 = time.perf_counter()
    worker = None if args.sync else InferenceWorker(cam, pipe, t0)
    fid = -1

    try:
        while True:
            if worker is not None:
                item = worker.get()
                if item is None:
                    print("Camera stopped delivering frames")
                    break
                frame, t, hands, face = item
            else:
                fid, frame = cam.read(fid)
                if frame is None:
                    print("Camera stopped delivering frames")
                    break
                frame = cv2.flip(frame, 1)          # selfie view; handedness assumes this
                t, hands, face = pipe.infer(frame, time.perf_counter() - t0)

            pipe.update(frame, t, hands, face, fps.fps, debug)
            fps.tick()
            cv2.imshow(S.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("d"):
                debug = not debug
            elif key == ord("c"):
                pipe.state.reset_cooldowns()
            elif key == ord("r"):
                pipe.state.refill()
    finally:
        if worker is not None:
            worker.stop()
        cam.release()
        pipe.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
