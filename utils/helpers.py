"""Shared utilities: threaded capture, FPS counter, ROI blending."""
import threading
import time
from collections import deque

import cv2
import numpy as np


class ThreadedCamera:
    """Reads frames on a background thread and keeps only the newest one.

    cv2.VideoCapture.read() blocks for the camera's frame interval; doing it
    on the main thread serializes capture and inference. With this, the main
    loop always processes the freshest frame and stale frames are dropped
    instead of queuing up (which is what causes 'lag')."""

    def __init__(self, index, width, height, fps):
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.frame_id = 0
        self.lock = threading.Lock()
        self.running = self.cap.isOpened()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = frame
                self.frame_id += 1

    def read(self, last_id=-1, timeout=1.0):
        """Return (frame_id, frame) for a frame newer than `last_id`."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self.lock:
                if self.frame is not None and self.frame_id != last_id:
                    return self.frame_id, self.frame
            time.sleep(0.001)
        return last_id, None

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


class FPSCounter:
    def __init__(self, window=30):
        self.times = deque(maxlen=window)

    def tick(self):
        self.times.append(time.perf_counter())

    @property
    def fps(self):
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / (self.times[-1] - self.times[0])


def clip_roi(shape, x0, y0, x1, y1):
    h, w = shape[:2]
    return max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))


def add_layer(frame, layer, x0, y0):
    """Saturating additive blend of `layer` onto frame at (x0, y0), clipped to
    the frame. Additive blending is what makes energy effects look emissive."""
    h, w = layer.shape[:2]
    fx0, fy0, fx1, fy1 = clip_roi(frame.shape, x0, y0, x0 + w, y0 + h)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    lx0, ly0 = fx0 - int(x0), fy0 - int(y0)
    roi = frame[fy0:fy1, fx0:fx1]
    cv2.add(roi, layer[ly0:ly0 + (fy1 - fy0), lx0:lx0 + (fx1 - fx0)], dst=roi)


def alpha_layer(frame, layer, alpha, x0, y0):
    """Alpha blend `layer` (HxWx3 uint8) using `alpha` (HxW float 0..1)."""
    h, w = layer.shape[:2]
    fx0, fy0, fx1, fy1 = clip_roi(frame.shape, x0, y0, x0 + w, y0 + h)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    lx0, ly0 = fx0 - int(x0), fy0 - int(y0)
    sl = (slice(ly0, ly0 + fy1 - fy0), slice(lx0, lx0 + fx1 - fx0))
    a = alpha[sl][..., None]
    roi = frame[fy0:fy1, fx0:fx1]
    roi[:] = (roi * (1.0 - a) + layer[sl] * a).astype(np.uint8)


def soft_blur(layer, sigma):
    """Gaussian blur; large blurs run on a downscaled copy. A blur is a
    low-pass filter anyway, so this looks identical and is ~f^2 cheaper."""
    h, w = layer.shape[:2]
    f = int(min(8, sigma // 2.5))
    if f >= 2 and min(h, w) // f >= 4:
        small = cv2.resize(layer, (w // f, h // f), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma / f)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(layer, (0, 0), max(sigma, 0.5))


def glow(layer, sigma, strength=1.0):
    """Layer + blurred copy of itself = bloom."""
    return cv2.addWeighted(layer, 1.0, soft_blur(layer, sigma), strength, 0)


def scale_color(color, s):
    return tuple(int(min(255, c * s)) for c in color)


def rotate(v, deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])
