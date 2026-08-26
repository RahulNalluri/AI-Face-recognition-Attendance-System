"""Small, testable webcam connection and backend fallback helpers."""

from __future__ import annotations

import os
import time

import cv2


def connect_camera(
    source: int,
    attempts: int,
    delay_seconds: float,
    capture_factory=None,
    sleeper=time.sleep,
):
    """Open a webcam and require a real frame before declaring it ready."""
    factory = capture_factory or cv2.VideoCapture
    last_capture = None
    backends = [None]
    if capture_factory is None and os.name == "nt":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
    for attempt in range(1, attempts + 1):
        backend = backends[(attempt - 1) % len(backends)]
        capture = factory(source) if backend is None else factory(source, backend)
        last_capture = capture
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            success, frame = capture.read()
            if success and frame is not None:
                return capture, frame, attempt
        capture.release()
        if attempt < attempts:
            sleeper(delay_seconds)
    if last_capture is not None:
        last_capture.release()
    raise RuntimeError(
        f"Camera index {source} did not provide a frame after {attempts} attempts. "
        "Close Windows Camera, Teams, Zoom, or browser camera tabs and check "
        "Settings > Privacy & security > Camera."
    )
