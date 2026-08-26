"""Run real-time multi-face recognition from a webcam or video source.

The program detects and aligns faces with YuNet, extracts SFace embeddings,
applies the calibrated unknown threshold, and confirms identities across
multiple processed frames. Confirmed recognition events are written locally;
camera frames are never saved.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from camera_reliability import connect_camera
from liveness import (
    FaceLandmarkAnalyzer,
    LivenessController,
    LivenessSignals,
    LivenessState,
)


ROOT = Path(__file__).resolve().parent.parent
YUNET_MODEL = ROOT / "models" / "pretrained" / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = ROOT / "models" / "pretrained" / "face_recognition_sface_2021dec.onnx"
CLASSIFIER = ROOT / "models" / "sface" / "classifier.npz"
LABELS = ROOT / "models" / "sface" / "labels.json"
CONFIG = ROOT / "models" / "sface" / "classifier_config.json"
THRESHOLD = ROOT / "models" / "sface" / "unknown_threshold.json"
DEFAULT_EVENT_LOG = ROOT / "artifacts" / "realtime" / "recognition_events.jsonl"
DEFAULT_ATTENDANCE_API = "http://127.0.0.1:5000/api/recognition-events"
DEFAULT_DEVICE_TOKEN = ROOT / "instance" / "device_token.txt"


@dataclass
class Observation:
    box: tuple[int, int, int, int]
    label: str
    similarity: float
    detection_confidence: float


@dataclass
class FaceTrack:
    track_id: int
    box: tuple[int, int, int, int]
    last_processed_frame: int
    history_size: int
    history: deque[tuple[str, float]] = field(init=False)
    display_label: str = "Checking"
    display_similarity: float = 0.0
    detection_confidence: float = 0.0
    liveness_text: str = "Confirming identity..."
    liveness_passed: bool = False
    attendance_text: str = ""
    last_attendance_sync: float = float("-inf")

    def __post_init__(self) -> None:
        self.history = deque(maxlen=self.history_size)

    def observe(self, observation: Observation, frame_number: int) -> None:
        self.box = observation.box
        self.last_processed_frame = frame_number
        self.detection_confidence = observation.detection_confidence
        self.history.append((observation.label, observation.similarity))

    def confirm(self, required: int, minimum_ratio: float) -> tuple[str, float]:
        if not self.history:
            return "Checking", 0.0
        counts = Counter(label for label, _ in self.history)
        label, count = counts.most_common(1)[0]
        if count < required or count / len(self.history) < minimum_ratio:
            return "Checking", float(np.mean([score for _, score in self.history]))
        scores = [score for observed_label, score in self.history if observed_label == label]
        return label, float(np.mean(scores))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="auto", help="auto, a camera index, or video/image path.")
    parser.add_argument("--process-every", type=int, default=2)
    parser.add_argument("--confirmation-frames", type=int, default=3)
    parser.add_argument("--history-size", type=int, default=5)
    parser.add_argument("--confirmation-ratio", type=float, default=0.60)
    parser.add_argument("--cooldown-seconds", type=float, default=10.0)
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--max-frames", type=int, default=0, help="0 runs until stopped.")
    parser.add_argument("--min-similarity", type=float, default=None)
    parser.add_argument("--liveness-timeout", type=float, default=12.0)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-event-log", action="store_true")
    parser.add_argument("--event-log", type=Path, default=DEFAULT_EVENT_LOG)
    parser.add_argument("--attendance-api", default=DEFAULT_ATTENDANCE_API)
    parser.add_argument("--device-token-file", type=Path, default=DEFAULT_DEVICE_TOKEN)
    parser.add_argument("--attendance-sync-seconds", type=float, default=30.0)
    parser.add_argument("--ui-frame-fps", type=float, default=10.0)
    parser.add_argument("--camera-read-failures", type=int, default=5)
    parser.add_argument("--camera-reconnect-attempts", type=int, default=5)
    parser.add_argument("--camera-reconnect-delay", type=float, default=1.0)
    parser.add_argument("--no-attendance-api", action="store_true")
    parser.add_argument("--no-ui-frame", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.process_every < 1:
        raise ValueError("--process-every must be at least 1")
    if args.confirmation_frames < 1:
        raise ValueError("--confirmation-frames must be at least 1")
    if args.history_size < args.confirmation_frames:
        raise ValueError("--history-size must be at least --confirmation-frames")
    if not 0 < args.confirmation_ratio <= 1:
        raise ValueError("--confirmation-ratio must be in (0, 1]")
    if args.cooldown_seconds < 0:
        raise ValueError("--cooldown-seconds cannot be negative")
    if args.max_width < 320:
        raise ValueError("--max-width must be at least 320")
    if args.max_frames < 0:
        raise ValueError("--max-frames cannot be negative")
    if args.min_similarity is not None and not -1 <= args.min_similarity <= 1:
        raise ValueError("--min-similarity must be between -1 and 1")
    if args.liveness_timeout < 5:
        raise ValueError("--liveness-timeout must be at least 5 seconds")
    if args.attendance_sync_seconds < 5:
        raise ValueError("--attendance-sync-seconds must be at least 5 seconds")
    if not 0.5 <= args.ui_frame_fps <= 15:
        raise ValueError("--ui-frame-fps must be between 0.5 and 15")
    if args.camera_read_failures < 1:
        raise ValueError("--camera-read-failures must be at least 1")
    if args.camera_reconnect_attempts < 1:
        raise ValueError("--camera-reconnect-attempts must be at least 1")
    if not 0 <= args.camera_reconnect_delay <= 30:
        raise ValueError("--camera-reconnect-delay must be between 0 and 30 seconds")


def intersection_over_union(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / max(union, 1)


class TrackManager:
    def __init__(self, history_size: int, max_missing_processed_frames: int = 5) -> None:
        self.history_size = history_size
        self.max_missing = max_missing_processed_frames
        self.next_id = 1
        self.tracks: dict[int, FaceTrack] = {}

    def update(self, observations: list[Observation], frame_number: int) -> list[FaceTrack]:
        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if frame_number - track.last_processed_frame <= self.max_missing
        }
        pairs = []
        for track_id, track in self.tracks.items():
            for observation_index, observation in enumerate(observations):
                overlap = intersection_over_union(track.box, observation.box)
                if overlap >= 0.20:
                    pairs.append((overlap, track_id, observation_index))
        matched_tracks: set[int] = set()
        matched_observations: set[int] = set()
        for _, track_id, observation_index in sorted(pairs, reverse=True):
            if track_id in matched_tracks or observation_index in matched_observations:
                continue
            self.tracks[track_id].observe(observations[observation_index], frame_number)
            matched_tracks.add(track_id)
            matched_observations.add(observation_index)
        for index, observation in enumerate(observations):
            if index in matched_observations:
                continue
            track = FaceTrack(
                track_id=self.next_id,
                box=observation.box,
                last_processed_frame=frame_number,
                history_size=self.history_size,
            )
            track.observe(observation, frame_number)
            self.tracks[self.next_id] = track
            self.next_id += 1
        return list(self.tracks.values())


class SFaceRuntime:
    def __init__(self, threshold_override: float | None = None) -> None:
        for required in (YUNET_MODEL, SFACE_MODEL, CLASSIFIER, LABELS, CONFIG, THRESHOLD):
            if not required.exists():
                raise FileNotFoundError(f"Required file is missing: {required}")
        stored = np.load(CLASSIFIER)
        self.train_features = stored["train_features"]
        self.train_labels = stored["train_labels"]
        self.labels = json.loads(LABELS.read_text(encoding="utf-8"))
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        threshold_config = json.loads(THRESHOLD.read_text(encoding="utf-8"))
        self.threshold = (
            float(threshold_override)
            if threshold_override is not None
            else float(threshold_config["threshold"])
        )
        self.detector = cv2.FaceDetectorYN.create(
            str(YUNET_MODEL), "", (320, 320), score_threshold=0.55,
            nms_threshold=0.3, top_k=5000,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")

    def predict_label(self, feature: np.ndarray) -> tuple[str, float]:
        similarities = feature @ self.train_features.T
        method = self.config["method"]
        if method == "centroid":
            centroids = np.stack([
                self.train_features[self.train_labels == index].mean(axis=0)
                for index in range(self.config["class_count"])
            ])
            centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
            predicted = int(np.argmax(feature @ centroids.T))
        else:
            neighbours = int(method.removeprefix("knn_"))
            nearest = np.argpartition(-similarities, neighbours - 1)[:neighbours]
            scores = np.zeros(self.config["class_count"], dtype=np.float64)
            for index in nearest:
                scores[self.train_labels[index]] += max(float(similarities[index]), 0.0) ** 4
            predicted = int(scores.argmax())
        similarity = float(similarities[self.train_labels == predicted].max())
        label = self.labels[str(predicted)] if similarity >= self.threshold else "Unknown"
        return label, similarity

    def observe(self, frame: np.ndarray) -> list[Observation]:
        self.detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []
        observations = []
        for face in faces:
            try:
                aligned = self.recognizer.alignCrop(frame, face)
                feature = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
            except cv2.error:
                continue
            feature /= max(float(np.linalg.norm(feature)), 1e-12)
            label, similarity = self.predict_label(feature)
            x, y, width, height = (int(round(value)) for value in face[:4])
            x = max(0, min(x, frame.shape[1] - 1))
            y = max(0, min(y, frame.shape[0] - 1))
            width = max(1, min(width, frame.shape[1] - x))
            height = max(1, min(height, frame.shape[0] - y))
            observations.append(Observation(
                box=(x, y, width, height), label=label, similarity=similarity,
                detection_confidence=float(face[-1]),
            ))
        return observations


def resize_frame(frame: np.ndarray, maximum_width: int) -> np.ndarray:
    if frame.shape[1] <= maximum_width:
        return frame
    scale = maximum_width / frame.shape[1]
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def signal_for_track(
    track: FaceTrack, signals: list[LivenessSignals]
) -> LivenessSignals | None:
    if not signals:
        return None
    overlaps = [(intersection_over_union(track.box, signal.box), signal) for signal in signals]
    overlap, signal = max(overlaps, key=lambda item: item[0])
    if overlap >= 0.10:
        return signal
    track_x, track_y, track_width, track_height = track.box
    signal_x, signal_y, signal_width, signal_height = signal.box
    signal_center = (signal_x + signal_width / 2, signal_y + signal_height / 2)
    if (
        track_x <= signal_center[0] <= track_x + track_width
        and track_y <= signal_center[1] <= track_y + track_height
    ):
        return signal
    return None


def draw_track(frame: np.ndarray, track: FaceTrack) -> None:
    x, y, width, height = track.box
    color = (
        (0, 80, 230) if track.display_label == "Unknown"
        else (0, 190, 255) if track.display_label == "Checking"
        else (60, 190, 70)
    )
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
    text = (
        f"Track {track.track_id}: Checking..."
        if track.display_label == "Checking"
        else f"{track.display_label}  {track.display_similarity:.2f}"
    )
    cv2.putText(frame, text, (x, max(24, y - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)
    live_color = (70, 220, 70) if track.liveness_passed else (0, 210, 255)
    cv2.putText(
        frame,
        track.liveness_text,
        (x, max(48, y - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        live_color,
        2,
    )
    if track.attendance_text:
        cv2.putText(
            frame,
            track.attendance_text,
            (x, min(frame.shape[0] - 10, y + height + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (65, 210, 90) if "marked" in track.attendance_text.lower() else (0, 190, 255),
            2,
        )


def write_event(path: Path | None, event: dict) -> None:
    encoded = json.dumps(event, ensure_ascii=False)
    print(encoded, flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(encoded + "\n")


class AttendanceClient:
    """Small fail-safe client for forwarding trusted camera events to Flask."""

    def __init__(self, api_url: str, token_file: Path, enabled: bool = True) -> None:
        self.api_url = api_url
        self.frame_url = api_url.rsplit("/api/", 1)[0] + "/api/camera-frame"
        self.enabled = enabled
        self.token = ""
        if enabled:
            if token_file.exists():
                self.token = token_file.read_text(encoding="utf-8").strip()
            if not self.token:
                print(
                    f"Attendance API disabled: device token not found at {token_file}. "
                    "Start the web application once or use --device-token-file.",
                    flush=True,
                )
                self.enabled = False

    def send(self, event: dict) -> dict:
        if not self.enabled:
            return {"outcome": "api_disabled", "message": "Database sync disabled"}
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(event).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Device-Token": self.token,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=2.5) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            try:
                payload = json.loads(error.read().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = {"error": f"HTTP {error.code}"}
            return {"outcome": "api_error", "message": payload.get("error", str(payload))}
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            return {"outcome": "api_offline", "message": f"Attendance server unavailable: {error}"}

    def send_frame(self, frame: np.ndarray) -> None:
        if not self.enabled:
            return
        if frame.shape[1] > 720:
            scale = 720 / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not encoded:
            return
        request = urllib.request.Request(
            self.frame_url,
            data=jpeg.tobytes(),
            headers={"Content-Type": "image/jpeg", "X-Device-Token": self.token},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=0.75):
                pass
        except (urllib.error.URLError, TimeoutError, OSError):
            pass


class FramePublisher:
    """Publish only the newest preview frame without blocking recognition."""

    def __init__(self, client: AttendanceClient, enabled: bool) -> None:
        self.client = client
        self.enabled = enabled and client.enabled
        self.frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self.stopping = False
        self.worker: threading.Thread | None = None
        if self.enabled:
            self.worker = threading.Thread(target=self._run, daemon=True)
            self.worker.start()

    def submit(self, frame: np.ndarray) -> None:
        if not self.enabled:
            return
        preview = frame.copy()
        try:
            self.frames.put_nowait(preview)
        except queue.Full:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frames.put_nowait(preview)
            except queue.Full:
                pass

    def _run(self) -> None:
        while not self.stopping:
            try:
                frame = self.frames.get(timeout=0.2)
            except queue.Empty:
                continue
            self.client.send_frame(frame)

    def close(self) -> None:
        self.stopping = True
        if self.worker is not None:
            self.worker.join(timeout=1.0)


def recognition_event(track: FaceTrack, runtime: SFaceRuntime) -> dict:
    return {
        "event": "recognition_and_liveness_passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "identity": track.display_label,
        "similarity": round(track.display_similarity, 6),
        "threshold": round(runtime.threshold, 6),
        "liveness": "passed",
        "challenges": [],
        "track_id": track.track_id,
    }


def source_value(value: str) -> int | str:
    return int(value) if value.isdecimal() else value


def report_camera_status(status: str, **details: object) -> None:
    print(json.dumps({"camera_status": status, **details}, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    source = source_value(args.source)
    camera_candidates = list(range(4)) if str(source).lower() == "auto" else (
        [source] if isinstance(source, int) else []
    )
    is_camera = bool(camera_candidates)
    pending_frame = None
    if is_camera:
        for candidate in camera_candidates:
            report_camera_status("starting", source=candidate)
            try:
                capture, pending_frame, startup_attempt = connect_camera(
                    candidate,
                    args.camera_reconnect_attempts if len(camera_candidates) == 1 else 3,
                    args.camera_reconnect_delay,
                )
            except RuntimeError:
                continue
            source = candidate
            report_camera_status("ready", source=source, attempt=startup_attempt)
            break
        else:
            tried = ", ".join(str(candidate) for candidate in camera_candidates)
            raise RuntimeError(
                f"No camera provided frames (tried indices {tried}). "
                "Close other camera applications and check Windows camera permission."
            )
    else:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open source {args.source}. Check the file path and format."
            )
    runtime = SFaceRuntime(args.min_similarity)
    landmark_analyzer = FaceLandmarkAnalyzer()
    liveness = LivenessController(timeout_seconds=args.liveness_timeout)
    tracker = TrackManager(args.history_size)
    event_log = None if args.no_event_log else args.event_log
    attendance_client = AttendanceClient(
        args.attendance_api,
        args.device_token_file,
        enabled=not args.no_attendance_api,
    )
    frame_publisher = FramePublisher(
        attendance_client, enabled=not args.no_ui_frame
    )
    last_event_by_identity: dict[str, float] = {}
    frame_number = 0
    processed_number = 0
    previous_time = time.perf_counter()
    displayed_fps = 0.0
    last_ui_frame = float("-inf")
    mirror = is_camera and not args.no_mirror
    consecutive_read_failures = 0
    reconnect_count = 0
    print(
        f"Recognition started: source={source}, threshold={runtime.threshold:.4f}. "
        "Press Q or Esc to stop.", flush=True,
    )
    try:
        while True:
            if pending_frame is not None:
                success, frame = True, pending_frame
                pending_frame = None
            else:
                success, frame = capture.read()
            if not success or frame is None:
                if not is_camera:
                    break
                consecutive_read_failures += 1
                if consecutive_read_failures < args.camera_read_failures:
                    if consecutive_read_failures == 1:
                        report_camera_status(
                            "degraded", source=source,
                            message="A camera frame was dropped; retrying.",
                        )
                    time.sleep(0.05)
                    continue
                report_camera_status(
                    "reconnecting", source=source,
                    reconnect_count=reconnect_count,
                )
                capture.release()
                capture, pending_frame, reconnect_attempt = connect_camera(
                    source, args.camera_reconnect_attempts, args.camera_reconnect_delay
                )
                reconnect_count += 1
                consecutive_read_failures = 0
                tracker = TrackManager(args.history_size)
                for track_id in list(liveness.sessions):
                    liveness.reset(track_id)
                report_camera_status(
                    "reconnected", source=source, attempt=reconnect_attempt,
                    reconnect_count=reconnect_count,
                )
                continue
            if consecutive_read_failures:
                consecutive_read_failures = 0
                report_camera_status("ready", source=source, reconnect_count=reconnect_count)
            frame_number += 1
            if mirror:
                frame = cv2.flip(frame, 1)
            frame = resize_frame(frame, args.max_width)
            now = time.monotonic()
            signals = landmark_analyzer.analyze(frame, int(now * 1000))
            if (frame_number - 1) % args.process_every == 0:
                processed_number += 1
                tracks = tracker.update(runtime.observe(frame), processed_number)
                for track in tracks:
                    label, similarity = track.confirm(args.confirmation_frames, args.confirmation_ratio)
                    track.display_label = label
                    track.display_similarity = similarity
            active_track_ids = set(tracker.tracks)
            for track_id in list(liveness.sessions):
                if track_id not in active_track_ids:
                    liveness.reset(track_id)
            for track in tracker.tracks.values():
                signal = signal_for_track(track, signals)
                session = liveness.update(
                    track.track_id, track.display_label, signal, now
                )
                if session is None:
                    track.liveness_passed = False
                    track.liveness_text = (
                        "Unknown - attendance blocked"
                        if track.display_label == "Unknown"
                        else "Confirming identity..."
                    )
                    continue
                track.liveness_passed = session.state == LivenessState.PASSED
                track.liveness_text = session.prompt()
                if session.state != LivenessState.PASSED:
                    continue
                label = track.display_label
                if not session.event_emitted:
                    last_event = last_event_by_identity.get(label, float("-inf"))
                    if now - last_event >= args.cooldown_seconds:
                        last_event_by_identity[label] = now
                        session.event_emitted = True
                        event = recognition_event(track, runtime)
                        event["challenges"] = [challenge.value for challenge in session.challenges]
                        write_event(event_log, event)
                if now - track.last_attendance_sync >= args.attendance_sync_seconds:
                    event = recognition_event(track, runtime)
                    event["challenges"] = [challenge.value for challenge in session.challenges]
                    result = attendance_client.send(event)
                    track.last_attendance_sync = now
                    track.attendance_text = result.get("message", result.get("outcome", "Attendance sync"))
                    print(
                        json.dumps({"attendance_sync": result, "identity": label}, ensure_ascii=False),
                        flush=True,
                    )
            for track in tracker.tracks.values():
                draw_track(frame, track)
            current_time = time.perf_counter()
            instantaneous = 1.0 / max(current_time - previous_time, 1e-9)
            previous_time = current_time
            displayed_fps = instantaneous if displayed_fps == 0 else 0.90 * displayed_fps + 0.10 * instantaneous
            cv2.putText(
                frame, f"FPS {displayed_fps:.1f} | threshold {runtime.threshold:.2f}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
            )
            if not args.no_ui_frame and now - last_ui_frame >= 1.0 / args.ui_frame_fps:
                frame_publisher.submit(frame)
                last_ui_frame = now
            if not args.headless:
                cv2.imshow("AI Face Recognition Attendance", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames and frame_number >= args.max_frames:
                break
    finally:
        frame_publisher.close()
        capture.release()
        landmark_analyzer.close()
        if not args.headless:
            cv2.destroyAllWindows()
    print(f"Recognition stopped after {frame_number} frames.", flush=True)


if __name__ == "__main__":
    main()
