"""On-device active liveness signals and randomized challenge state machine."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
FACE_LANDMARKER_MODEL = ROOT / "models" / "pretrained" / "face_landmarker.task"


class Challenge(str, Enum):
    BLINK = "blink"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"


class LivenessState(str, Enum):
    VERIFYING = "verifying"
    PASSED = "passed"
    FAILED = "failed"


@dataclass
class LivenessSignals:
    box: tuple[int, int, int, int]
    blink_left: float
    blink_right: float
    yaw_degrees: float
    pitch_degrees: float


class FaceLandmarkAnalyzer:
    """Produce per-face blink and pose signals from video frames."""

    def __init__(self, maximum_faces: int = 6) -> None:
        if not FACE_LANDMARKER_MODEL.exists():
            raise FileNotFoundError(
                f"Missing {FACE_LANDMARKER_MODEL}; run src/download_face_models.py"
            )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=maximum_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.last_timestamp_ms = -1

    def close(self) -> None:
        self.landmarker.close()

    def analyze(self, frame: np.ndarray, timestamp_ms: int) -> list[LivenessSignals]:
        timestamp_ms = max(timestamp_ms, self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        count = min(
            len(result.face_landmarks),
            len(result.face_blendshapes),
            len(result.facial_transformation_matrixes),
        )
        signals = []
        for index in range(count):
            landmarks = result.face_landmarks[index]
            blendshapes = {
                category.category_name: float(category.score)
                for category in result.face_blendshapes[index]
            }
            matrix = np.asarray(result.facial_transformation_matrixes[index])
            pitch, yaw, _ = cv2.RQDecomp3x3(matrix[:3, :3])[0]
            xs = [landmark.x for landmark in landmarks]
            ys = [landmark.y for landmark in landmarks]
            left = max(0, int(min(xs) * frame.shape[1]))
            top = max(0, int(min(ys) * frame.shape[0]))
            right = min(frame.shape[1], int(max(xs) * frame.shape[1]))
            bottom = min(frame.shape[0], int(max(ys) * frame.shape[0]))
            signals.append(
                LivenessSignals(
                    box=(left, top, max(1, right - left), max(1, bottom - top)),
                    blink_left=blendshapes.get("eyeBlinkLeft", 0.0),
                    blink_right=blendshapes.get("eyeBlinkRight", 0.0),
                    yaw_degrees=float(yaw),
                    pitch_degrees=float(pitch),
                )
            )
        return signals


@dataclass
class LivenessSession:
    identity: str
    challenges: tuple[Challenge, ...]
    started_at: float
    timeout_seconds: float
    state: LivenessState = LivenessState.VERIFYING
    challenge_index: int = 0
    baseline_yaw: float | None = None
    closed_frames: int = 0
    blink_closed_seen: bool = False
    pose_frames: int = 0
    finished_at: float | None = None
    event_emitted: bool = False

    @property
    def current_challenge(self) -> Challenge | None:
        if self.challenge_index >= len(self.challenges):
            return None
        return self.challenges[self.challenge_index]

    def prompt(self) -> str:
        if self.state == LivenessState.PASSED:
            return "LIVE - verified"
        if self.state == LivenessState.FAILED:
            return "Liveness timed out - retrying"
        prompts = {
            Challenge.BLINK: "Liveness: blink once",
            Challenge.TURN_LEFT: "Liveness: turn head left",
            Challenge.TURN_RIGHT: "Liveness: turn head right",
        }
        challenge = self.current_challenge
        return prompts.get(challenge, "Liveness: hold still")

    def update(
        self,
        signals: LivenessSignals | None,
        now: float,
        blink_closed_threshold: float = 0.45,
        blink_open_threshold: float = 0.25,
        yaw_threshold_degrees: float = 14.0,
        consecutive_pose_frames: int = 2,
    ) -> LivenessState:
        if self.state != LivenessState.VERIFYING:
            return self.state
        if now - self.started_at > self.timeout_seconds:
            self.state = LivenessState.FAILED
            self.finished_at = now
            return self.state
        if signals is None:
            return self.state
        if self.baseline_yaw is None:
            self.baseline_yaw = signals.yaw_degrees
        challenge = self.current_challenge
        if challenge == Challenge.BLINK:
            closed = (
                signals.blink_left >= blink_closed_threshold
                and signals.blink_right >= blink_closed_threshold
            )
            opened = (
                signals.blink_left <= blink_open_threshold
                and signals.blink_right <= blink_open_threshold
            )
            self.closed_frames = self.closed_frames + 1 if closed else 0
            if self.closed_frames >= 1:
                self.blink_closed_seen = True
            if self.blink_closed_seen and opened:
                self._complete_challenge(now)
        elif challenge in {Challenge.TURN_LEFT, Challenge.TURN_RIGHT}:
            yaw_change = signals.yaw_degrees - self.baseline_yaw
            reached = (
                yaw_change <= -yaw_threshold_degrees
                if challenge == Challenge.TURN_LEFT
                else yaw_change >= yaw_threshold_degrees
            )
            self.pose_frames = self.pose_frames + 1 if reached else 0
            if self.pose_frames >= consecutive_pose_frames:
                self._complete_challenge(now)
        return self.state

    def _complete_challenge(self, now: float) -> None:
        self.challenge_index += 1
        self.closed_frames = 0
        self.blink_closed_seen = False
        self.pose_frames = 0
        if self.challenge_index >= len(self.challenges):
            self.state = LivenessState.PASSED
            self.finished_at = now


class LivenessController:
    """Own one randomized liveness session per recognized face track."""

    def __init__(
        self,
        timeout_seconds: float = 12.0,
        retry_delay_seconds: float = 2.0,
        rng: random.Random | random.SystemRandom | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.retry_delay_seconds = retry_delay_seconds
        self.rng = rng or random.SystemRandom()
        self.sessions: dict[int, LivenessSession] = {}

    def _challenges(self) -> tuple[Challenge, Challenge]:
        challenges = [Challenge.BLINK, self.rng.choice([Challenge.TURN_LEFT, Challenge.TURN_RIGHT])]
        self.rng.shuffle(challenges)
        return tuple(challenges)

    def reset(self, track_id: int) -> None:
        self.sessions.pop(track_id, None)

    def update(
        self,
        track_id: int,
        identity: str,
        signals: LivenessSignals | None,
        now: float,
    ) -> LivenessSession | None:
        if identity in {"Checking", "Unknown"}:
            self.reset(track_id)
            return None
        session = self.sessions.get(track_id)
        identity_changed = session is not None and session.identity != identity
        retry_ready = (
            session is not None
            and session.state == LivenessState.FAILED
            and session.finished_at is not None
            and now - session.finished_at >= self.retry_delay_seconds
        )
        if session is None or identity_changed or retry_ready:
            session = LivenessSession(
                identity=identity,
                challenges=self._challenges(),
                started_at=now,
                timeout_seconds=self.timeout_seconds,
            )
            self.sessions[track_id] = session
        session.update(signals, now)
        return session
