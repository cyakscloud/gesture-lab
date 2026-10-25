"""PALM//PILOT: a small, local-first real-time hand gesture workstation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import threading
import time
import urllib.request
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "gesture_recognizer.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)
APP_NAME = "PALM//PILOT"

# MediaPipe's 21-point hand layout, expressed as landmark index pairs.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)

COLORS = {
    "ink": (17, 20, 27),
    "panel": (26, 31, 41),
    "panel_light": (39, 47, 60),
    "white": (238, 244, 248),
    "muted": (151, 166, 179),
    "cyan": (214, 239, 86),
    "mint": (125, 231, 184),
    "orange": (73, 166, 255),
    "red": (86, 96, 250),
}

GESTURE_NAMES = {
    "Closed_Fist": "Fist",
    "Open_Palm": "Open palm",
    "Pointing_Up": "Point",
    "Thumb_Down": "Thumbs down",
    "Thumb_Up": "Thumbs up",
    "Victory": "Peace",
    "ILoveYou": "I love you",
    "None": "No clear read",
}


@dataclass
class GestureEvent:
    time: str
    hand: str
    gesture: str
    confidence: float


class SessionTracker:
    """Turns frame-by-frame predictions into calmer, countable gesture events."""

    def __init__(self, stable_frames: int = 5, history_size: int = 7) -> None:
        self.stable_frames = stable_frames
        self.history: dict[str, deque[str]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self.active: dict[str, str] = {}
        self.last_seen: dict[str, float] = {}
        self.counts: Counter[str] = Counter()
        self.events: list[GestureEvent] = []

    def update(self, hand: str, gesture: str, confidence: float) -> str:
        now = time.monotonic()
        self.last_seen[hand] = now
        self.history[hand].append(gesture)
        winner, votes = Counter(self.history[hand]).most_common(1)[0]
        if votes >= self.stable_frames and winner not in {"No clear read", "—"}:
            if winner != self.active.get(hand):
                self.active[hand] = winner
                self.counts[winner] += 1
                self.events.append(
                    GestureEvent(
                        time=datetime.now().astimezone().strftime("%H:%M:%S"),
                        hand=hand,
                        gesture=winner,
                        confidence=round(confidence, 4),
                    )
                )
        return self.active.get(hand, winner)

    def expire_missing_hands(self, visible_hands: set[str]) -> None:
        now = time.monotonic()
        for hand in list(self.active):
            if hand not in visible_hands and now - self.last_seen.get(hand, 0) > 0.9:
                self.active.pop(hand, None)
                self.history.pop(hand, None)

    def reset(self) -> None:
        self.history.clear()
        self.active.clear()
        self.last_seen.clear()
        self.counts.clear()
        self.events.clear()


class LiveResultBuffer:
    """Keeps the newest async model result without stalling the camera loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[int, float] = {}
        self.result: Any = None
        self.timestamp_ms = -1
        self.updated_at = 0.0
        self.response_ms = 0.0
        self.completed = 0
        self.response_times: list[float] = []

    def submitted(self, timestamp_ms: int) -> None:
        now = time.perf_counter()
        with self._lock:
            self._pending[timestamp_ms] = now
            self._pending = {
                stamp: began for stamp, began in self._pending.items()
                if now - began < 3.0
            }

    def receive(self, result: Any, _output_image: Any, timestamp_ms: int) -> None:
        now = time.perf_counter()
        with self._lock:
            began = self._pending.pop(timestamp_ms, now)
            response = (now - began) * 1000
            self.result = result
            self.timestamp_ms = timestamp_ms
            self.updated_at = now
            self.response_ms = response
            self.completed += 1
            self.response_times.append(response)

    def snapshot(self) -> tuple[Any, int, float, float, int]:
        with self._lock:
            return self.result, self.timestamp_ms, self.updated_at, self.response_ms, self.completed

    def latencies(self) -> list[float]:
        with self._lock:
            return list(self.response_times)


def ensure_model() -> Path:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return MODEL_PATH

    print("Model bundle missing. Fetching Google's Gesture Recognizer model…")
    temporary_path = MODEL_PATH.with_suffix(".download")
    request = urllib.request.Request(
        MODEL_URL, headers={"User-Agent": "PalmPilot-GestureLab/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            total = int(response.headers.get("Content-Length", "0"))
            received = 0
            with temporary_path.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
                    received += len(block)
                    if total:
                        print(f"  {received / total:6.1%}  ({received:,} / {total:,} bytes)", end="\r")
        if temporary_path.stat().st_size < 1_000_000:
            raise RuntimeError("The downloaded model is unexpectedly small.")
        temporary_path.replace(MODEL_PATH)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"\nModel ready: {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1_000_000:.1f} MB)")
    return MODEL_PATH


def friendly_gesture(category: Any, score_threshold: float) -> tuple[str, float]:
    if category is None:
        return "No clear read", 0.0
    raw_name = getattr(category, "category_name", None) or getattr(category, "display_name", None) or "None"
    score = float(getattr(category, "score", 0.0))
    if raw_name == "None" or score < score_threshold:
        return "No clear read", score
    return GESTURE_NAMES.get(raw_name, raw_name.replace("_", " ")), score


def draw_hand(frame: Any, landmarks: list[Any], label: str, confidence: float) -> None:
    height, width = frame.shape[:2]
    points = [
        (max(0, min(width - 1, int(point.x * width))),
         max(0, min(height - 1, int(point.y * height))))
        for point in landmarks
    ]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], COLORS["cyan"], 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        radius = 4 if index in {4, 8, 12, 16, 20} else 3
        cv2.circle(frame, point, radius, COLORS["ink"], -1, cv2.LINE_AA)
        cv2.circle(frame, point, max(1, radius - 1), COLORS["mint"], -1, cv2.LINE_AA)

    x_min = min(point[0] for point in points)
    y_min = min(point[1] for point in points)
    text = f"{label}  {confidence:.0%}"
    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
    x = max(8, min(x_min, width - text_width - 26))
    y = max(88, y_min - 15)
    cv2.rectangle(frame, (x, y - 27), (x + text_width + 18, y + 5), COLORS["panel"], -1)
    cv2.rectangle(frame, (x, y - 27), (x + 4, y + 5), COLORS["mint"], -1)
    cv2.putText(frame, text, (x + 11, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                COLORS["white"], 2, cv2.LINE_AA)


def overlay_panel(frame: Any, fps: float, hand_count: int, tracker: SessionTracker,
                  inference_ms: float, source_label: str) -> None:
    height, width = frame.shape[:2]
    header_h, footer_h = 76, 86
    layer = frame.copy()
    cv2.rectangle(layer, (0, 0), (width, header_h), COLORS["ink"], -1)
    cv2.rectangle(layer, (0, height - footer_h), (width, height), COLORS["ink"], -1)
    cv2.addWeighted(layer, 0.88, frame, 0.12, 0, frame)

    cv2.putText(frame, APP_NAME, (24, 34), cv2.FONT_HERSHEY_DUPLEX, 0.83,
                COLORS["white"], 2, cv2.LINE_AA)
    cv2.putText(frame, "REAL-TIME HAND INTERFACE  /  LOCAL INFERENCE", (25, 57),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLORS["muted"], 1, cv2.LINE_AA)
    cv2.circle(frame, (width - 204, 34), 6, COLORS["red"], -1, cv2.LINE_AA)
    cv2.putText(frame, f"LIVE  ·  {source_label}", (width - 188, 39),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLORS["white"], 1, cv2.LINE_AA)

    labels = list(tracker.active.items())
    active_text = "   /   ".join(f"{hand}: {gesture}" for hand, gesture in labels) or "Show a hand to wake the rig"
    cv2.putText(frame, active_text[:74], (24, height - 49), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, COLORS["mint"] if labels else COLORS["muted"], 2, cv2.LINE_AA)
    cv2.putText(frame, "Q  quit    S  save session    R  reset counters", (24, height - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["muted"], 1, cv2.LINE_AA)

    right_x = width - 282
    cv2.rectangle(frame, (right_x, 94), (width - 18, min(height - footer_h - 14, 296)),
                  COLORS["panel"], -1)
    cv2.putText(frame, "SESSION TELEMETRY", (right_x + 16, 121),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLORS["muted"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:5.1f}", (right_x + 16, 164), cv2.FONT_HERSHEY_DUPLEX,
                1.13, COLORS["cyan"], 2, cv2.LINE_AA)
    cv2.putText(frame, "FPS  /  live", (right_x + 105, 161), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, COLORS["white"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"Hands tracked     {hand_count:>2}", (right_x + 16, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["white"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"Stable events     {len(tracker.events):>2}", (right_x + 16, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["white"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"Model response    {inference_ms:>4.0f} ms", (right_x + 16, 245),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["white"], 1, cv2.LINE_AA)
    top_count = tracker.counts.most_common(2)
    if top_count and height > 410:
        short_counts = "   ·   ".join(f"{name} ×{count}" for name, count in top_count)
        cv2.putText(frame, short_counts[:35], (right_x + 16, 273),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.39, COLORS["mint"], 1, cv2.LINE_AA)


def save_session(tracker: SessionTracker, metrics: dict[str, Any]) -> tuple[Path, Path]:
    session_dir = ROOT / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = session_dir / f"session_{stamp}.csv"
    json_path = session_dir / f"session_{stamp}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time", "hand", "gesture", "confidence"])
        writer.writeheader()
        writer.writerows(asdict(event) for event in tracker.events)
    summary = {**metrics, "gesture_counts": dict(tracker.counts), "events": [asdict(event) for event in tracker.events]}
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, json_path


def make_recognizer(model_path: Path, max_hands: int, score_threshold: float,
                    result_callback: Any) -> Any:
    options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        num_hands=max_hands,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
        canned_gesture_classifier_options=mp.tasks.components.processors.ClassifierOptions(
            score_threshold=score_threshold,
            max_results=1,
        ),
        result_callback=result_callback,
    )
    return mp.tasks.vision.GestureRecognizer.create_from_options(options)


def percentile(values: list[float], portion: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * portion))
    return ordered[index]


def run(args: argparse.Namespace) -> int:
    model_path = ensure_model()
    capture_source: Any = args.source if args.source else args.camera
    if args.source:
        capture = cv2.VideoCapture(capture_source)
        selected_backend = "FILE"
    else:
        backend_names = {
            "msmf": ("MSMF", cv2.CAP_MSMF),
            "dshow": ("DirectShow", cv2.CAP_DSHOW),
            "any": ("OpenCV auto", cv2.CAP_ANY),
        }
        order = (("msmf", "dshow") if sys.platform == "win32" else ("any",)) if args.backend == "auto" else (args.backend,)
        capture = None
        selected_backend = "unavailable"
        for key in order:
            backend_name, backend_id = backend_names[key]
            candidate = cv2.VideoCapture(capture_source, backend_id)
            if candidate.isOpened():
                capture = candidate
                selected_backend = backend_name
                break
            candidate.release()
        if capture is None:
            print(f"Could not open camera {args.camera} with {', '.join(order)}.", file=sys.stderr)
            return 2
    source_label = f"VIDEO · {Path(args.source).name}" if args.source else f"CAM {args.camera} · {selected_backend}"
    if not capture.isOpened():
        print(f"Could not open {source_label}. Try --camera 1, or pass --source path/to/video.mp4.", file=sys.stderr)
        return 2
    if not args.source:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, 30)

    tracker = SessionTracker()
    frame_times: deque[float] = deque(maxlen=40)
    frame_number = 0
    consumed_timestamp = -1
    live_results = LiveResultBuffer()
    render_hands: list[tuple[list[Any], str, float]] = []
    started = time.perf_counter()
    last_tick = started
    last_timestamp = -1
    print("Camera is live. Raise a hand; press Q to close the window.")
    print("Frames stay on this machine. Press S at any time to save session metrics and gesture events.")

    try:
        with make_recognizer(model_path, args.max_hands, args.score_threshold,
                             live_results.receive) as recognizer:
            window = f"{APP_NAME}  |  Gesture Lab"
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if args.mirror and not args.source:
                    frame = cv2.flip(frame, 1)
                if max(frame.shape[1], frame.shape[0]) > args.max_dimension:
                    scale = args.max_dimension / max(frame.shape[1], frame.shape[0])
                    frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = max(last_timestamp + 1, int((time.perf_counter() - started) * 1000))
                last_timestamp = timestamp
                live_results.submitted(timestamp)
                recognizer.recognize_async(image, timestamp)
                frame_number += 1

                result, result_timestamp, result_at, inference_ms, _ = live_results.snapshot()
                if result is not None and result_timestamp > consumed_timestamp:
                    visible_hands: set[str] = set()
                    next_render_hands: list[tuple[list[Any], str, float]] = []
                    for index, landmarks in enumerate(result.hand_landmarks):
                        handedness_group = result.handedness[index] if index < len(result.handedness) else []
                        hand = (handedness_group[0].category_name if handedness_group else f"Hand {index + 1}")
                        visible_hands.add(hand)
                        gesture_group = result.gestures[index] if index < len(result.gestures) else []
                        raw_gesture, confidence = friendly_gesture(
                            gesture_group[0] if gesture_group else None, args.score_threshold
                        )
                        display_gesture = tracker.update(hand, raw_gesture, confidence)
                        next_render_hands.append((landmarks, f"{hand}  ·  {display_gesture}", confidence))
                    render_hands = next_render_hands
                    tracker.expire_missing_hands(visible_hands)
                    consumed_timestamp = result_timestamp

                # Don't leave an old skeleton floating on screen after a camera/model hiccup.
                if result_at and time.perf_counter() - result_at > 0.9:
                    render_hands = []
                    tracker.expire_missing_hands(set())
                for landmarks, label, confidence in render_hands:
                    draw_hand(frame, landmarks, label, confidence)

                now = time.perf_counter()
                frame_times.append(now - last_tick)
                last_tick = now
                fps = 1.0 / statistics.mean(frame_times) if frame_times and statistics.mean(frame_times) > 0 else 0.0
                overlay_panel(frame, fps, len(render_hands), tracker, inference_ms, source_label)
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if args.duration and time.perf_counter() - started >= args.duration:
                    break
                if key == ord("s"):
                    metrics = current_metrics(started, frame_number, live_results.completed,
                                              live_results.latencies())
                    csv_path, json_path = save_session(tracker, metrics)
                    print(f"Saved {len(tracker.events)} events → {csv_path.name} and {json_path.name}")
                if key == ord("r"):
                    tracker.reset()
                    print("Session counters reset.")
    except KeyboardInterrupt:
        print("\nStopping camera loop…")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    metrics = current_metrics(started, frame_number, live_results.completed,
                              live_results.latencies())
    csv_path, json_path = save_session(tracker, metrics)
    print("\nSession recap")
    print(json.dumps({**metrics, "gesture_counts": dict(tracker.counts)}, indent=2))
    print(f"Event log: {csv_path}")
    print(f"Metrics:   {json_path}")
    return 0


def current_metrics(started: float, frame_count: int, recognition_outputs: int,
                    inference_times: list[float]) -> dict[str, Any]:
    elapsed = max(time.perf_counter() - started, 0.001)
    return {
        "frames_processed": frame_count,
        "recognition_outputs": recognition_outputs,
        "elapsed_seconds": round(elapsed, 2),
        "camera_fps": round(frame_count / elapsed, 2),
        "mean_model_response_ms": round(statistics.mean(inference_times), 2) if inference_times else 0.0,
        "p95_model_response_ms": round(percentile(inference_times, 0.95), 2),
        "runtime": "CPU",
        "video_saved": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PALM//PILOT — real-time hand gestures, read locally from your camera."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0).")
    parser.add_argument("--source", type=str, help="Optional video file for a no-webcam demo.")
    parser.add_argument("--width", type=int, default=640, help="Requested webcam width.")
    parser.add_argument("--height", type=int, default=480, help="Requested webcam height.")
    parser.add_argument("--max-dimension", type=int, default=1280, help="Resize large frames to fit this dimension.")
    parser.add_argument("--backend", choices=("auto", "msmf", "dshow", "any"), default="auto",
                        help="Camera driver backend; auto tries Media Foundation first on Windows.")
    parser.add_argument("--max-hands", type=int, choices=range(1, 3), default=1, help="Track one or two hands.")
    parser.add_argument("--score-threshold", type=float, default=0.45, help="Minimum gesture confidence (0–1).")
    parser.add_argument("--duration", type=float, default=0,
                        help="Stop after this many seconds (handy for a quick runtime check).")
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True,
                        help="Mirror webcam view like a selfie camera (use --no-mirror to turn it off).")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
