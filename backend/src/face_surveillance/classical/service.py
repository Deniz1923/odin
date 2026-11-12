"""One-source service: bounded capture, private enrollment, tracking and events."""

import copy
import hashlib
import io
import json
import os
import queue
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from .detection import HaarDetector
from .pipeline import Pipeline, Policy
from .recognition import LBPRecognizer
from .storage import Store


def decode_image(content: bytes) -> np.ndarray:
    if not content or len(content) > 8_000_000:
        raise ValueError("Each image must be at most 8 MB")
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.width * image.height > 12_000_000:
                raise ValueError("Image exceeds 12 megapixels")
            image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid image") from exc
    frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Image cannot be decoded")
    return frame


class RuntimeWindow:
    """Rolling measurements over recently processed frames; operator feedback, not a benchmark."""

    def __init__(self, size: int = 120):
        self.frames = deque(maxlen=size)

    def add(self, finished: float, captured: int, total_ms: float, stages: dict):
        self.frames.append((finished, captured, total_ms, stages))

    def summary(self) -> dict | None:
        if not self.frames:
            return None
        totals = np.array([frame[2] for frame in self.frames])
        span = self.frames[-1][0] - self.frames[0][0]
        rate = None if len(self.frames) < 2 or span <= 0 else (len(self.frames) - 1) / span
        input_rate = (
            None
            if len(self.frames) < 2 or span <= 0
            else (self.frames[-1][1] - self.frames[0][1]) / span
        )
        stages = {
            name: float(np.median([frame[3][name] for frame in self.frames]))
            for name in self.frames[-1][3]
        }
        return dict(
            window_frames=len(self.frames),
            processed_fps=rate,
            input_fps=input_rate,
            processing_ms_median=float(np.median(totals)),
            processing_ms_p95=float(np.percentile(totals, 95)),
            stage_ms_median=stages,
        )


class Service:
    def __init__(self, directory: Path, *, detector=None, recognizer=None):
        self.lock = threading.RLock()
        self.recognizer = recognizer if recognizer is not None else LBPRecognizer()
        self.store = Store(
            directory,
            retention_days=int(os.environ.get("ODIN_EVENT_RETENTION_DAYS", "7")),
            recognizer=self.recognizer,
        )
        for orphan in self.store.directory.glob("upload-*.video"):
            orphan.unlink(missing_ok=True)
        self.detector = detector if detector is not None else HaarDetector()
        self.policy_path = self.store.directory / "policy.json"
        if self.policy_path.exists():
            saved = json.loads(self.policy_path.read_text())
            # Files without a space tag predate the recognizer boundary and contain LBP settings.
            if saved.get("descriptor_space", LBPRecognizer.space) != self.recognizer.space:
                raise ValueError("Incompatible decision settings descriptor space")
            self.policy = Policy.model_validate(saved.get("policy", saved))
        else:
            self.policy = Policy(
                threshold=self.recognizer.default_threshold,
                margin=self.recognizer.default_margin,
            )
        self.recognizer.validate_settings(self.policy.threshold, self.policy.margin)
        gallery = self.store.gallery()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="odin-command")
        self.command_slots = threading.BoundedSemaphore(8)
        self.stop_event = threading.Event()
        self.generation, self.worker, self.owned_path = 0, None, None
        self.capture_worker = None
        self.jpeg, self.last_frame_at = None, None
        self.snapshot = dict(
            state="stopped",
            source=None,
            session=None,
            tracks=[],
            processed=0,
            captured=0,
            dropped=0,
            error=None,
        )
        self.reload_gallery(gallery)

    def reload_gallery(self, gallery=None):
        people = self.store.people()
        self.names = {p["id"]: p["name"] for p in people}
        enabled = {p["id"] for p in people if p["alert_enabled"]}
        next_id = self.pipeline.next_id if hasattr(self, "pipeline") else 1
        self.pipeline = Pipeline(
            self.store.gallery() if gallery is None else gallery,
            enabled,
            self.policy,
            self.detector,
            recognizer=self.recognizer,
        )
        self.pipeline.next_id = next_id
        self.snapshot["tracks"], self.jpeg = [], None

    def enroll(self, name, images, authorized, alert_enabled=False):
        if not authorized:
            raise ValueError("Confirm authorization before enrollment")
        if not 1 <= len(images) <= 32:
            raise ValueError("Upload 1–32 distinct images; several varied samples are recommended")
        if len({hashlib.sha256(image).hexdigest() for image in images}) != len(images):
            raise ValueError("Duplicate enrollment images")
        with self.lock:
            descriptors = []
            for index, content in enumerate(images):
                frame = decode_image(content)
                detections = self.detector.detect(frame)
                if len(detections) != 1:
                    raise ValueError(
                        f"Image {index + 1}: expected exactly one face, found {len(detections)}"
                    )
                descriptor, reason = self.recognizer.extract(frame, detections[0])
                if descriptor is None:
                    raise ValueError(f"Image {index + 1}: {reason}")
                descriptors.append(descriptor)
            result = self.store.enroll(
                name, descriptors, authorized=authorized, alert_enabled=alert_enabled
            )
            self.reload_gallery()
            return result

    def delete(self, person_id):
        with self.lock:
            self.store.delete(person_id)
            self.reload_gallery()

    def set_alert(self, person_id, enabled):
        with self.lock:
            self.store.set_alert(person_id, enabled)
            self.reload_gallery()

    def configure(self, policy: Policy):
        self.recognizer.validate_settings(policy.threshold, policy.margin)
        self.stop()
        with self.lock:
            staged = self.policy_path.with_suffix(".tmp")
            staged.write_text(
                json.dumps(
                    dict(descriptor_space=self.recognizer.space, policy=policy.model_dump()),
                    indent=2,
                )
                + "\n"
            )
            staged.chmod(0o600)
            staged.replace(self.policy_path)
            self.policy = policy
            self.reload_gallery()

    def start(self, source: str | int, label: str, *, owned=False):
        self.stop()
        with self.lock:
            if self.capture_worker and self.capture_worker.is_alive():
                raise ValueError(
                    "Previous capture has not released; retry or restart the application"
                )
            self.generation += 1
            self.stop_event = threading.Event()
            self.reload_gallery()
            self.owned_path = Path(source) if owned else None
            self.last_frame_at = None
            self.snapshot = dict(
                state="starting",
                source=label,
                session=uuid.uuid4().hex,
                tracks=[],
                processed=0,
                captured=0,
                dropped=0,
                error=None,
                mode="replay" if isinstance(source, str) else "webcam",
                timestamp_basis="frame_index/fps"
                if isinstance(source, str)
                else "host_receipt_monotonic",
            )
            self.worker = threading.Thread(
                target=self._run,
                args=(source, self.generation, self.stop_event, self.owned_path),
                daemon=True,
            )
            self.worker.start()

    def stop(self):
        with self.lock:
            self.generation += 1
            self.stop_event.set()
            self.jpeg = None
            self.snapshot.update(state="stopped", tracks=[])
            if self.owned_path:
                self.owned_path.unlink(missing_ok=True)
                self.owned_path = None
        if self.worker and self.worker is not threading.current_thread():
            self.worker.join(timeout=1)

    def close(self):
        self.stop()
        self.executor.shutdown(wait=False, cancel_futures=True)

    def status(self):
        with self.lock:
            status = copy.deepcopy(self.snapshot)
            status.update(
                policy=self.policy.model_dump(),
                calibrated=False,
                descriptor_space=self.recognizer.space,
                recognizer=self.recognizer.name,
                methods=[
                    dict(
                        name=self.recognizer.name,
                        score_kind=self.recognizer.score_kind,
                        score_range=self.recognizer.score_range,
                        higher_is_better=self.recognizer.higher_is_better,
                        threshold=self.policy.threshold,
                        margin=self.policy.margin,
                    )
                ],
                decision_rule="single_method",
                retention_days=self.store.retention_days,
                score_kind=self.recognizer.score_kind,
                score_range=self.recognizer.score_range,
                higher_is_better=self.recognizer.higher_is_better,
                detector=self.detector.name,
                frame_age=None
                if self.last_frame_at is None
                else time.monotonic() - self.last_frame_at,
            )
            return status

    def preview(self):
        with self.lock:
            return self.jpeg

    def _capture(self, source, stop, slot, counts):
        capture = None

        def publish(value, *, terminal=False):
            if terminal:
                while not stop.is_set():
                    try:
                        slot.put(value, timeout=0.1)
                        return
                    except queue.Full:
                        pass
                return
            try:
                slot.put_nowait(value)
            except queue.Full:
                try:
                    slot.get_nowait()
                except queue.Empty:
                    pass
                counts["dropped"] += 1
                slot.put_nowait(value)

        try:
            capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                raise ValueError("Cannot open camera/video")
            if (
                capture.get(cv2.CAP_PROP_FRAME_WIDTH) * capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                > 8_500_000
            ):
                raise ValueError("Video exceeds supported frame dimensions")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if isinstance(source, str) and not 1 <= fps <= 240:
                raise ValueError("Replay needs constant-frame-rate video (1–240 FPS)")
            start, index = time.monotonic(), 0
            while not stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    if isinstance(source, str) and index == 0:
                        raise ValueError("Video contains no decodable frames")
                    expected = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                    if isinstance(source, str) and expected > 0 and index + 1 < expected:
                        raise ValueError("Video ended before its declared frame count")
                    publish(("end", None, None, None), terminal=True)
                    return
                if frame.shape[0] * frame.shape[1] > 8_500_000:
                    raise ValueError("Video frame exceeds 8.5 megapixels")
                if isinstance(source, str) and stop.wait(
                    max(0, start + index / fps - time.monotonic())
                ):
                    return
                now = time.monotonic()
                stamp = index / fps if isinstance(source, str) else now - start
                counts["captured"] += 1
                publish(("frame", frame, stamp, now))
                index += 1
        except Exception as exc:
            publish(("error", str(exc), None, None), terminal=True)
        finally:
            if capture is not None:
                capture.release()

    def _run(self, source, generation, stop, owned_path):
        slot, counts = queue.Queue(maxsize=1), dict(captured=0, dropped=0)
        capture = threading.Thread(
            target=self._capture, args=(source, stop, slot, counts), daemon=True
        )
        self.capture_worker = capture
        capture.start()
        last_received = time.monotonic()
        runtime = RuntimeWindow()
        try:
            while not stop.is_set():
                try:
                    kind, value, stamp, received = slot.get(timeout=0.25)
                except queue.Empty:
                    if time.monotonic() - last_received > 10:
                        raise ValueError("No fresh frame for 10 seconds")
                    continue
                with self.lock:
                    if generation != self.generation:
                        return
                    if kind != "frame":
                        if kind == "error":
                            raise ValueError(value)
                        self.snapshot.update(
                            state="ended" if isinstance(source, str) else "disconnected",
                            tracks=[],
                            **counts,
                        )
                        self.jpeg = None
                        return
                    last_received = received
                    queue_ms = (time.monotonic() - received) * 1000
                    started = time.perf_counter()
                    result = self.pipeline.process(value, stamp)
                    events_started = time.perf_counter()
                    for event in result["events"]:
                        self.store.add_event(
                            source=self.snapshot["source"],
                            session=self.snapshot["session"],
                            policy={
                                **self.policy.model_dump(),
                                "descriptor_space": self.recognizer.space,
                                "score_kind": self.recognizer.score_kind,
                                "detector_hash": self.detector.model_hash,
                                "calibrated": False,
                            },
                            cooldown=self.policy.cooldown,
                            **event,
                        )
                    preview_started = time.perf_counter()
                    for track in result["tracks"]:
                        track["name"] = self.names.get(track["person_id"])
                        if not track["missing"]:
                            x, y, w, h = track["bbox"]
                            color = (
                                (90, 210, 100) if track["state"] == "confirmed" else (80, 190, 240)
                            )
                            cv2.rectangle(value, (x, y), (x + w, y + h), color, 2)
                            text = f"#{track['track_id']} {track['name'] or track['state']}"
                            cv2.putText(
                                value,
                                text,
                                (x, max(20, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                1,
                            )
                    scale = min(1, 960 / max(value.shape[:2]))
                    preview = cv2.resize(value, None, fx=scale, fy=scale)
                    ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    self.jpeg = encoded.tobytes() if ok else None
                    self.last_frame_at = received
                    finished = time.perf_counter()
                    total_ms = (finished - started) * 1000
                    runtime.add(
                        time.monotonic(),
                        counts["captured"],
                        total_ms,
                        dict(
                            **result["timings"],
                            events_ms=(preview_started - events_started) * 1000,
                            preview_ms=(finished - preview_started) * 1000,
                            queue_ms=queue_ms,
                        ),
                    )
                    self.snapshot.update(
                        state="running",
                        tracks=result["tracks"],
                        processed=self.snapshot["processed"] + 1,
                        media_time=stamp,
                        processing_ms=total_ms,
                        runtime=runtime.summary(),
                        **counts,
                    )
        except Exception as exc:
            with self.lock:
                if generation == self.generation:
                    self.snapshot.update(state="failed", error=str(exc), tracks=[])
                    self.jpeg = None
        finally:
            stop.set()
            capture.join(timeout=0.5)
            if owned_path:
                owned_path.unlink(missing_ok=True)
