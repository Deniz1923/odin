"""Spatial association and time-spaced identity evidence, separate from HTTP/video IO."""

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    threshold: float = 0.35
    margin: float = Field(default=0.02, ge=0)
    min_observations: int = Field(default=3, ge=1, le=16)
    min_span: float = Field(default=0.5, ge=0, le=2)
    observation_spacing: float = Field(default=0.25, ge=0.05, le=1)
    evidence_window: float = Field(default=2, ge=0.5, le=10)
    stale_after: float = Field(default=1, ge=0.25, le=5)
    track_ttl: float = Field(default=2, ge=0.5, le=10)
    cooldown: float = Field(default=30, ge=0, le=3600)

    @model_validator(mode="after")
    def feasible_evidence(self):
        required_span = max(self.min_span, (self.min_observations - 1) * self.observation_spacing)
        if required_span > self.evidence_window or self.observation_spacing > self.stale_after:
            raise ValueError("Temporal settings cannot accumulate the required evidence")
        return self


def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by)
    )
    return intersection / max(1, aw * ah + bw * bh - intersection)


@dataclass
class Track:
    id: int
    bbox: list
    seen: float
    state: str = "insufficient"
    person_id: str | None = None
    score: float | None = None
    margin: float | None = None
    reason: str = "collecting_evidence"
    evidence: deque = field(default_factory=lambda: deque(maxlen=16))
    last_evidence: float = -float("inf")
    alerted: set = field(default_factory=set)
    methods: list = field(default_factory=list)

    def reset(self, reason):
        self.state, self.person_id, self.reason = "insufficient", None, reason
        self.evidence.clear()
        self.last_evidence = -float("inf")
        self.score = self.margin = None
        self.methods = []


class Pipeline:
    def __init__(self, gallery, enabled, policy: Policy, detector=None, *, recognizer):
        recognizer.validate_settings(policy.threshold, policy.margin)
        self.gallery, self.enabled, self.policy = gallery, enabled, policy
        self.detector = detector
        self.recognizer = recognizer
        self.tracks: list[Track] = []
        self.next_id = 1
        self.last_time = -float("inf")

    def process(self, frame: np.ndarray, timestamp: float) -> dict:
        started = time.perf_counter()
        found = self.detector.detect(frame)[:32]
        detected = time.perf_counter()
        detections = []
        for detection in found:
            descriptor, reason = self.recognizer.extract(frame, detection)
            detections.append(dict(**detection, reason=reason, descriptor=descriptor))
        extracted = time.perf_counter()
        result = self.update(detections, timestamp)
        result["timings"] = dict(
            detect_ms=(detected - started) * 1000,
            extract_ms=(extracted - detected) * 1000,
            match_track_ms=(time.perf_counter() - extracted) * 1000,
        )
        return result

    def method_evidence(self, result: dict) -> list[dict]:
        """Per-recognizer evidence for the latest observation; one entry per executed method."""
        recognizer = self.recognizer
        return [
            dict(
                method=recognizer.name,
                score_kind=recognizer.score_kind,
                higher_is_better=recognizer.higher_is_better,
                state=result["state"],
                person_id=result["person_id"],
                score=result["score"],
                margin=result["margin"],
                threshold=self.policy.threshold,
                required_margin=self.policy.margin,
            )
        ]

    def update(self, detections: list[dict], timestamp: float) -> dict:
        if not np.isfinite(timestamp) or timestamp < self.last_time:
            raise ValueError("Frame time must be finite and monotonic")
        self.last_time = timestamp
        p = self.policy
        self.tracks = [t for t in self.tracks if timestamp - t.seen <= p.track_ttl]
        for track in self.tracks:
            if timestamp - track.last_evidence > p.stale_after:
                track.reset("stale_evidence")
        detections = detections[:32]
        pairs = [
            (overlap(t.bbox, d["bbox"]), i, j)
            for i, t in enumerate(self.tracks)
            for j, d in enumerate(detections)
            if overlap(t.bbox, d["bbox"]) >= 0.2
        ]
        used_tracks, used_detections, assigned = set(), set(), []
        ambiguous_tracks = {
            i
            for _, i, j in pairs
            if sum(pair[1] == i for pair in pairs) > 1 or sum(pair[2] == j for pair in pairs) > 1
        }
        for i in ambiguous_tracks:
            self.tracks[i].reset("ambiguous_association")
        for _, i, j in sorted(pairs, reverse=True):
            if i in used_tracks or j in used_detections:
                continue
            ambiguous = sum(x[1] == i for x in pairs) > 1 or sum(x[2] == j for x in pairs) > 1
            assigned.append((self.tracks[i], detections[j], ambiguous))
            used_tracks.add(i)
            used_detections.add(j)
        for j, detection in enumerate(detections):
            if j not in used_detections and len(self.tracks) < 32:
                track = Track(self.next_id, detection["bbox"], timestamp)
                self.next_id += 1
                self.tracks.append(track)
                assigned.append((track, detection, any(x[2] == j for x in pairs)))
        events = []
        for track, detection, ambiguous in assigned:
            track.bbox, track.seen = detection["bbox"], timestamp
            if ambiguous:
                track.reset("ambiguous_association")
                continue
            descriptor = detection["descriptor"]
            if descriptor is None:
                track.reason = detection["reason"]
                continue
            result = self.recognizer.match(descriptor, self.gallery, p.threshold, p.margin)
            identity = result["person_id"]
            # Space positive evidence, never checks for contradictory identity evidence.
            if (
                timestamp - track.last_evidence < p.observation_spacing
                and identity == track.person_id
                and result["state"] == "candidate"
            ):
                track.score, track.margin = result["score"], result["margin"]
                track.methods = self.method_evidence(result)
                continue
            if result["state"] != "candidate":
                track.reset(result["state"])
                track.state = result["state"]
            else:
                if track.person_id is not None and track.person_id != identity:
                    track.reset("identity_conflict")
                while track.evidence and timestamp - track.evidence[0] > p.evidence_window:
                    track.evidence.popleft()
                track.evidence.append(timestamp)
                track.person_id = identity
                confirmed = (
                    len(track.evidence) >= p.min_observations
                    and timestamp - track.evidence[0] >= p.min_span
                )
                track.state = "confirmed" if confirmed else "candidate"
                track.reason = "temporal_support" if confirmed else "collecting_evidence"
                if confirmed and identity in self.enabled and identity not in track.alerted:
                    events.append(
                        dict(
                            person_id=identity,
                            track=track.id,
                            score=result["score"],
                            score_kind=self.recognizer.score_kind,
                            observations=len(track.evidence),
                        )
                    )
                    track.alerted.add(identity)
            track.score, track.margin = result["score"], result["margin"]
            track.methods = self.method_evidence(result)
            track.last_evidence = timestamp
        return dict(
            tracks=[
                dict(
                    track_id=t.id,
                    bbox=t.bbox,
                    state=t.state,
                    person_id=t.person_id,
                    score=t.score,
                    score_kind=self.recognizer.score_kind,
                    margin=t.margin,
                    reason=t.reason,
                    observations=len(t.evidence),
                    evidence_age=None
                    if not np.isfinite(t.last_evidence)
                    else timestamp - t.last_evidence,
                    missing=t.seen < timestamp,
                    methods=[dict(entry) for entry in t.methods],
                )
                for t in self.tracks
            ],
            events=events,
        )
