"""Offline, exhaustive classical replay and validation-only threshold selection.

Example: python -m face_surveillance.classical.evaluate --manifest study.json --output runs/study-01
"""

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .detection import HaarDetector, prepare_face
from .lbp import SPACE, describe, match
from .pipeline import Pipeline, Policy, overlap
from .service import decode_image


def assignments(predictions, truth):
    pairs = sorted(
        [
            (overlap(p["bbox"], t["bbox"]), i, j)
            for i, p in enumerate(predictions)
            for j, t in enumerate(truth)
        ],
        reverse=True,
    )
    used_p, used_t, result = set(), set(), {}
    for score, i, j in pairs:
        if score >= 0.3 and i not in used_p and j not in used_t:
            result[j] = predictions[i]
            used_p.add(i)
            used_t.add(j)
    return result


def score_frame(tracks, truth, known_ids):
    predictions = [t for t in tracks if not t.get("missing", False)]
    matched = assignments(predictions, truth)
    result = Counter(
        dict(
            faces=len(truth),
            detected=len(matched),
            false_detections=len(predictions) - len(matched),
            known=0,
            known_correct=0,
            known_wrong=0,
            known_detection_missed=0,
            known_rejected=0,
            known_unresolved=0,
            unknown=0,
            unknown_false_accept=0,
            unknown_rejected=0,
            unknown_unresolved=0,
        )
    )
    for index, target in enumerate(truth):
        known = target["person_id"] in known_ids
        result["known" if known else "unknown"] += 1
        prediction = matched.get(index)
        if prediction is None:
            result["known_detection_missed" if known else "unknown_unresolved"] += 1
        elif prediction["state"] == "confirmed":
            if known:
                result[
                    "known_correct"
                    if prediction["person_id"] == target["person_id"]
                    else "known_wrong"
                ] += 1
            else:
                result["unknown_false_accept"] += 1
        elif prediction["state"] == "unknown":
            result["known_rejected" if known else "unknown_rejected"] += 1
        else:
            result["known_unresolved" if known else "unknown_unresolved"] += 1
    return dict(result)


def score_events(events, encounters):
    used, correct, duplicates, false, delays = set(), 0, 0, 0, []
    for event in events:
        candidates = [
            i
            for i, encounter in enumerate(encounters)
            if encounter["person_id"] == event["person_id"]
            and encounter["start"] <= event["time"] <= encounter["end"]
        ]
        available = [i for i in candidates if i not in used]
        if available:
            index = available[0]
            used.add(index)
            correct += 1
            delays.append(event["time"] - encounters[index]["start"])
        elif candidates:
            duplicates += 1
        else:
            false += 1
    return dict(
        correct=correct,
        duplicates=duplicates,
        false=false,
        missed=len(encounters) - len(used),
        eligible_encounters=len(encounters),
        delay_seconds=delays,
    )


def select_threshold(rows, *, split, max_unknown_fpir=0.05, margin=0.02):
    if split != "validation":
        raise ValueError("Calibration requires a validation split, never the final test split")
    known = sum(row["known"] for row in rows)
    unknown = len(rows) - known
    if not known or not unknown:
        raise ValueError("Calibration requires usable known and unknown observations")
    if not 0 <= max_unknown_fpir <= 1:
        raise ValueError("Unknown FPIR constraint must lie between zero and one")
    choices = []
    for threshold in sorted({0.0, *(row["distance"] for row in rows)}):
        accepted = [
            row
            for row in rows
            if row["distance"] <= threshold
            and (row["margin"] is None or row["margin"] >= margin)
            and row["margin"] != 0
        ]
        correct = sum(row["known"] and row["correct"] for row in accepted)
        wrong = sum(row["known"] and not row["correct"] for row in accepted)
        false = sum(not row["known"] for row in accepted)
        if false / unknown <= max_unknown_fpir:
            choices.append((correct, -wrong, -false, -threshold, threshold))
    if not choices:
        raise ValueError("No threshold satisfies validation constraint; use a stricter margin")
    best = max(choices)
    return dict(
        threshold=best[-1],
        margin=margin,
        known_correct=best[0],
        known_wrong=-best[1],
        unknown_false_accept=-best[2],
        known=known,
        unknown=unknown,
        max_unknown_fpir=max_unknown_fpir,
        calibrated_probability=False,
        scope="single-observation distance threshold; validate temporal/event behavior separately",
    )


def run(manifest_path: Path, output: Path, *, calibrate=False, max_unknown_fpir=0.05):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("split") not in (
        "validation",
        "test",
        "smoke",
    ):
        raise ValueError("Expected schema_version 1 and split validation/test/smoke")
    if calibrate and manifest["split"] != "validation":
        raise ValueError("Calibration requires validation split")
    policy = Policy(**manifest.get("policy", {}))
    detector, gallery, enabled = HaarDetector(), {}, set()
    files = {}

    def resolve(relative):
        path = (manifest_path.parent / relative).resolve()
        files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    for person in manifest["enrollment"]:
        if not person.get("authorized") or not person["images"] or person["person_id"] in gallery:
            raise ValueError("Enrollment requires authorization, samples and unique person IDs")
        samples = []
        for name in person["images"]:
            frame = decode_image(resolve(name).read_bytes())
            boxes = detector.detect(frame)
            if len(boxes) != 1:
                raise ValueError(f"Enrollment {name}: expected exactly one detected face")
            gray, reason = prepare_face(frame, boxes[0])
            if gray is None:
                raise ValueError(f"Enrollment {name}: {reason}")
            samples.append(describe(gray))
        gallery[person["person_id"]] = samples
        if person.get("alert_enabled", False):
            enabled.add(person["person_id"])
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    report = dict(
        status="running",
        split=manifest["split"],
        model="classical-lbph",
        descriptor_space=SPACE,
        detector_hash=detector.model_hash,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        policy=policy.model_dump(),
        calibrated_probability=False,
        metrics_scope="annotated frames only; absent annotations are not negative labels",
        videos=[],
        conditions={},
        input_hashes=files,
    )
    totals, conditions, durations, calibration_rows = Counter(), defaultdict(Counter), [], []
    seen_videos = set()
    started = time.perf_counter()
    try:
        with (output / "predictions.jsonl").open("w") as predictions_file:
            for video in manifest["videos"]:
                path = resolve(video["path"])
                digest = files[video["path"]]
                if digest in seen_videos:
                    raise ValueError("Duplicate video bytes would inflate independent exposure")
                seen_videos.add(digest)
                pipeline = Pipeline(gallery, enabled, policy, detector)
                annotations = {int(a["frame"]): a for a in video.get("annotations", [])}
                if len(annotations) != len(video.get("annotations", [])):
                    raise ValueError("Duplicate frame annotations")
                capture = cv2.VideoCapture(str(path))
                count, events, last_event = 0, [], {}
                try:
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    if not capture.isOpened() or not 1 <= fps <= 240:
                        raise ValueError("Video must open with valid constant-frame-rate FPS")
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        if frame.shape[0] * frame.shape[1] > 8_500_000:
                            raise ValueError("Video frame exceeds 8.5 megapixels")
                        timestamp = count / fps
                        before = time.perf_counter()
                        result = pipeline.process(frame, timestamp)
                        durations.append((time.perf_counter() - before) * 1000)
                        for event in result["events"]:
                            person = event["person_id"]
                            if timestamp - last_event.get(person, -float("inf")) >= policy.cooldown:
                                events.append({**event, "time": timestamp})
                                last_event[person] = timestamp
                        predictions_file.write(
                            json.dumps(
                                dict(
                                    video=video["path"],
                                    frame=count,
                                    time=timestamp,
                                    tracks=result["tracks"],
                                )
                            )
                            + "\n"
                        )
                        annotation = annotations.get(count)
                        if annotation is not None:
                            truth = annotation["faces"]
                            metrics = score_frame(result["tracks"], truth, set(gallery))
                            condition = annotation.get(
                                "condition", video.get("condition", "unspecified")
                            )
                            totals.update(metrics)
                            conditions[condition].update(metrics)
                            if calibrate:
                                observations = []
                                for box in detector.detect(frame):
                                    gray, _ = prepare_face(frame, box)
                                    if gray is not None:
                                        observations.append(
                                            dict(bbox=box, descriptor=describe(gray))
                                        )
                                for index, observation in assignments(observations, truth).items():
                                    candidate = match(observation["descriptor"], gallery, 1, 0)
                                    if candidate["distance"] is not None:
                                        calibration_rows.append(
                                            dict(
                                                distance=candidate["distance"],
                                                margin=candidate["margin"],
                                                known=truth[index]["person_id"] in gallery,
                                                correct=candidate["person_id"]
                                                == truth[index]["person_id"],
                                            )
                                        )
                        count += 1
                    if count == 0 or any(i < 0 or i >= count for i in annotations):
                        raise ValueError("Empty video or annotation frame out of range")
                    report["videos"].append(
                        dict(
                            path=video["path"],
                            frames=count,
                            fps=fps,
                            seconds=count / fps,
                            annotated_frames=len(annotations),
                            events=events,
                            event_metrics=score_events(events, video["encounters"])
                            if "encounters" in video
                            else None,
                        )
                    )
                finally:
                    capture.release()
        report.update(
            status="passed",
            metrics=dict(totals),
            conditions={key: dict(value) for key, value in conditions.items()},
            processing_ms=dict(
                median=float(np.median(durations)), p95=float(np.percentile(durations, 95))
            ),
        )
        if calibrate:
            report["calibration"] = select_threshold(
                calibration_rows,
                split=manifest["split"],
                max_unknown_fpir=max_unknown_fpir,
                margin=policy.margin,
            )
        report["unique_video_hours"] = sum(v["seconds"] for v in report["videos"]) / 3600
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--max-unknown-fpir", type=float, default=0.05)
    args = parser.parse_args()
    report = run(
        args.manifest, args.output, calibrate=args.calibrate, max_unknown_fpir=args.max_unknown_fpir
    )
    print(json.dumps({k: report[k] for k in ("status", "metrics", "processing_ms")}, indent=2))


if __name__ == "__main__":
    main()
