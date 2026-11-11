> Historical record archived on 11 November 2026. Claims and instructions below
> describe their original milestone. Use [current status](../../PROJECT_STATUS.md) for active scope
> and progress; the current project horizon is approximately eight weeks.

# Classical odin baseline

The user selected fully classical Haar detection plus our own NumPy LBP/histogram
recognition and authorized implementation on 31 October 2026. This document records
the agreed scope and implementation choices; no further scope approval is needed.

## Purpose and approaches

Build the first complete local odin application and an inspectable classical
recognition baseline. Reuse its enrollment, tracking, events, interface and replay
evaluation when integrating ArcFace/FaceNet later. Alternatives considered were
SCRFD + LBPH (better controlled recognition comparison, less fully classical) and
fully classical Haar + LBPH (selected for learning). Fusion remains optional.

## Behavior

- One local operator, one webcam or uploaded video at a time. File replay is labeled.
- Haar detector uses a pinned pretrained frontal-face XML. LBP extraction, spatial
  histograms and chi-square matching are implemented in NumPy, not cv2.face.
- Grayscale equalized 128px crops; 8-neighbor radius-one LBP, 8x8 regional grids,
  normalized histograms. Version the descriptor/preprocessing together.
- Uploaded enrollment images must each contain exactly one usable face; all-or-none
  enrollment. Store encrypted descriptors, not original face images. Names are
  pseudonyms, authorization must be affirmed, alerts default off.
- Nearest per-person template distance, second distinct-person margin, validation-
  configurable rejection threshold. Display distance (lower is better), never a
  probability. Defaults are explicitly uncalibrated engineering settings.
- Greedy spatial IoU association with ambiguity reset, bounded observations,
  time-spaced voting, evidence expiry and conservative contradiction reset.
  This is a basic tracker, not a promised solution to crossing/occlusion.
- One persisted event per confirmed appearance, enabled identities only, with
  source/person cooldown; new session resets temporal evidence. Unknown and
  insufficient never create identity alerts. Event acknowledgement is persisted.
- SQLite metadata, encrypted template blobs, restricted local key/data permissions,
  configurable seven-day event retention, immediate person deletion including
  templates, events and worker evidence. No saved unknown crops or full live video.
- Local React monitor/gallery/events/evaluation workflow. Browser video uploads are
  temporarily stored privately and removed after replay. Latest JPEG preview only;
  do not persist preview images. Source errors/EOF/stopped state visible.
- Separate capture helper with latest-frame slot in live mode; offline CLI processes
  all frames. Bounded track count, upload size and decoded-image dimensions.
- Evaluation CLI accepts explicit participant/condition manifests, outputs per-frame
  predictions, metrics with denominators, config/input hashes and timing; a grid
  threshold calibration command uses labeled validation descriptors. No invented
  participant accuracy or robustness claims without recordings.

## Boundaries and compatibility

Use a matching OpenCV 5 contrib wheel, exclude transitive opencv-python to prevent
namespace overlap. Preserve all other existing locked versions. Pin/check Haar XML
and its upstream license. Existing model probes must report contrib metadata and
still execute real models on CPU; GPU verification requires accessible hardware.
CPU/no-camera host limitations must be reported, not treated as passing hardware
tests. No model downloads at runtime, no neural inference in this baseline.

Run one FastAPI worker. Heavy capture/inference lives in a service thread outside
HTTP handlers. Serialize enrollment/deletion/session changes with a service lock;
capture may block in its daemon helper, but stop invalidates the session generation
so stale work cannot publish. Untrusted network hosting, multi-camera operation,
event video recording and fusion are outside this increment. Exact local Origin/
Host checks and mutation token protect browser mutation requests. Encryption keys
are local files with OS permissions, not a defense against the same OS user.

## Verification

Hand-derived LBP fixtures; descriptor invariance/distance/rejection; persistence,
retention/deletion; tracker identity reset and dedup; API workflow; generated replay
with real Haar detector; existing model CPU inference; existing unit suite; frontend
lint/build; real HTTP smoke when host networking permits. A pre-existing AnyIO
TestClient portal stall is known and must be distinguished from application errors.
