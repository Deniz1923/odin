# Dataset sources and actor appearance study

## Decision and purpose

The study includes three complementary sources; actor media does not replace
controlled participant testing.

| Source | Purpose | Current status |
|---|---|---|
| Public research datasets: ChokePoint and ROF | Documented benchmark material for surveillance-like video and occlusion comparisons | Reserved for later; not downloaded |
| Team members and consenting friends | Collect our own photos/videos with known identities and controlled lighting, pose, distance, masks, crossings and unknown-person encounters; evaluate recognition, tracking, alerts and live operation | Included in scope; roster and recordings remain to be collected |
| Public-person media, especially actors | Compare recognition across independently sourced situations, ages, hairstyles and film makeup | Small actor starter collection acquired; expanded coverage and annotations pending |

Collecting our own data means recording and annotating actual scenarios. Synthetic
alterations may supplement those recordings but must be labelled separately.
For participant recordings, agree to collection and report use, keep independent
enrollment/validation/test sessions, and include people absent from the gallery.
Report each source separately so abundant actor frames do not hide failures on
controlled recordings or benchmarks.

Use externally sourced actor photographs and videos to study recognition across
appearance changes: age, hairstyle, facial hair, makeup, scars/prosthetics, pose and
lighting. These are real people too; being famous does not itself grant permission
to redistribute their photographs or films. Record the source and applicable terms
for each asset. Keep raw media under ignored `data/datasets/`, separate from the
application gallery/database in `data/classical/`.

This is an appearance-robustness study, not proof that a model understands identity
like a human or has never memorized a person. Pretrained celebrity recognition
models may have seen these identities or exact images. In particular, the selected
FaceNet checkpoint uses VGGFace2, a celebrity dataset with pose and age variation:
[official VGGFace2 description](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/).
Record training overlap as known, excluded or unverified for each checkpoint;
absence of exact duplicate files does not establish unseen identities.

## Sources reserved for later

| Source | Useful evidence | Limits and acquisition status |
|---|---|---|
| [ChokePoint](https://arma.sourceforge.net/chokepoint/), [author archive](https://zenodo.org/records/815657) | Surveillance-like portal sequences; several views and sessions; detection, tracking and identity experiments | Not downloaded. Inspect annotation format before promising full box/trajectory scoring. Use original sequences for pipeline evaluation, not only face crops. Research-use conditions and citation notice accompany the source. Keep simultaneous camera views of one passage in the same split. |
| [Real World Occluded Faces (ROF)](https://github.com/ekremerakin/RealWorldOccludedFaces) | Separate image recognition comparison for neutral faces, sunglasses and real masks | Not downloaded. Cropped still images cannot establish tracking or event performance. Repository licensing does not establish rights to every third-party photograph. Inspect acquisition code/data before execution and record provenance. |

Do not silently combine these datasets' percentages with the actor collection.
They answer different questions and have different capture conditions.

## Actor collection

Start with Johnny Depp, Gary Oldman, Tilda Swinton, Keanu Reeves and Henry Rollins.
This is a convenience pilot, not a representative population or a frozen test set.
Wikimedia Commons provides an initial source of individually attributed photographs
and some interview videos. A dated event portrait supports age/style variation;
it is not evidence of recognition under film makeup.

The initial local acquisition is indexed in
[`experiments/actor_media_inventory.json`](../experiments/actor_media_inventory.json).
It records source pages, download URLs, creator/license metadata, hashes and local
paths. Items remain unassigned to enrollment/validation/test until reviewed.
Downloaded event photos and interviews are a starter collection; actual distinct
film roles, scars and prosthetics still need suitable source material.

Initial acquisition: ten photos (two per actor) and one Henry Rollins interview
video, approximately 249 MB total, in `data/datasets/actor_pilot/`. Photo source dates
span 2003–2024. The interview is approximately 9 minutes 55 seconds at 1280×720
(container reports 60 FPS). Photo decoding and all local SHA-256/source SHA-1 hashes
were checked; the video container was probed. This does not establish constant
frame timing or complete-video decoding. Some photos contain additional people,
small faces, glasses or a downturned face; preserve those conditions during review
rather than assuming all ten are usable enrollment portraits.

For film-role coverage, target several independently sourced appearances per actor:

| Actor | Candidate appearances to source and review |
|---|---|
| Johnny Depp | Ordinary portrait; Jack Sparrow; Edward Scissorhands; older appearance |
| Gary Oldman | Ordinary portrait; Sirius Black; Winston Churchill prosthetics |
| Tilda Swinton | Ordinary portrait; White Witch; Ancient One; older appearance |
| Keanu Reeves | Younger short-haired appearance; Neo; John Wick with longer hair/beard |
| Henry Rollins | Earlier performance portrait; later interview; film appearance if obtainable |

These are acquisition targets, not claims that images/clips have been obtained or
that every transformation preserves enough visible facial information to recognize.
Prefer explicitly licensed source files, author-provided research media or available
film material whose use has been established. Preserve film/title/year, credited
actor/character, source timestamp, acquisition date and transformations. A trailer
being publicly viewable does not itself permit downloading or redistributing it.
Document unresolved usage terms rather than inventing a permission record.

## Experimental design

1. Review every asset visually: correct actor, visible faces, duplicates, date/event
   or film, makeup/style tags, resolution and usable face size. One file may contain
   multiple people; only the intended actor is enrolled. Inventory labels are not
   yet reviewed bounding-box ground truth.
2. Enroll from an ordinary appearance using a fixed template budget per identity.
   Keep evaluation appearances out of the gallery. Compare a second, explicitly
   labelled multi-appearance enrollment condition if studying gallery diversity.
3. Split by entire film, recording/event and underlying original image. Crops,
   resized copies, neighboring video frames, promotional images from one scene
   and views of the same recording stay together. Deduplicate across sources.
4. Reserve separate development/validation/test appearances. Choose thresholds and
   any fusion weights on validation, then freeze them before final-test runs.
   Keep test-unknown actors outside enrollment and validation-unknown identities.
   Add separate unknown actors; the initial five-person collection alone is not
   a complete open-set study.
5. Compare LBP and ArcFace first, then FaceNet/ensemble if implemented, on the same
   reviewed material and equal enrollment budgets with method-specific thresholds.
   Report ordinary and changed-appearance results separately, including wrong
   identities, rejections, detection misses and sample/identity counts. Plot each
   condition; do not select only spectacular successful examples for the report.
6. Separate face-crop recognition from full-frame pipeline performance. A missed
   face under makeup is a detection failure, even if a manually supplied crop can
   be recognized. Native video supports temporal/event evaluation; a slideshow of
   stills does not supply independent tracking or real-time evidence.

Useful comparisons are same-session versus held-out appearance, single-appearance
versus diverse enrollment, and classical versus learned descriptors. None alone
isolates memorization from recognition. Human recognition is motivation, not a
measured baseline unless a separate controlled human study is conducted.

Unit tests continue using deterministic small fixtures. Use reviewed dataset subsets
for integration checks and larger offline experiments outside routine pytest.
The existing runner consumes video manifests; the acquisition inventory is not a
runner manifest, and still-image scoring/import is not implemented by this document.
See [evaluation protocol](EVALUATION.md) for annotation, calibration and metrics.
