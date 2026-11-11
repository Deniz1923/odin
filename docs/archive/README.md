# Historical documents

These records preserve earlier assumptions, decisions and observations. Instructions,
feature inventories, test counts and next steps describe their original milestones.
Use [project status](../PROJECT_STATUS.md) for active scope and the approximately
eight-week horizon, [architecture](../ARCHITECTURE.md) for implemented behavior,
[environment](../ENVIRONMENT.md) for reproduction, and [evaluation](../EVALUATION.md)
for the study protocol.

| Record | Why retained |
|---|---|
| [12 October specification](2026-10-12-project-specification.md) | Original course mapping/proposal; 12-week assumptions and expanded required scope were superseded |
| [Architecture through 31 October](2026-10-31-architecture-and-delivery-plan.md) | Larger proposed system, preferences and rationale; 10-week horizon is superseded and proposed components are not implementation evidence |
| [16 October model feasibility](2026-10-16-model-feasibility.md) | Original PyTorch/ONNX CUDA timings, preprocessing, strict-CUDA diagnosis and overlay limits |
| [Environment handoff through 31 October](2026-10-31-model-environment-handoff.md) | FaceNet export parity, shared ONNX CUDA/CPU measurements and environment cleanup |
| [Initial project review](2026-10-31-initial-project-review.md) | Pre-classical snapshot; missing-application statements were superseded |
| [Model feasibility plan](plans/2026-10-16-model-feasibility.md) | Completed preparation tasks and review corrections |
| [Shared ONNX plan](plans/2026-10-19-shared-onnx-runtime.md) | Completed export/runtime tasks and validation gates |
| [Classical design](specs/2026-10-31-classical-pipeline-design.md) | Scope and choices for the implemented increment |
| [Classical implementation plan](plans/2026-10-31-classical-pipeline.md) | Completed work, edge-case fixes and environment repair ledger |

## Interpreting historical results

On the recorded RTX 4070 Laptop GPU, final shared-runtime stage medians were about
8.79 ms for SCRFD, 3.46 ms for ArcFace and 11.03 ms for FaceNet. These are short,
instrumented batch-one model-stage measurements, excluding decoding, preparation,
tracking, matching, persistence and preview. They are not application FPS or current
hardware measurements. Older PyTorch and ONNX runs used differing settings and do
not establish an ONNX speed improvement.

The FaceNet export record reports eleven numerical cases, maximum component
difference `2.71946e-7` and maximum pairwise cosine difference `8.49366e-7`.
This supports conversion parity on those inputs, not identity discrimination or
mask robustness. Source/derived hashes and local export reports remain part of
the active model manifest's verification chain.

Relative Markdown links were adjusted when documents moved. Historical prose,
commands and code-block paths are retained as records; use the active environment
guide for current commands. Ignored model/run files referenced here are local
artifacts and will not appear in a fresh clone.
