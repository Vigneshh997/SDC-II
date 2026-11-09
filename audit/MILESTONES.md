# Implementation plan and milestone record

Dependency order: preserve baseline and inspect all three inputs; audit data; repair independent model components; verify losses and optimization; implement evidence; integrate the application; run checks; document exact release limits. Dependent genuine training/evaluation stopped when data readiness failed.

| Milestone | Changes and files | Executed evidence | Status and remaining issue |
|---|---|---|---|
| 1 Baseline and stabilization | `baseline_source.zip`, hashes, Git HEAD/status/binary diff; inspected PPT text plus embedded diagram and full extracted DAViT text | Recovery archive created; safe old checkpoint inspection; API startup | Baseline preserved. Existing paper edits/deletions untouched. No new Git commit/branch required for recovery archive. |
| 2 Dataset readiness | Read-only `backend/scripts/audit_dataset.py`, strict manifest loader and validation | All 31,877 image files hashed/verified; patient overlap and label counts recorded | BLOCKED: no genuine notes or vitals, no paired manifest or provenance review. |
| 3 Encoders | `models/multimodal.py`, preprocessing and metadata validation | Authentic cached weights loaded on CPU/GPU, shapes and gradients verified | Component software verified. Representation suitability and clinical training unverified. |
| 4 Cross-modal fusion | Three receiving-query attention operations per block, masks/IDs/residual/FFN | Q/K/V construction test, padding invariance, all-modality sensitivity and gradient tests, actual GPU execution | Component software verified. Clinical utility unverified. |
| 5 Training pipeline | `pipelines/train.py`, losses, safe checkpoint contract and config | Actual loop on temporary fixtures, optimization/checkpoint/resume tests; authentic disposable optimization step | Software verified; genuine training BLOCKED. Hyperparameters not validated. |
| 6 Explainability | `explainability.py` LayerCAM, signed SHAP, embedding-gradient text evidence, attention links | Positive spatial gradient utility, geometry, SHAP additivity, text checks and authentic GPU explanations | Software verified; localization/fidelity validation, all-modality SHAP and formal cross-modal saliency outstanding. |
| 7 Application | FastAPI readiness/schema/validation, `InferenceService`, JSON reports, Next.js UI | API negative tests, report integration on fixtures, frontend lint/type/build; real local readiness startup | Software connected. Genuine trained inference blocked. No fabricated success path. |
| 8 QA and release | Ruff, Bandit, dependency checks/audits, CI definition, reproducibility guides | See testing evidence files and test report | NOT READY. Data, training, held-out evaluation, missing segmentation and other scientific requirements remain. Remote CI not executed. |

A component passing tests is not counted as a completed PPT requirement under the requested end-to-end definition of done. The traceability report records this distinction explicitly. No code was changed to force a diagnostic score or bypass dataset readiness.
