# X-Fusion implementation baseline — 31 October 2026

Read-only source discovery preceded this report and all source edits in this session. Existing uncommitted changes on `main`, including reorganized/deleted documents and modified backend/frontend files, belong to the user and were preserved. The starting implementation already contained the v2 corrections documented under `audit/`; those are not new contributions of this session.

| Component | Current status | Verified in source / prior evidence | Required improvement |
|---|---|---|---|
| Image processing | Implemented | Official encoder transform; optional CLAHE | Controlled CLAHE experiment; no validated segmentation |
| Image encoder | Implemented | Pinned BiomedCLIP visual trunk, spatial patch tokens | Genuine downstream training/evaluation |
| Text encoder | Implemented | Pinned CXR-BERT backbone/tokenizer, padding, 256-token limit | Paired pre-prediction notes and truncation analysis |
| Tabular encoder | Implemented, strict | Train-only fitted age normalization, categorical sex, MLP | Missing numeric masks/imputation and validated richer fields |
| CNN refinement | Implemented | Two spatial 3x3 convolutions and GELU | Compare against image-only and no-CNN variants |
| Cross-attention | Implemented | Three Q-versus-other-modality K/V operations per block | Controlled fusion comparisons and dominance measurement |
| Contrastive learning | Implemented | Six directional InfoNCE terms over three pairs | Repeated patient handling; false-negative analysis for similar metadata |
| Multi-label prediction | Implemented | 14 NIH labels, ASL, independent sigmoid, validation thresholds | Unknown/uncertain label masks and alternate schemas |
| Explainability | Implemented software | LayerCAM, conditional metadata KernelSHAP, text gradients, attention links | Localization/fidelity evaluation and population-level aggregation |
| Backend | Implemented | FastAPI health/schema/predict, provenance checkpoint contract | Genuine checkpoint and real-examination integration |
| Frontend | Implemented | Next.js upload/schema/readiness, probability/evidence view, JSON export | Genuine examination validation |

Source inspection establishes code presence, not diagnostic validity. Prior GPU fixture evidence is `audit/encoder_gpu_verification.json`; current tests are recorded separately in `TEST_REPORT.md` after execution. The app deliberately returns not-ready without a compatible trained checkpoint. No valid tri-modal diagnostic result is available.

Prior full dataset inventory: 31,877 local images, 31,630 matched NIH CSV images, 12,180 patients; zero paired notes/vitals. These counts are inherited from `audit/dataset_inventory.json`, not a newly repeated full content audit. The MIMIC ED demo has genuine clinical fields but no matching X-rays/compatible diagnostic targets. Legacy checkpoints and historical scores remain preserved and excluded from valid tri-modal claims because their training used label-derived text/clinical variables.
