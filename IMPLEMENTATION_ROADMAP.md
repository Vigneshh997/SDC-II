# Incremental implementation roadmap

1. Completed discovery: inspect existing source and both scientific references; preserve dirty checkout, datasets and checkpoints. Record baseline and separate existing v2 functionality from new work.
2. Immediate, data-independent improvements: held-out Brier score, per-label positive-event calibration error with reliability bins, regression tests; protect existing training artifacts from accidental fresh-run reuse. No architecture/checkpoint tensor changes.
3. Data prerequisite: authorized paired source, independently justified labels, prediction-time review, patient-disjoint splits and label-schema decision. This requires data access, not invented clinical variables.
4. Add explicit partial-label masks throughout ingestion/loss/metrics together, with a versioned manifest contract. Preserve existing binary manifests. Fit any imputation/normalization only on training patients.
5. Add explicit experiment modes for image-only/no-CNN/no-contrastive/concatenation comparisons. Preserve final model behavior by default; tag checkpoints with experiment identity. Start with inexpensive frozen encoders and small training smoke tests.
6. Controlled learning: compare identical patient splits, seeds, budgets, validation-only thresholds and optional CLAHE. Never use test results for selection. Full expensive runs need user confirmation.
7. Evaluate held-out discrimination, calibration, subgroup uncertainty and explanation sensitivity/localization. Estimate intervals at patient level. External-domain claims require a genuinely withheld source and reviewed label mappings.
8. Integrate the first genuine trained checkpoint with the existing API/client/report. Measure real end-to-end latency/memory and missing-input behavior. Consider Grad-CAM comparison, segmentation and attribution aggregation only when justified by data.

Current blocker: no reviewed paired manifest or compatible trained model. Software smoke tests can validate mechanics; stages 6–8 cannot honestly be completed as clinical research on the current inputs.
