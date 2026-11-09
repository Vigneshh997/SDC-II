# Evaluation status — 31 October 2026

No new held-out diagnostic evaluation was performed. Default training preflight fails with `DatasetReadinessError` because `backend/data/paired/manifest.jsonl` is absent. No published DAViT score or historical label-leaking X-Fusion score is treated as a current result.

The existing `python -m backend.pipelines.eval` command now automatically includes per-disease Brier scores, positive-event ECE and ten equal-width reliability bins, plus macro Brier/ECE. Existing discrimination metrics and their names remain unchanged. Evaluation still checks the manifest fingerprint and uses the stored validation-selected thresholds. The new metrics neither tune thresholds nor fit any calibration transformation on the test data.

For class c, Brier = mean_i (p_ic - y_ic)^2. ECE = sum_b n_b/N * abs(mean_b(p_c) - mean_b(y_c)). Bins are [b/10, (b+1)/10), with p=1 included in the last bin. Empty bins expose count zero and null means. Macro values average the fourteen disease metrics equally. Positive-event ECE describes disease probability versus empirical prevalence, not confidence in the thresholded decision. It is sensitive to binning/sample size and may hide subgroup errors. Report class prevalence and reliability diagrams alongside it in actual studies.

Known fixture cases verify exact endpoints, empty bins, unequal bin weighting, constant 0.5 scores with zero ECE but Brier 0.25, maximally wrong scores, and unchanged perfect-discrimination metrics. These establish software correctness only. Calibration fitting, patient-bootstrap intervals, subgroup validation and external-domain studies remain pending genuine data.

Prior dataset, training and evaluation evidence remains in `audit/DATASET_AUDIT.md`, `audit/TRAINING_REPORT.md`, and `audit/EVALUATION_REPORT.md`. No historical files were overwritten by this session.
