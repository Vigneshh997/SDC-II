# Journal readiness assessment

31 October 2026. **Not ready for empirical diagnostic-method submission.** The software exists and has prior automated/GPU smoke evidence, but no current genuine paired training/evaluation cohort or compatible trained diagnostic checkpoint. This stage produced source-backed acquisition, linkage, experiment and reproduction plans and an additive readiness checker. It did not acquire restricted data, train a diagnostic model, submit a manuscript or contact a journal.

## Evidence gates

| Area | Current evidence | Required before an empirical performance submission |
|---|---|---|
| Scientific question | Proposed tri-modal architecture; substantial prior-art overlap | A narrow, prespecified contribution beyond combining pretrained components; see novelty assessment |
| Permitted data access | User confirms PhysioNet/Stanford not approved | Actual authorization, version/license receipt and applicable institutional data-governance determination |
| Patient/exam/time alignment | Loader checks source declarations; no restricted source audit | Measured eligible cohort, timestamp rules, ambiguity/exclusion counts and manual review; no subject-only joins |
| Target validity | NIH-14 head; no directly compatible paired cohort | Independent complete targets or versioned reviewed shared-task/partial-label migration |
| Genuine training | Historical scores excluded due to label-derived features; fixtures only | Reproducible run records and compatible checkpoints trained on observed paired data |
| Comparative performance | None established | Matched unimodal/bimodal/simple-fusion baselines, learned ablations, prespecified selection and paired uncertainty |
| Generalization | No external-domain diagnostic evaluation | Independent source/task mapping and locked external evaluation, or explicitly limit claims to internal validation |
| Calibration / reliability | Brier/ECE software implemented | Measured held-out calibration with denominators and intervals; fitted calibrator only if validation data support it |
| Explainability | LayerCAM, conditional metadata SHAP, text gradients and attention links execute | Independent localization and sensitivity/fidelity tests; acknowledge noncausality and background dependence |
| Reproduction | DAViT source now inspected, actual reproduction not performed | Source/weight/data provenance and code-faithful versus corrected evaluation track, or clear limitation |
| Reproducibility | Existing tests and local setup; no empirical release | Task/split/cache/manifest hashes, run configs/seeds/environment, checkpoint/model card, aggregate results and exclusions |
| Manuscript integrity | Existing papers contain historical material, not audited anew in this stage | Trace every result/table/figure to a valid run; remove unsupported superiority/clinical claims; guide/team review |

## Study and reporting scope

Prepare a retrospective model-development/validation paper only after valid data and experiments. [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378) is relevant reporting guidance for prediction-model work; a completed reporting checklist does not validate the research. Cover intended use/population, eligibility, dates/settings, predictors/outcomes and their timing, missingness, sample-size rationale, partitions, model development, calibration, uncertainty, subgroup performance, limitations, access and code availability.

[DECIDE-AI](https://www.bmj.com/content/377/bmj-2022-070904) addresses early **live clinical** decision-support evaluation. It is not evidence that this offline prototype was tested clinically. Do not imply a prospective/live study, regulatory approval, improved patient outcomes or equivalence to a radiologist from retrospective classifier metrics.

An ethics/IRB determination must come from the applicable institution and data context, not an assumption that public deidentification automatically exempts this team. Dataset agreements and permitted derived-model/feature sharing must be reviewed before release. Avoid publishing raw notes, private linkage keys, restricted images or patient feature caches. This is a concrete source-data requirement, not a request to establish new infrastructure.

## Manuscript preparation after evidence exists

1. Rewrite the abstract around actual cohort/task and validated contribution. Label hypotheses, limitations and completed experiments separately. Treat DAViT publication numbers as literature references only.
2. Methods must specify prediction time, independently sourced labels, clinical-feature availability, one-exam-per-patient selection, encoder pretraining exposure, task migration and uncertainty policies. Include a cohort attrition flow and class/split denominator table.
3. Results should include the locked baseline/ablation matrix with actual seeds and paired intervals, calibration, supported subgroups, external results when available, explanation controls and measured resource use. A missing experiment is marked not performed, never filled from another paper or artificial fixture.
4. Discussion must address prior-art overlap, possible indication/referral confounding, weak report-derived labels, complete-case bias, lack of gold-standard localization, MIMIC pretraining overlap and hardware/budget limits. Do not equate attention distribution with causal evidence.
5. Archive run manifests, configurations, code revision and permissible artifacts under a stable release. Have Dr. K Vishal Reddy and the project team review provenance, claims, authorship and venue scope. Venue-specific format/fees/policies should be checked only when a supported manuscript and target venue are selected.

## Actionable handoff

Immediate external action: complete dataset access with the provider; no credentials should be sent in chat. Then inspect metadata/labels/timestamps and decide between expensive complete NIH-14 annotation and the proposed versioned task migration. The seven deliverables form a preparatory package, not a completed clinical research study. Full training and empirical validation remain blocked; no expensive run or publication action has been started.

Deliverables: `DATASET_ACQUISITION_PLAN.md`, `DATA_LINKAGE_AUDIT.md`, `TRAINING_READINESS_REPORT.md`, `EXPERIMENT_PROTOCOL.md`, `DAVIT_BASELINE_PLAN.md`, `RESEARCH_NOVELTY_ASSESSMENT.md`, and this assessment. New software evidence is recorded under `audit/current_review/`; pre-existing audit/history files remain intact.
