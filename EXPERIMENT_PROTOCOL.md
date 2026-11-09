# Controlled experiment protocol (proposed; not executed)

31 October 2026. No valid paired training cohort is currently available. This protocol preserves the existing architecture; most variants below need explicit experiment implementations and tests before execution. A table entry is not a claim that a training flag exists.

## Primary question and cohort

Does pre-prediction text plus observed patient data improve multi-label CXR classification over strong image/text baselines, and does the current cross-attention/contrastive design add value under the same data and compute constraints?

Use the reviewed task and time boundary in `DATA_LINKAGE_AUDIT.md`. Freeze dataset release, eligible patients, selected study/view, task schema, label masks and split hashes before fitting. Current v2 requires complete NIH-14 labels and one study per patient. A future seven-finding or partial-label experiment needs the versioned migration in `TRAINING_READINESS_REPORT.md` first. Do not put MIMIC labels into the existing head by inventing negatives.

Keep official held-out patients where available and verify their actual disjointness. Otherwise use a fixed patient-level 70/15/15 train/validation/test allocation, balanced at patient level using a recorded method; freeze it once. A site-held-out experiment is additional and only possible when trustworthy site keys and comparable source tasks exist. Single-hospital random splits are not domain generalization. Do not compare models on different complete-case cohorts.

## Required comparisons

I=image, T=pre-prediction clinical text, M=observed structured attributes.

| ID | Inputs / change | Comparison purpose | Current software status |
|---|---|---|---|
| I0 | Historical image-only model if a valid executable artifact is located | Preserve genuine existing baseline | No independently validated active image-only checkpoint established; do not use old leaking tri-modal checkpoint as I0 |
| I1 | Frozen BiomedCLIP + current CNN + pooled classifier | Matched image-only baseline | Needs explicit experiment path |
| T1 | Frozen CXR-BERT + masked pooling + classifier | Text shortcut and clinical-context baseline | Needs experiment head |
| M1 | Same normalizer/MLP + classifier; logistic regression sensitivity baseline | Test age/sex/physiology signal and confounding | Needs experiment head |
| IT | Image + text | Incremental metadata benefit relative to ITM | Needs two-modality fusion adapter |
| IM | Image + metadata | Incremental text benefit | Needs two-modality fusion adapter |
| TM | Text + metadata | Test whether imaging adds information | Needs two-modality fusion adapter |
| ITM | Existing full tri-modal model | Reference architecture | Implemented; data-blocked |
| ITM-NCE0 | ITM, contrastive weight exactly zero | Isolate InfoNCE benefit | Current trainer rejects zero; requires explicit experiment mode, not validation weakening |
| ITM-CONCAT | Concatenate modality pools, project to matched classifier width | Isolate cross-attention | Requires experiment fusion module; do not call it cross-attention |
| ITM-CNN0 | Native spatial patches → trained linear projection; omit CNN | Isolate spatial refinement | Needs experiment adapter; do not substitute an untrained/mismatched checkpoint |

Learn a separate model for each variant; zeroing inputs only at inference is an occlusion study, not a retrained unimodal baseline. Pairwise/unimodal models must not receive absent-modality padding embeddings that disclose labels or cohort membership. Use the same present-modality encoders/initialization and label supervision; record parameter counts because equivalent compute and equivalent capacity cannot both be assumed.

## Budgets and selection

- Feasibility pass: 1–2 epochs on a small **genuine** training subset, no test access; check finite forward/backward/loss, modality masks, checkpoint restoration and memory. Any artificial fixture remains software-only evidence.
- Pilot budget: frozen encoders, 224-pixel inputs, 256-token limit, shared dimension 512, 2 attention layers/8 heads for the reference, batch 2 initially (4 only if measured safe), up to 12 epochs, AdamW lr 1e-4/weight decay 1e-5, early stopping patience 5, AMP/clipping as existing configuration. Prespecify seed 42 for feasibility, then 42/43/44 for pilot replication when affordable. Reserve five seeds for a final confirmation if feasible; do not claim statistical power from seed count alone.
- Allocate the same maximum patient presentations, optimizer updates, validation schedule and tuning-trial count per model; report GPU-hours as well. Use a shared small validation-only search (for example three prespecified learning rates) or one fixed recipe for all. Keep the comparison definition when an architecture consumes a different budget; disclose all differences.
- Select checkpoints by validation ASL, as the current trainer does. For cross-loss comparisons use a common prespecified validation criterion. Per-class F1 thresholds are validation-only; lock them before test evaluation. A calibration transformation, if added, also fits validation data only. Current code measures calibration but does not fit it.
- Primary pilot comparison: ITM versus I1 and IT. Expand to all ten comparisons only after the cohort/schema is valid and pilot memory/time are measured. Do not launch the whole grid on this laptop without approval.

## Analysis and uncertainty

Primary discrimination endpoint: macro AUROC over prespecified evaluable targets; co-report macro average precision to expose class imbalance. Report per-label positive/negative/unknown denominators, AUROC/AP, sensitivity, specificity, precision, F1 and confusion counts, plus micro summaries. Record undefined metrics as null, not 0/1. Report Brier, positive-event ECE and reliability diagrams using the existing definitions, with bin counts and prevalence; ECE is not clinical confidence.

Use 2,000 paired patient-bootstrap resamples of held-out predictions for differences and 95% intervals; this is a proposed analysis, not yet implemented here. Apply multiplicity correction to the prespecified primary family or explicitly identify exploratory comparisons. Report paired seed variability separately from patient uncertainty. If a bootstrap sample lacks a class, record invalid replicates and valid denominators rather than silently replacing an undefined AUC. No test-driven threshold/architecture selection.

Assess sex/age and available site/view subgroups with adequate support, reporting missingness and selection attrition. For missing-modality robustness, evaluate safe rejection now; learned missing-modality performance requires separately trained/validated models. Image blur/contrast robustness should be a prespecified perturbation study, not labeled unseen-hospital performance.

## Mechanism and explanation studies

- Compare InfoNCE against no-NCE with identical batches/negatives. Gradient accumulation does not increase in-batch negative count; age/sex-identical patients can be false negatives. Log modality gradient norms and sensitivity rather than asserting alignment prevents dominance.
- Compare CLAHE enabled/disabled within training and inference consistently; keep the current default until this experiment is measured. Cached frozen features fix augmentation: changing augmentation requires a new keyed cache and a separately documented experiment.
- Explain a stratified, fixed held-out subset. Compare LayerCAM with Grad-CAM and available independent localization annotations; include deletion/insertion, parameter randomization and sensitivity checks with limitations. Report text and metadata perturbation effects; preserve meaningful negation. Treat attention links as descriptive weights.
- Evaluate an external institution only after independently reviewing label and timing compatibility. Pretrained CXR-BERT has MIMIC exposure; this restricts claims of fully unseen data even with patient-disjoint downstream partitions. Measure and disclose known overlap rather than claiming none.

Each run needs a new output directory, experiment/task/schema ID, source/split/cache hashes, encoder revisions, seeds, versions, hyperparameters, time/memory, exclusions and raw aggregate metrics. No diagnostic checkpoint should be accepted by serving until task compatibility and genuine training provenance are verified.

## Resource estimates and optimization

Hardware: RTX 4060 Laptop GPU, 8,188 MiB; 16 GB system RAM supplied by the user. Current OS RAM query was denied, so 16 GB is not independently remeasured. Previous stage's actual cached-encoder fixture check peaked at 1,327,502,848 allocated CUDA bytes (~1.24 GiB) for batch 2, with frozen encoders and one disposable update. It is a feasibility floor, not measured full-epoch peak or reserved/driver memory. Evidence: `audit/current_review/encoder_verification.json`.

For P trainable FP32 parameters, weights+gradients+two Adam moment tensors need roughly 16P bytes before activations/workspaces. The current CNN/fusion/head adapters are roughly 26 million parameters by layer arithmetic (~0.39 GiB for those tensors), plus frozen backbone weights, activations and CUDA overhead. This is an analytic estimate, not a new profiled run. Start batch 2, num_workers=0; measure allocated AND reserved CUDA peaks and process RAM before increasing either. Keep several GiB GPU headroom for variable token lengths/explanations and avoid running explanations during training.

Cache **pre-CNN** visual patch tokens and pre-projection text tokens so current trainable CNN/projections remain learnable. At 196x768 image +256x768 text values in FP16, the raw upper-bound token payload is 694,272 bytes/study: about 1.29 GiB for 2,000 studies and 12.93 GiB for 20,000, excluding IDs/masks/indexing. This is a shape-based estimate; pad only per batch and memory-map/stream shards. Cache key must include image/note/source hashes, preprocessing/CLAHE, augmentation choice, encoder revision, tokenizer/truncation and split provenance. Such caching is proposed, not implemented. Never cache clinical features into tracked artifacts.

AMP and frozen encoders already exist. Gradient accumulation and a feature-cache training adapter are not implemented: add/test only when pilot profiling justifies them. Accumulation changes effective supervised batch, not InfoNCE's within-step denominator. Feature queues/large negative pools would alter the research method and need separate controls. Defer end-to-end unfreezing, LoRA, quantization, ONNX and ensembles until necessary. Full expensive runs remain approval-gated.
