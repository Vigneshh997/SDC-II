# Project audit — 31 October 2026

## Evidence and scope

Inspected backend models/losses, paired data validation/transforms, training/evaluation/checkpoint code, inference/explanations/reporting, API, frontend source/configuration, tests, dependency declarations, existing logs, inventory and checkpoint evidence. Inspected the 12-page DAViT paper through its full extracted text and the 12-slide presentation, including the embedded diagrams on slides 11 and 12. Extracted diagram copies are in `audit/current_review/`; original documents were unchanged. Runtime validation is tracked in `TEST_REPORT.md`.

## Working and partial behavior

The architecture and strict data/API contracts are implemented, with existing fixture and pretrained GPU verification evidence. A working readiness/error interface must not be mistaken for working diagnostic inference: genuine paired training and a compatible trained checkpoint are absent. Existing reports show historical tri-modal target leakage; this session does not restore that unsafe training path or overwrite the associated artifacts.

## Gaps against intended X-Fusion

- nnU-Net is absent; keep it optional until mask weights and evaluation exist.
- No missing-modality prediction or missing numerical imputation. Rejection is safe but limits applicability.
- Only complete binary labels are supported; uncertainty is rejected, never silently made negative.
- Fourteen NIH outputs are fixed. Presentation examples such as COVID-19 and “Lung” are not supported labels; Normal is not an independent output.
- Conditional patient-specific metadata SHAP exists; global/all-modality SHAP and validated cross-modal saliency do not.
- No calibration fitting, OOD detector, cohort evaluation or external-domain experiment. Calibration measurement is the first incremental addition in this session.
- No controlled learned fusion/loss/CNN/CLAHE ablations or evidence of reduced modality dominance. Current positive contrastive-weight restriction prevents a no-contrastive experiment; a future explicit experimental-mode contract is needed.
- Config enables CLAHE although its benefit has not been measured. Preserve the current configuration for compatibility; evaluate both settings before selecting a final model.

## DAViT differences

BiomedCLIP is not DINOv2-large; two 3x3/GELU layers are not DAViT's three-convolution BN/ReLU bottleneck; medical pretraining is not a reproduced CXR8 pathology/benign adaptation stage. The binary detection and viral/bacterial tasks and Grad-CAM baseline are not active v2 tasks. No faithful DAViT result was reproduced. See `BASE_PAPER_REPRODUCTION.md`.

## Priority and feasibility

The highest research value comes from independently labeled, correctly timed patient-paired data, followed by image-only baselines and controlled fusion/contrastive ablations on identical splits. With the existing image dataset, an independently implemented image-only study is feasible after lineage/label review, but is not tri-modal evidence. Frozen encoders, small batches, AMP, cached features and streamed loaders are plausible on RTX 4060/16 GB RAM. Full DINOv2-large adaptation and end-to-end tri-modal fine-tuning need a measured memory budget; the paper used two RTX 3090 GPUs.

Fresh runs currently reuse `best.pt`/`last.pt` without protection. Add a pre-encoder output-directory guard. Existing checkpoint files must remain unchanged. Do not launch expensive training or download large data to bypass the paired-data blocker.
