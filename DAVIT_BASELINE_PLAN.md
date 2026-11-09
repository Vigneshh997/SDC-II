# Faithful DAViT baseline plan

31 October 2026. **Not reproduced.** This is a source-verified execution plan. Current X-Fusion remains BiomedCLIP-based, and no architecture or original checkpoint was replaced.

## New source evidence

The GitHub connector retrieved the official source successfully after prior-stage web retrieval failed. Inspected repository commit: [021509e7d546dbf01c0ee898edd6a18debaa9051](https://github.com/awsm-research/DAViT/commit/021509e7d546dbf01c0ee898edd6a18debaa9051), dated 2 July 2025. Pin that commit for a reproduction, not mutable main. Public source inspection is not execution or weight verification.

| File | Verified implementation facts |
|---|---|
| [model_davit.py](https://github.com/awsm-research/DAViT/blob/021509e7d546dbf01c0ee898edd6a18debaa9051/davit/model_davit.py) | Remove first special token, reshape patch sequence to square grid; 1024→2048 1x1, grouped 2048→2048 3x3 (32 groups), then 2048→2048 1x1; BN after each, one ReLU after the final BN; average pooling, 2048→1000→2 linear layers. CrossEntropyLoss during supervision; two-class softmax at inference. Last convolution does not reduce channels. |
| [main.py](https://github.com/awsm-research/DAViT/blob/021509e7d546dbf01c0ee898edd6a18debaa9051/davit/main.py) | Loads domain_adapted_davit.bin with shape-filtered non-strict state loading; runner hardcodes CUDA/two GPUs. Transforms are applied once while building in-memory examples. Validation minimizes loss despite checkpoint-best-f1 directory name. Clipping call is commented out. Test ROC-AUC uses argmax class predictions, not probabilities. |
| [detection script](https://github.com/awsm-research/DAViT/blob/021509e7d546dbf01c0ee898edd6a18debaa9051/davit/train_detection.sh) / [subtype script](https://github.com/awsm-research/DAViT/blob/021509e7d546dbf01c0ee898edd6a18debaa9051/davit/train_type.sh) | facebook/dinov2-large; 30 epochs, batch 32, lr 1e-5, seed 123456. Subtype script activates classify_pneumonia_type. |
| [data download](https://github.com/awsm-research/DAViT/blob/021509e7d546dbf01c0ee898edd6a18debaa9051/data/download_data.sh) / [weight download](https://github.com/awsm-research/DAViT/blob/021509e7d546dbf01c0ee898edd6a18debaa9051/davit/saved_models/checkpoint-best-f1/download_models.sh) | Google Drive gdown links for data archives and three artifacts; script contents do not establish actual artifact bytes, checksum, filenames or permitted redistribution. Downloads were not executed. gdown is not listed in the inspected requirements.txt. |

The runner maps bacterial=0, viral=1; detection uses pneumonia=1. This differs from the paper's subtype label convention and matters for class-specific precision/recall/F1. It also computes binary AUC from hard predictions, which equals balanced accuracy when both classes exist and does not measure score ranking. The paper's reported AUC should not be silently relabeled as a reproduced probability-based ROC-AUC.

Other discrepancies requiring resolution: paper describes per-training augmentation and clipping, while code precomputes stochastic transforms once and leaves clipping disabled. Source uses Resize to a tuple rather than an aspect-preserving short-side resize. The examined runner consumes an already adapted checkpoint; it does not reconstruct the original CXR8 adaptation procedure from scratch. Downloaded weights' training lineage and the adaptation split/recipe remain unverified.

## Two separate scientific tracks

**Code-faithful replication:** isolate upstream code/version/environment and task labels; retain original preprocessing/selection/metrics behavior. Audit the source and obtain authoritative data/weights and permitted-use evidence first. Record loaded/missing keys and hashes. Reproduce detection and subtype separately; do not load either into X-Fusion's NIH-14 head. Report this track's hard-prediction metric explicitly.

**Methodological validation:** on the same audited original split, compute ROC-AUC from positive-class probabilities and retain hard-prediction AUC under a distinct name for comparability. If clipping, lazy loading, augmentation refresh or selection logic is corrected, label that as a corrected implementation with a deviation table. Do not tune changes using original test outcomes. Check a single official checkpoint on CPU or small GPU batches before considering retraining.

Published outcomes remain reference-only: detection F1 96.95%, AUC 95.56%; subtype F1 80.95%, AUC 84.05%. The original downstream cohort is pediatric Kermany, not an adult multimodal NIH/MIMIC cohort. Those numbers cannot be compared directly against a different task/split. The local paper is indexed in `BASE_PAPER_REPRODUCTION.md`.

## Acquisition and compute gates

1. Verify repository licensing before redistributing/vendoring source; no local copy of upstream implementation was added. Resolve model artifact names, hashes and origin. Review Kermany/CXR8 access terms and local availability; no source count or class label is assumed from filename alone without an audit.
2. Preserve train 5,216 / validation 16 / test 624 for the original task if confirmed from authoritative archives. Verify patient identities/duplicates where available; if grouping cannot be established, disclose that and perform a separate grouped sensitivity study where feasible. The 16-image validation set creates substantial selection uncertainty.
3. Restore official adapted weights for evaluation first; verify shape and label orientation on a small batch. Original Grad-CAM notebook target layer/preprocessing still needs independent inspection before faithful CAM reproduction. Use gradients on the correct class, never an unrelated pooled feature map.
4. DINOv2-large is roughly 300M parameters. FP32 weights alone are ~1.1 GiB; full Adam weights/gradients/moments alone are ~4.5 GiB before activations and CNN state. Batch 32 end-to-end on an 8 GB laptop GPU is not a justified assumption. Small-batch frozen inference is the realistic first step. Full adaptation/fine-tuning needs profiling and explicit expensive-training approval; freezing the backbone is a resource variant, not faithful end-to-end reproduction.
5. Record original code track versus corrected track and retain all unsuccessful runs. Only claim reproduction once provenance, task/splits, loaded weights, numerical outputs and observed differences are documented. Until then, status remains planned and blocked on acquisition/verification.
