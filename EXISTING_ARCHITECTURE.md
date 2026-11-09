# Existing architecture

See `audit/ARCHITECTURE.md` for the detailed pre-existing v2 architecture.

The active path is `backend.main` → checkpoint loader → `InferenceService` → `MultimodalModel` → explanation aggregator → JSON report. The Next.js client calls `/health`, `/schema`, and `/predict`.

BiomedCLIP spatial patches → two 3x3 CNN/GELU layers → image projection. CXR-BERT tokens → text projection, with attention masking. Observed structured fields → checkpointed training-only normalization → two-layer MLP → metadata token. Learned modality embeddings and two default cross-attention blocks operate at dimension 512 with 8 heads. Each modality queries the other two, followed by residual normalization and feed-forward layers. Masked means are averaged and classified into 14 independent disease logits.

The supervised ASL objective is summed per class and averaged over examinations. Contrastive alignment is the mean of six directional cross-entropies between L2-normalized image/text/metadata embeddings, with temperature 0.07 by default. Total loss is ASL + 0.1 InfoNCE by default. Frozen encoders stay in evaluation mode. Training uses AMP on CUDA, gradient clipping, AdamW, cosine scheduling and validation-only selection/thresholds; testing is separate.

The paired loader checks source declarations, exact binary label schema, patient/study uniqueness, image integrity/content duplicates and explicit train/val/test partitions. Source declarations still need substantive human review. One examination per patient is required. Missing modalities are rejected, rather than imputed. Legacy training entry points fail explicitly because the older workflow leaked labels.

LayerCAM targets CNN spatial activations; KernelSHAP conditions on fixed image/text with an observed training background; text uses gradient-times-input; attention links describe model attention and do not establish faithful causal explanations. The exported report includes limitations and a research-only disclaimer.
