# Training readiness and versioned data migration

31 October 2026. **Blocked:** no genuine paired manifest/review and no compatible trained diagnostic checkpoint. The user confirmed dataset access remains unapproved. New read-only evidence is `audit/current_review/stage2_readiness.json`. Architecture, loader validation, checkpoint acceptance and inference readiness are unchanged.

## Exact current v2 preflight requirements

The authoritative code is `backend/data/dataset.py:load_manifest` and `backend/pipelines/train.py:preflight`.

- A nonempty UTF-8 JSONL manifest at configured `manifest`; each row has patient_id, study_id, image_path, clinical_note, metadata, labels, split, provenance.
- Nonempty patient/study IDs, globally unique patient and study; one examination per patient across all splits. Explicit nonempty train/val/test. Images must have unique resolved paths and byte hashes and pass PIL verification. Relative paths resolve against the manifest directory.
- Nonblank observed note, at most 20,000 characters; only whitespace cleanup. Pretrained text encoder later truncates at 256 tokens.
- Metadata keys must exactly match configured fields; default is age+sex. Age must be numeric, finite, within 0–120, not a boolean/missing value. Sex must be female/male/other/explicitly unknown; no invented default. Optional supported vitals need configuration and observed valid values.
- Labels must name **exactly** the 14 labels/order in `backend/config.py`, each an int/float 0 or 1. Uncertain, unobserved, null and boolean labels are rejected. No Finding means the all-zero vector only when all fourteen negatives are actually justified.
- Provenance observed=true and nonempty license; note_available_before_prediction=true and labels_derived_from_note=false. Each image/note/metadata source declaration must repeat the same patient_id/study_id and identify its source. These are assertions needing independent human verification, not substitutes for timing/label review.
- A nonempty file at provenance_review, batch_size >=2 and at least batch_size distinct training patients. With default batch size 4, literal structural minimum is 4 train +1 val +1 test =6 patients; this is **not** an adequate research sample. Metadata normalization fits only training rows.

To actually train, runtime dependencies/cached weights/device must work, epochs and contrastive weight must be positive, finite gradients/loss must hold, and a fresh output directory must not contain best.pt/last.pt/history.json. The current configuration uses frozen encoders and AMP on CUDA. Validation chooses thresholds and checkpoint; test evaluation is a separate command. The loader does not test class prevalence, timestamps, legal rights or pretraining overlap.

## Task incompatibility is a real blocker

MIMIC/Plus use CheXpert-family observations rather than the NIH target ontology. This correspondence is a **proposed reviewed task mapping**, not an implemented automatic conversion. [Observation semantics](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)

| Current NIH output | Candidate source correspondence | Decision |
|---|---|---|
| Atelectasis, Cardiomegaly, Consolidation, Edema, Pneumonia, Pneumothorax | Same-named observations | Review annotation definitions before shared-task use |
| Effusion | Pleural Effusion | Review and explicitly rename |
| Mass, Nodule | Lung Lesion | Broader source cannot resolve both NIH outputs; do not duplicate labels |
| Infiltration | Lung Opacity | Not equivalent; do not auto-map |
| Pleural_Thickening | Pleural Other | Not equivalent; do not auto-map |
| Emphysema, Fibrosis, Hernia | No corresponding independent source observation | Unobserved, not negative |
| No NIH counterpart | Enlarged Cardiomediastinum, Fracture, Support Devices, No Finding | Exclude from a disease-only shared task or define explicit new tasks |

## Two legitimate routes

**Retain v2:** obtain complete independent annotation for all NIH-14 targets on genuinely paired cases, including verified negatives for unsupported fields. This has real expert cost and may be impractical. Do not overwrite source labels or fill absent diseases with zeros just to make preflight pass.

**Recommended migration proposal (not implemented):** introduce a new manifest schema and architecture/checkpoint version for a reviewed shared seven-finding task or explicit CheXpert disease task. Preserve v2 loading/serving and historical artifacts. Migration must be a distinct command writing a new file with source fingerprints and a reviewed mapping, with no in-place conversion.

1. Define schema_version, task_id, ordered label_names and explicit four-state labels: positive/negative/uncertain/unobserved. Add prediction_time and source timing/linkage sidecars. Never overload v2 floats with sentinel values that existing ASL could misread.
2. Convert supervision to y and observed_mask only inside a new task-aware data adapter. For a conservative primary analysis mask uncertain and unobserved; uncertain-positive/negative policies may be separately prespecified sensitivity analyses. Do not interpret blank as a confirmed negative.
3. Mask ASL/BCE losses before reduction; normalize over observed entries with a documented scale. Reject batches with no supervised entries or handle unsupervised-only updates explicitly. Contrastive loss eligibility needs genuinely present paired modalities, independent of label observation.
4. Use each label's observed subset for thresholds/metrics/calibration; report denominators and null metrics without both classes. Exclude masked entries from all confusion counts and bootstrap calculations. Threshold fitting remains validation-only.
5. Parameterize classification dimension, checkpoint task/label order and API/frontend schema together; reject old/new task mismatches. Test binary backward compatibility, zero gradients at masked targets, unknown-versus-negative distinction, save/load roundtrip and unchanged v2 rejection behavior.
6. Missing numerical values require train-only imputation statistics plus explicit masks and a changed MLP input/checkpoint contract. Missing whole modalities require masks, fusion/sampler/loss changes and dedicated validation; current safe rejection remains until that work is complete. Median imputation is a recorded preprocessing operation, never an invented measured clinical value. Compare complete-case selection bias first.

## Minimum viable genuine research cohort

An initial 100–300 complete, independently reviewed examinations can test ingestion and runtime mechanics; it cannot establish diagnostic performance. A proposed learning pilot of 1,000–2,000 patients is feasible to budget, but final size must follow observed class support and target precision. For sensitivity near 0.8 with desired 95% half-width 0.08, the rough binomial calculation 1.96²×0.8×0.2/0.08² suggests ~97 positive test patients **per target**, before clustering/selection effects. Rare findings can require much larger cohorts. Freeze eligibility and patient splits before model selection; never cherry-pick test positives from observed outcomes to improve apparent performance.

Commands (read-only; no weights loaded/training started):

```powershell
.\.venv-gpu\Scripts\python.exe -m backend.scripts.check_training_readiness --config config.yaml
.\.venv-gpu\Scripts\python.exe -m backend.pipelines.train --config config.yaml --preflight-only
```

The new checker calls the existing preflight, returns blocked/exit 1 on structural failure, records source fingerprints and aggregate class counts when available, and reports review_required even after structural success. Optional `--output NEW_FILE.json` uses exclusive creation and never replaces a file. It does not certify clinical truth, enable predictions or change the current schema.
