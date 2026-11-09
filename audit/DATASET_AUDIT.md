# Dataset audit, 22 October 2026

**Follow-up:** an open MIMIC-IV-ED demo was subsequently downloaded and its real clinical components were exercised on the GPU. It supplies no matching X-rays or compatible diagnostic labels. See [the dataset discovery/use report](DATASET_SEARCH_AND_USE.md). The NIH inventory below and the full tri-modal training blocker remain unchanged.

**Genuine paired tri-modal dataset readiness: false.** The user confirmed that no such dataset is available. Affected training, held-out evaluation and genuine-examination inference were not run.

`dataset_inventory.json` is the machine-readable result of the read-only audit, generated with `python -m backend.scripts.audit_dataset`. The original datasets and checkpoint were preserved.

| Check | Observed result |
|---|---|
| NIH-format CSV rows | 112,120 |
| Local image files | 31,877 |
| Images matching CSV | 31,630 |
| CSV rows without local images | 80,490 |
| Unmatched images | 247 |
| Matched patient identifiers | 12,180 |
| Duplicate CSV image IDs | 0 |
| Duplicate image hash groups | 10 |
| Excess duplicate image files | 237 |
| Invalid files in PIL format verification | 0 |
| Matched image dimensions | 31,630 at 1024×1024 |
| Remaining image dimensions | 247 at 224×224 |
| Ages outside 0–120 years | 3 |
| Official local train/validation images | 24,530 |
| Official local test images | 7,100 |
| Official image-list overlap | 0 |
| Local official patient overlap | 0 |
| Unassigned matched local images | 0 |
| Observed patient attributes found | Age and sex |
| Genuine clinical notes found | 0 |
| Genuine measured vitals found | 0 |

SHA-256 fingerprints are in `dataset_inventory.json`. PIL `verify()` checks file structure; it does not establish clinical image quality, diagnosis, de-identification or radiologist label accuracy. Format verification and content hashing covered every file in the local images directory. Exact duplicate grouping is content-based; perceptually similar images and cross-institution identifiers require further review.

The 247 unmatched 224×224 images and their duplicates are consistent with the historical proxy generator, which wrote repeated noise images into the dataset directory. This is an inference from file dimensions, duplication and preserved source code, not established download provenance. They were neither deleted nor used for training.

## Labels and imbalance

The local matched labels include only 65 Pneumonia and 120 Hernia positives, compared with 5,190 Infiltration, 4,483 Atelectasis, 4,286 Effusion and 6,163 No Finding records. Full counts are in the inventory. This subset is selected and markedly imbalanced. Labels are weak annotations from the CSV; no radiologist adjudication was performed. Missing unused `Unnamed: 11` values are reported separately in the inventory.

## Root cause of invalid historical tri-modal experiments

The original `backend/data/dataset.py` generated clinical reports directly from target disease labels, including invented laterality. It also sampled temperature, pulse, respiratory rate and oxygen saturation from label-conditioned distributions. Those values were not observed patient measurements. Patient-disjoint splitting did not remove that target dependency. Historical evaluation files under `outputs/xfusion` are therefore excluded from valid tri-modal diagnostic claims.

The CSV and download scripts provide NIH-format provenance hints, but local download receipts, permitted-use review and complete data lineage were not supplied. This audit does not certify a license or identity of every local file. The dataset has no examination-level notes/vitals to match. Image-to-patient mapping follows CSV identifiers only.

## Replacement ingestion contract

The paired JSONL loader verifies all required fields, exactly 14 binary labels, valid observed metadata, readable images, unique study/patient/image identifiers, explicit nonempty patient-disjoint train/val/test partitions, content duplicates and matching patient/study identifiers in each modality's source declarations. One examination per patient is currently required. Training preflight runs before expensive encoder initialization. The normalizer is fitted exclusively on training patients. Thresholds are chosen on validation only; test evaluation is a separate command.

Provenance declarations cannot prove that notes were truly observed or labels independently obtained. A human review must inspect source references, measurement timing, linkage quality, permitted use, de-identification, label provenance and report-derived leakage. The training command requires a nonempty review file and fingerprints both it and the manifest.

## Legitimate route to paired data

[MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/) provides images, reports and patient/study identifiers under credentialed access. [Official MIMIC linkage guidance](https://mimic.mit.edu/docs/faq/data.html) explains linkage with MIMIC-IV using subject identifiers. Access requires the applicable agreements and credentials. No restricted data were downloaded or access agreements accepted in this work.

For diagnostic prediction, full radiology reports that produced disease labels cannot also serve as independent input evidence. An approved study would need appropriately timed clinical context, such as an indication or pre-imaging clinical record, and independently justified labels. Time-varying measurements require study-time linkage, not a subject-only join. MIMIC's label schema differs from NIH's 14 labels and requires a reviewed task definition. CXR-BERT pretraining used MIMIC data; pretraining overlap must be assessed separately from downstream split separation.

The remaining prerequisite is an authorized, de-identified, genuinely paired source with a documented prediction time, compatible independently justified targets, source-level provenance and a human-reviewed manifest. Generating more notes or vitals from labels is prohibited.
