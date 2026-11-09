# X-Fusion dataset acquisition plan

Assessment date: 31 October 2026. The user confirmed in this stage that neither PhysioNet nor Stanford access is approved. No restricted data, large datasets or paid services were acquired. Published cohort totals below are not counts of complete, temporally eligible tri-modal records. Ranking reflects suitability for the proposed prediction task, not model performance.

## Ranked sources

| Rank / source | Verified scope and modalities | Labels / structured fields | Acquisition and principal limitation |
|---|---|---|---|
| 1. MIMIC-CXR 2.1.0 + CXR-JPG 2.1.0 + MIMIC-IV-ED 2.2; selected MIMIC-IV 3.1 tables | CXR publication: 377,110 images / 227,835 studies / 65,379 patients. Reports and image IDs are linked; clinical modules require subject AND temporal joins. | CXR-JPG CheXpert/NegBio observations; ED chief complaint, sex and vitals; IV patient age and selected time-stamped labs. No guarantee all modalities exist for each study. | PhysioNet individual credentialing, required CITI training and project DUAs/license. Strongest route to measured physiology; original NIH-14 targets are incompatible. [CXR](https://physionet.org/content/mimic-cxr/2.1.0/), [JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/), [ED](https://physionet.org/content/mimic-iv-ed/2.2/), [IV](https://physionet.org/content/mimiciv/3.1/) |
| 2. CheXpert Plus | Current Stanford page: 223,462 image/report pairs, 187,711 studies, 64,725 patients; reports, DICOM/PNG and study-linked demographics. | CheXpert/CheXbert observation schema; age/sex and other demographics. Not a general EHR/lab/vital-sign dataset. | Apply through Stanford's current Redivis portal; review the actual dataset agreement before use. Good low-complexity age/sex route if genuinely pre-exam clinical-history text is available and justified. [Provider](https://aimi.stanford.edu/datasets/chexpert-plus), [portal](https://stanford.redivis.com/datasets/5yyj-1a9f6ap0x?v=next) |
| 3. ReXGradient-160K | 273,004 images / 160,000 studies / 109,487 patients; multiple sites, sectioned reports and demographics. | PatientAge/PatientSex and study identifiers; no verified native NIH-14 disease target vector. | Hugging Face login, information sharing and provider's non-commercial, non-clinical data agreement. Candidate external cohort after task-specific annotation; not ready for current loader. [Provider card/terms](https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K), [dataset paper](https://arxiv.org/html/2505.00228v2) |
| 4. PadChest | Provider: over 160,000 images / 67,000 patients, Spanish reports, acquisition and demographic metadata. | 174 findings, 19 differential diagnoses, 104 anatomical locations; 27% reports manually annotated, remainder machine-labeled. No verified measured labs/vitals. | Formal application and research-use agreement; provider specifies no sharing/reidentification. Spanish input and ontology mapping need review; English CXR-BERT is not automatically suitable. [Provider](https://bimcv.cipf.es/bimcv-projects/padchest/) |
| 5. Existing NIH ChestX-ray14 | Local prior audit: 31,630 matched images / 12,180 patients; age/sex and NIH-14 weak labels. | Existing task schema; no genuine paired notes. | Preserve current data. Candidate image-only baseline after local provenance review; cannot supply tri-modal training. Local evidence: `audit/DATASET_AUDIT.md`. |

MIMIC-IV 3.1 is a clinical superset, not an additional image corpus; its provider lists 364,627 patients as of v3.0. ED 2.2 contains approximately 425,000 stays. Neither count is the size of the CXR-linked eligible cohort. Shared patient shifts allow within-patient linkage across modules; cross-patient shifted calendar years do not establish real calendar order. [IV version/linkage documentation](https://physionet.org/content/mimiciv/3.1/)

Original CheXpert has 224,316 radiographs from 65,240 patients but does not provide the report/demographic augmentation of Plus. Treat it as an image-label benchmark, not an alternative tri-modal source. [Original provider](https://stanfordmlgroup.github.io/competitions/chexpert/)

Other sources have narrower roles: IU/Open-i is not a verified complete demographic/EHR source; do not infer patient attributes from images or free text. REFLACX supplies 3,032 reading/eye-tracking cases and localization annotations tied to MIMIC images, useful for held-out explanation evaluation after access, not independent prospective clinical inputs. [REFLACX](https://physionet.org/content/reflacx-xray-localization/1.0.0/)

## Storage and transfer decision

Current C: free space was 105,751,220,224 bytes (about 98.5 GiB) during this stage; it fluctuates. No full-image corpus below should be downloaded to it without a fresh size check and storage decision.

| Source | Verified publisher transfer size | Explicit planning estimate / local reservation |
|---|---|---|
| MIMIC original DICOM | Over 4.7 TB, per MIT documentation | Full release infeasible locally. Prefer authorized selective JPG acquisition. [MIT storage guidance](https://mimic.mit.edu/docs/gettingstarted/cloud/gcs.html) |
| CXR-JPG | Exact current total not visible in unauthenticated listing inspected | Budget roughly 0.57 TB **as a planning calculation only**, using 377,110 images × assumed 1.5 MB/image; replace with authorized manifest byte totals. For a 2,000-study one-image pilot reserve 10–20 GiB including sources/cache/logs; actual transfer must be calculated before download. |
| MIMIC IV/ED tables | Exact chosen-version compressed/uncompressed totals not verified | Acquire only necessary columns/tables or approved extracts. Reserve 5–10 GiB for a selected cohort; full event-table extracts may greatly exceed that. No paid cloud query is authorized by this plan. |
| CheXpert Plus | Full package and format-specific transfer sizes unverified behind portal | DICOM/PNG can require TB-scale storage; no precise size claim. Select cohort first, inspect file byte totals, download one format only; 10–20 GiB is a pilot budget, not the full release size. |
| ReXGradient | Total bytes unverified while gated; images distributed as multipart archive | Concatenating parts can require archive + extraction space simultaneously. Confirm whether selective member transfer is supported; do not assume a small cohort means a small download. |
| PadChest | Provider advertises 1.02 TB full / 167 MB sample | Full collection infeasible on current disk; sample completeness and permitted use still require review. |
| NIH / existing ED demo | Already local; no new download needed | Preserve sources and prior receipts; demo has no matching imaging. |

## Access and acquisition sequence

1. **Now:** investigators complete PhysioNet credentialing and project-specific agreements themselves. Use [MIT's access instructions](https://mimic.mit.edu/docs/faq/how-to-get-access.html). Each team member handling restricted data needs appropriate authorization. In parallel, request CheXpert Plus access from the current provider portal. No timeline or approval is guaranteed.
2. **After access:** retain dataset/version/license receipts locally. Obtain small metadata/report/label tables first, not all images. Pin versions and publisher hashes; document download and uncompressed byte totals. Inspect fields and timing before choosing the cohort.
3. **Task decision:** either retain the existing NIH-14 task and obtain independent complete annotations, or implement the reviewed versioned partial-label/shared-task migration described in `TRAINING_READINESS_REPORT.md`. Access alone does not make either MIMIC or Plus pass v2 preflight.
4. **Pilot:** select patient-disjoint eligible records without looking at held-out outcomes; download only associated image files where supported. Aim initially for 100–300 complete observed examinations for mechanics and timing review, then size a real pilot from class support. These are proposed cohort sizes, not acquired records or clinical evidence.
5. **Verify:** source/time audit, unit handling, duplicate checks, eligibility flowchart and manual review; then strict loader/preflight. Record exclusions before freezing splits. Keep raw data, private linkage keys and feature caches out of Git/public papers.
6. **Scale only after measured pilot:** estimate disk, training memory, effective batch/contrastive behavior and annotation burden. Obtain approval before large download, expensive training or paid compute.

The strongest recommendation is MIMIC CXR+ED for physiological context, with CheXpert Plus as the simpler demographic-only alternate. No currently available unrestricted local source removes both the pairing and task-schema blockers.
