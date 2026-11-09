# Dataset discovery and actual use, 22 October 2026

**Result:** suitable full-data routes were identified, and an open clinical demo was downloaded and used. Full tri-modal training remains blocked. The user confirmed that neither Stanford nor PhysioNet dataset access is approved yet.

## Candidate assessment

| Dataset | Suitability | Current decision and source |
|---|---|---|
| MIMIC-CXR + MIMIC-IV-ED | X-rays can be linked to observed ED chief complaints and measured attributes through subject IDs and examination/stay timing. This is the strongest route when genuine vitals are required. | Restricted imaging/full clinical access requires PhysioNet credentialing, training and the applicable agreements. No restricted files downloaded. [CXR](https://physionet.org/content/mimic-cxr/2.1.0/), [ED](https://physionet.org/content/mimic-iv-ed/2.2/), [access instructions](https://mimic.mit.edu/docs/faq/how-to-get-access.html). |
| CheXpert Plus | Paired English radiology reports, X-rays and patient metadata in one release. Strong candidate for the image/text/demographic design. | Follow Stanford's current official Redivis access link and dataset-specific terms. The Hugging Face endpoint returned unauthorized; current Redivis requirements could not be inspected through the text-only page. No data-access rights assumed. [Official dataset page](https://aimi.stanford.edu/datasets/chexpert-plus), [official access portal](https://stanford.redivis.com/datasets/5yyj-1a9f6ap0x?v=next). |
| PadChest | Images, Spanish reports, demographic fields and many annotations. | Full access requires an application and research-use terms. Spanish-language text and report-derived labels require additional validation before the requested English CXR-BERT diagnostic workflow. [Provider page](https://bimcv.cipf.es/bimcv-projects/padchest/). |
| IU/Open-i | Images with report sections, including indications. | Not verified as a sufficient source of observed age/sex; the original NLM publication documents removal of sex information. Do not estimate it from the X-ray. The current Open-i page also returned a maintenance notice. [NLM publication](https://lhncbc.nlm.nih.gov/LHC-publications/PDF/SPIE_2018_gender_fullpaper_final.pdf). |
| Montgomery/Shenzhen TB collections | Public chest images, readings and some age/sex information. | Readings describe the target abnormality, and the task is TB-focused. They do not establish independent pre-prediction clinical inputs for this multi-disease system. [NLM provider](https://www.lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayDatasets.html). |
| Existing NIH data | Already supplies images, labels and real age/sex. | Still lacks observed clinical notes and vitals. Unrelated demo clinical records must never be attached to NIH images. |

## Downloaded and used now

The [MIMIC-IV-ED Demo 2.2](https://physionet.org/content/mimic-iv-ed-demo/2.2/) is explicitly open under ODbL-1.0. The downloader fetched only its license, publisher checksum file, ED stays, triage and vital-sign tables. Every downloaded payload matched the publisher's SHA-256 value. The original license is retained. No authentication was used or agreement signed for a restricted resource.

Files are under `backend/data/external/mimic-iv-ed-demo-2.2/`, excluded from Git. `download_receipt.json` records exact URLs, byte counts and hashes. `audit/mimic_demo_preparation.json` contains aggregate validation results without clinical text or patient identifiers.

| Measured local result | Count |
|---|---:|
| ED stays / triage rows | 222 |
| Patients actually present in downloaded ED stays | 64 |
| Vital-sign rows in source table | 1,038 |
| Complete valid staged clinical records | 193 |
| Patients represented by staged records | 51 |
| Incomplete/invalid records excluded, without imputation | 29 |
| Patient split: train / validation / test | 40 / 5 / 6 |
| Record split: train / validation / test | 161 / 7 / 25 |
| Matching X-rays acquired | 0 |

The provider's demo is drawn from a 100-patient cohort; the actual downloaded ED tables contain only the 64 patients counted above. Counts come from execution, not from substituting the advertised cohort size.

Staging contains the actual chief complaint and observed sex, temperature, heart rate, respiratory rate and oxygen saturation. Source temperature in Fahrenheit is converted explicitly to Celsius. Missing, out-of-range or invalid measurements and missing/redacted chief complaints are excluded. Chief complaints are short clinical text, not full narrative clinical notes. Patient splits are deterministic with seed 42, and all stays of a patient remain together.

`verify_clinical_components.py` used eight held-out observed records on the RTX 4060. Authentic pretrained CXR-BERT returned finite token features of shape `[8, 8, 768]`. Training-patient-only normalization and the metadata MLP produced finite `[8, 512]` representations. The MLP was randomly initialized and used for forward validation only. No optimizer step, diagnostic output, clinical metric or serving checkpoint was produced. Evidence is in `clinical_component_verification.json`.

This is **partial clinical-component preparation**, not a paired image/text/metadata dataset. Its staging file intentionally lacks `image_path`, disease `labels` and study-level linkage, and carries `trimodal_training_ready: false`. It cannot pass the final tri-modal manifest validator. The main training configuration and inference readiness were not changed.

## Requirements before full use

1. Complete the provider's own access steps. For the vitals route, start with [PhysioNet credentialing](https://mimic.mit.edu/docs/faq/how-to-get-access.html), then obtain authorized CXR/CXR-JPG and the appropriate clinical data. Each person handling restricted data needs appropriate authorization. Do not send passwords or private tokens into chat.
2. Use subject IDs plus study time and ED stay interval to establish a unique match. Arrival time is not the exact vital measurement time. Reject ambiguous matches and measurements after the prediction boundary. The demo does not establish these links by itself.
3. Define compatible disease targets. MIMIC/CheXpert observations differ from the current NIH 14-label schema and contain uncertainty/unmentioned values. Never fill unsupported diseases with invented zero labels, blindly rename broader findings, or treat No Finding/Support Devices as extra diseases. Review and explicitly adapt the task/schema before training.
4. Use a chief complaint or appropriately timed indication/history as text input. Exclude report findings/impressions used to derive the target labels. Genuine text can still leak labels; authenticity alone is insufficient.
5. Select one complete examination per patient for the current contrastive loader, preserve patient-disjoint partitions, assess CXR-BERT pretraining overlap, and retain reviewed provenance. A tiny demo is useful for integration but insufficient for defensible diagnostic validation.

## Exact commands

```powershell
.\.venv-gpu\Scripts\python.exe -m backend.scripts.prepare_mimic_demo --download
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.\.venv-gpu\Scripts\python.exe -m backend.scripts.verify_clinical_components --device cuda
.\.venv-gpu\Scripts\python.exe -m unittest tests.test_mimic_demo -v
```

Five new importer tests verify unit conversion, absent/invalid-value exclusion, missing/redacted text rejection, exact stay matching/duplicate rejection, and reproducible patient-disjoint splits. They use artificial test fixtures; the separate GPU component check uses the downloaded observed data. Existing source, dataset and checkpoint files were preserved.
