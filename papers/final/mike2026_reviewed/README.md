# MIKE 2026 anonymous manuscript with measured reference baselines

Main source and PDF: X_Fusion_MIKE2026_Anonymous. Compile using pdfLaTeX, BibTeX, pdfLaTeX, pdfLaTeX. TikZ and newtx fonts are required. The architecture source is included.

The evidence directory contains newly executed independent-input logistic-regression baselines, split manifests, aggregate and per-label results, and saved test predictions. Reproduce with:

```
python evidence/independent_baselines.py --data PATH_TO_NIH_DATA --output NEW_RESULTS_DIRECTORY
```

The data directory must contain Data_Entry_2017_v2020.csv, train_val_list.txt, test_list.txt, and images/*.png. Original NIH images are not redistributed. Install evidence/requirements.txt in an appropriate environment.

The independent reference models use real pixels and age/sex only. They do not validate the original tri-modal model. No independent clinical notes or vital signs were available for new tri-modal training. The original contaminated scores remain clearly separated and are not clinical-performance estimates. The previous ZIP is backed up separately. No upload, submission, or acceptance has occurred.

The review manuscript remains anonymous. Enter actual authors separately in CMT. Scientific novelty and clinical validity remain subject to review; the new baseline experiment does not guarantee acceptance.

## Reviewed revision

The attention equation, backbone/fusion head distinction, Table 3 warning, references, and rare-label discussion have been revised. The per-disease vector chart includes exploratory patient-cluster bootstrap intervals. Its source is `evidence/revision_analysis.py`; run it from this extracted package to regenerate the chart and uncertainty files from the included prediction CSVs. It uses numpy, pandas, scikit-learn, and matplotlib. The bootstrap uses 500 resamples and seed 20261015.

`REVISION_AND_SOURCE_CHECKS.md` records reference sources, figure provenance, completed checks, and declarations requiring author confirmation. Coauthor names and mentor/team roles are recorded separately outside this anonymous source package. No new BiomedCLIP or fusion training was performed in this revision.
