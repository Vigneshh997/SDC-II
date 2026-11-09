# Second-stage verification — 31 October 2026

## Executed

- `python -m unittest tests.test_training_readiness tests.test_reliability -v`: 14 tests passed, zero failures/errors, 0.160 seconds reported by unittest. Seven readiness tests are new; seven reliability/artifact tests originated in the first stage. This is fixture/software validation, not clinical validation.
- `python -m ruff check backend tests`: passed.
- Ruff format check on the two new Python files: passed.
- `python -m compileall -q` on the two new Python files: passed.
- `git diff --check`: passed.
- The new readiness command ran with `config.yaml` and a new output file, producing `stage2_readiness.json`: structural_preflight_passed=false, manifest_present=false, provenance_review_present=false, no existing run artifacts, clinical_validity=not_certified. The checker exits 1 for this expected blocked condition. A chained shell command's overall success must not be interpreted as successful data preflight.

The readiness tests verify no encoder loading when data are missing, no raw exception/clinical-content disclosure, aggregate-only successful reports that still require review, preservation of existing run artifacts, nonmapping config rejection, blocked CLI exit 1 and exclusive output creation protecting even the config file from replacement. Mocked rows are isolated software fixtures; no manifest or patient dataset was generated to bypass the data blocker.

Only new software files were added: `backend/scripts/check_training_readiness.py` and `tests/test_training_readiness.py`. Existing model, losses, manifest loader, trainer, checkpoint contract, API and frontend were not edited in this stage. The implementation log was appended and seven requested research reports were created. Existing dataset/checkpoint/history artifacts were not replaced.

The earlier stage's 50-test full-suite pass and authentic pretrained GPU fixture check remain prior evidence, not freshly repeated in this stage. No frontend source changed, so frontend build/browser checks were not repeated. Full dataset content hashing, source-specific restricted-data linkage, genuine training, test-set evaluation, learned ablations and journal submission were not performed.

## Research verification boundaries

Official provider and author/proceedings sources were consulted for current data scope, access, labels, time semantics and prior work. Exact gated download totals and current Stanford license text remain unverified and are explicitly marked unknown in the acquisition plan. Budget values are labeled calculations/reservations rather than measured transfer sizes.

DAViT upstream source was read through the GitHub connector at commit `021509e7d546dbf01c0ee898edd6a18debaa9051`. No third-party source was executed/vendored and no upstream model artifact was downloaded. Source-level findings and links appear in `DAVIT_BASELINE_PLAN.md`.

Current free disk measurement: 105,751,220,224 bytes on C:. RTX 4060 8 GB and the earlier GPU evidence are documented in the experiment protocol. The OS physical-RAM query was denied; 16 GB is user-provided context, not independently measured here. No elevated hardware query was needed to complete the planning task.

Remaining external blocker: provider-approved access and an eligible genuinely paired cohort. Remaining engineering prerequisite for typical MIMIC/Plus labels: a reviewed task/schema migration or complete independent NIH-14 annotation. Research plans do not remove either prerequisite.
