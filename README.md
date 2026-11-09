# X-Fusion tri-modal research prototype

The corrected implementation combines authentic BiomedCLIP spatial features and CNN refinement, CXR-BERT clinical tokens, and an observed-attribute MLP through genuine cross-attention. Training combines multi-label Asymmetric Loss and all-pair InfoNCE. FastAPI and Next.js expose a strict three-input workflow, real model evidence and JSON reports.

**Release status: NOT READY.** Genuine paired clinical data and a compatible trained model are unavailable. Predictions remain disabled. Artificial-fixture software tests and pretrained-encoder GPU checks do not establish diagnostic performance.

Start with [the setup guide](HOW_TO_START.md). Evidence and limitations are recorded in [the completion report](audit/FINAL_COMPLETION_REPORT.md), [dataset audit](audit/DATASET_AUDIT.md), [architecture](audit/ARCHITECTURE.md), [training report](audit/TRAINING_REPORT.md), [evaluation report](audit/EVALUATION_REPORT.md) and [test report](audit/TESTING_REPORT.md).

Final paper PDFs, editable LaTeX, figures and bibliography files are indexed in [papers/README.md](papers/README.md). Project review documents are in [docs/](docs/) and the review presentation is in [presentations/](presentations/). The [file organization guide](FILE_ORGANIZATION.md) records the cleanup and recovery archive.

Existing datasets, legacy checkpoints and historical training outputs were preserved. Historical scores from label-derived clinical inputs are excluded from valid tri-modal performance claims.

The [31 October baseline](PROJECT_BASELINE.md), [current audit](PROJECT_AUDIT.md), [DAViT reproduction assessment](BASE_PAPER_REPRODUCTION.md), [implementation roadmap](IMPLEMENTATION_ROADMAP.md) and [implementation log](IMPLEMENTATION_LOG.md) distinguish existing functionality from new work. The latest incremental changes add held-out calibration measurements and protection against accidental training-output reuse; see [evaluation definitions](EVALUATION_REPORT.md) and [executed verification](TEST_REPORT.md).
