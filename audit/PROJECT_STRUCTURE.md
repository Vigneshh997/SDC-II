# Final project structure

The source and evidence structure is shown below. Existing papers, datasets, figures and historical training outputs remain in their original locations.

```text
3 Project/
  README.md
  HOW_TO_START.md
  config.yaml
  requirements.txt
  requirements-dev.txt
  requirements-verified.txt
  pyproject.toml
  .github/workflows/verify.yml
  backend/
    main.py
    config.py
    checkpoint.py
    inference.py
    explainability.py
    reporting.py
    data/
      dataset.py
      validation.py
      preprocessing.py
      download_dataset.py
      nih_cxr14/                  # Original local data, unchanged
    models/
      multimodal.py
      losses.py
      multimodal_lit.py           # Explicit legacy migration error
    pipelines/
      train.py
      eval.py
      metrics.py
      evaluate_roc.py             # Strict evaluation command adapter
    scripts/
      audit_dataset.py
      verify_encoders.py
      prepare_mimic_demo.py
      verify_clinical_components.py
      run_tests.py
      release_checks.py
      setup_dataset.py           # Blocks old proxy overwrite behavior
      download_*.py
    checkpoints/                 # Original legacy checkpoint, unchanged
    train*.py                    # Historical commands reject the old data workflow
  frontend/
    package.json
    package-lock.json
    next.config.ts
    src/app/
      page.tsx
      layout.tsx
      globals.css
    public/                      # Historical sample files preserved, not served as paired cases
  tests/
    helpers.py                   # Artificial software encoders/fixtures only
    test_core.py
    test_api.py
    test_mimic_demo.py
  audit/
    FINAL_COMPLETION_REPORT.md
    REQUIREMENTS_TRACEABILITY.md
    DATASET_AUDIT.md
    DATASET_SEARCH_AND_USE.md
    ARCHITECTURE.md
    TRAINING_REPORT.md
    EVALUATION_REPORT.md
    TESTING_REPORT.md
    MILESTONES.md
    PROJECT_STRUCTURE.md
    dataset_inventory.json
    mimic_demo_preparation.json
    clinical_component_verification.json
    dataset_work_evidence.json
    checkpoint_inventory.json
    encoder_gpu_verification.json
    test_summary.json
    backend_tests.txt
    release_checks.json
    release/                     # Per-check command logs
    local_startup.json
    bandit.json
    dependency_audit*.json
    npm_audit*.json
    ppt_extracted.json
    ppt_architecture.png
    davit_extracted.txt
    baseline_source.zip
    baseline_hashes.json
    baseline_git_status.txt
    baseline_git_head.txt
    baseline_git_diff.patch
  outputs/                       # Historical metrics retained; no new clinical scores
  .venv311/                      # Preserved original CPU environment
  .venv-gpu/                     # CUDA PyTorch + dependency fixes; local dependency bridge
```

The baseline source ZIP and binary diff are ignored by Git but remain locally recoverable. Existing `.git` history is unchanged. The Git status snapshot records preexisting paper changes. Never restore the snapshot blindly over newer user work; restore only selected files after review.
