# X-Fusion setup and reproduction

The corrected tri-modal research prototype is **NOT READY for diagnostic use**. No genuine paired dataset or compatible trained checkpoint is available. The API and UI can start and show this limitation. The old checkpoint must not be used to bypass readiness checks.

Run these PowerShell commands from the project root. Python 3.11 and Node.js 22 are the tested versions. The local `.venv-gpu` has CUDA PyTorch for the RTX 4060 and reuses other packages from `.venv311` through a local `.pth` bridge. GPU and security updates reside in `.venv-gpu`; the original CPU environment was preserved. Use the GPU environment for the verified workflow.

## Existing local environment

```powershell
.\.venv-gpu\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.\.venv-gpu\Scripts\python.exe -m backend.scripts.run_tests
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.\.venv-gpu\Scripts\python.exe -m backend.scripts.verify_encoders --device cuda --output audit/encoder_gpu_verification.json
```

The encoder verifier uses artificial software inputs, runs one disposable optimization step and saves no serving weights. Remove the offline environment variables when downloading the pinned public encoders into a new environment for the first time. No patient dataset is downloaded automatically.

## Fresh standalone environment

```powershell
python -m venv .venv-xfusion
.\.venv-xfusion\Scripts\python.exe -m pip install --upgrade pip setuptools
.\.venv-xfusion\Scripts\python.exe -m pip install torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url https://download.pytorch.org/whl/cu130
.\.venv-xfusion\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv-xfusion\Scripts\python.exe -m pip check
```

`requirements-verified.txt` records the active Windows GPU package versions. To reproduce them exactly in a fresh environment, use `python -m pip install -r requirements-verified.txt --extra-index-url https://download.pytorch.org/whl/cu130`. This lock is Windows/CUDA-specific; the CI environment uses CPU wheels and `requirements-dev.txt`. A fresh installation was not separately reproduced in this session.

## Prepare authorized paired data

The open clinical demo can already be downloaded and checked with `python -m backend.scripts.prepare_mimic_demo --download` and `python -m backend.scripts.verify_clinical_components --device cuda`. It contains only observed clinical components and cannot train the final model. See [dataset search and use](audit/DATASET_SEARCH_AND_USE.md) for actual results and the approved-access route for matching images.

Set `manifest` and `provenance_review` in `config.yaml`. Paths are relative to the project root; image paths are relative to the manifest directory. Supply one JSON object per line with these fields:

| Field | Contract |
|---|---|
| `patient_id`, `study_id` | Nonempty de-identified identifiers, one examination per patient |
| `image_path` | Existing real chest radiograph |
| `clinical_note` | Observed clinical context available at prediction time |
| `metadata` | Exactly the configured original fields, default `age` and `sex` |
| `labels` | All 14 names from `backend/config.py`, each 0 or 1 |
| `split` | `train`, `val` or `test`, all nonempty and patient-disjoint |
| `provenance.observed` | `true`, supported by reviewed source evidence |
| `provenance.license` | Applicable permitted-use record |
| `provenance.note_available_before_prediction` | `true`, supported by timing review |
| `provenance.labels_derived_from_note` | `false`; independent label provenance required |
| `provenance.image`, `.note`, `.metadata` | Each contains matching `patient_id`, `study_id` and a nonempty source reference |

Sex values are `female`, `male`, `other`, or explicitly `unknown`. Missing numeric values are rejected. Do not invent vitals. If observed vitals become available, explicitly extend `model.metadata_fields` using the supported schema. Uncertain labels need adjudication before import. No Finding is represented by fourteen zeros. The existing NIH files are not a valid tri-modal manifest.

The review file must discuss source access, permitted use, matching unit, timing, de-identification, label derivation and leakage. It is human evidence, not an auto-generated certificate. A file's presence cannot establish truth. Review MIMIC label differences and pretrained-encoder overlap if using MIMIC.

## Train and evaluate after the data blocker is resolved

```powershell
.\.venv-gpu\Scripts\python.exe -m backend.pipelines.train --config config.yaml --preflight-only
.\.venv-gpu\Scripts\python.exe -m backend.pipelines.train --config config.yaml
.\.venv-gpu\Scripts\python.exe -m backend.pipelines.train --config config.yaml --resume outputs/xfusion_v2/last.pt
.\.venv-gpu\Scripts\python.exe -m backend.pipelines.eval --checkpoint outputs/xfusion_v2/best.pt --manifest backend/data/paired/manifest.jsonl --device cuda
```

The first command currently fails intentionally because paired data are missing. Freeze split membership and dataset files before running training. Select hyperparameters using validation, not the test partition. Evaluate the locked test set only after model selection. Keep checkpoints, manifests, background attributes and reports private; do not commit them. Old laptop/cluster NIH commands now fail with migration errors.

## Start the API

```powershell
# Set this only after genuine training produces a compatible checkpoint:
# $env:XFUSION_CHECKPOINT = 'outputs/xfusion_v2/best.pt'
$env:XFUSION_DEVICE = 'cuda'
.\.venv-gpu\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Without a checkpoint, `/health` reports `not_ready`. This permits checking the UI and API without pretending inference is available. With a checkpoint, initialization errors also leave readiness false. `auto` selects CUDA if available. Explicit `cuda` does not silently fall back to CPU.

## Start the frontend in another terminal

```powershell
Set-Location frontend
npm ci
npm run lint
npm run typecheck
npm run build
npm start -- --hostname 127.0.0.1
```

Open [the workstation](http://localhost:3000). The project uses Webpack because the installed Windows Turbopack CSS worker failed. The external Google-font build dependency was removed. If the API is elsewhere, set `NEXT_PUBLIC_API_URL` before building. The API CORS default is `http://localhost:3000`; use that browser origin or configure `XFUSION_ALLOWED_ORIGINS` explicitly.

## API contract

- `GET /health`: status, device, readiness reason and research disclaimer.
- `GET /schema`: required multipart fields and configured metadata field ranges/categories/units.
- `POST /predict`: multipart `image`, `clinical_note`, JSON-string `metadata`, `examination_id`, and boolean `paired_inputs_confirmed`. Every input is required. JPEG/PNG/WebP only, at most 15 MB, 20 million pixels and dimensions between 32 and 8192.

Successful responses contain `predictions` and `thresholds` keyed by the same 14 disease names, `selected_conditions`, `explanations`, model/dataset identifiers, measured elapsed time and the complete `report`. The report includes the generated image and available attribution evidence. There is no synthetic success response while readiness is false.

Errors: 422 invalid/missing inputs, 415 unsupported/spoofed format, 413 oversized upload, 503 no usable checkpoint or GPU memory failure, 429 another inference in progress, and 500 failed inference. Explanation failure is reported inside a genuine prediction response and does not fabricate attribution. API documentation is available at [localhost API docs](http://127.0.0.1:8000/docs).

## Retention and troubleshooting

The backend does not persist uploaded images, notes, requests or generated reports. FastAPI may spool uploads to operating-system temporary files during request handling and closes them afterward. The UI holds the current examination in memory. Exporting JSON creates a user-controlled local copy with examination information. Training checkpoints contain the observed metadata background needed for SHAP; protect those files. Do not place identifying patient information in examination IDs or server logs.

If CUDA is unavailable, verify which Python executable starts the server. The preserved `.venv311` PyTorch is CPU-only even though the laptop has an RTX 4060. If `not_ready`, obtain valid paired training and a compatible v2 checkpoint; never substitute the old `.ckpt`. If tests hang at Windows `socketpair`, run in an execution context permitting local loopback sockets. If model loading fails offline, cache the exact pinned files with permitted network access. A separate operating-system CUDA toolkit is not required by the installed CUDA PyTorch wheels.

See `audit/FINAL_COMPLETION_REPORT.md`, `audit/REQUIREMENTS_TRACEABILITY.md`, `audit/DATASET_AUDIT.md` and `audit/TESTING_REPORT.md` for evidence and remaining gaps. These documents supersede older startup and project-completion claims.
