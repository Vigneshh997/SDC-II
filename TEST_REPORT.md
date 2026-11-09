# Verification — 31 October 2026

## Executed checks

| Check | Result |
|---|---|
| `python -m unittest discover -s tests -v` | 50 tests, 0 failures/errors, 7.468 seconds reported by unittest, outside sandbox |
| New reliability/artifact regression tests | 7 passed; also included in the full suite |
| `python -m ruff check backend tests` | Passed |
| Ruff format check on changed Python files | Passed |
| `python -m pip check` | No broken requirements |
| `python -m compileall -q backend tests` | Passed |
| `git diff --check` | Passed |
| Frontend `npm run lint` | Passed |
| Frontend `npm run typecheck` | Passed |
| Default training `--preflight-only` | Expected failure: genuine paired JSONL manifest missing |
| Cached authentic encoder GPU verification | Generated successful check evidence in `audit/current_review/encoder_verification.json` |

The full suite exercises input validation/API readiness, provenance/alignment/duplicate checks, padding and cross-attention behavior, gradient flow, loss stability, two-epoch tiny-model fixture training and checkpoint resume, explanation execution and SHAP additivity, and the new metrics/checkpoint-preservation cases. Fixture training uses injected tiny encoders and temporary artificial data. It is not genuine patient training.

An initial sandbox full-suite attempt stalled at Windows asyncio socket-pair creation. An isolated faulthandler trace identified that location. A sandbox non-API run completed 35 of 36 tests but could not rename a temporary checkpoint (`WinError 5`). The approved outside-sandbox full run passed all 50 tests. These limitations did not justify weakening application validation or checkpoint handling. FastAPI TestClient and SHAP emitted upstream deprecation warnings; there were no failing assertions in the successful run.

## Authentic encoder smoke check

Command: `.venv-gpu\Scripts\python.exe -m backend.scripts.verify_encoders --device cuda --output audit/current_review/encoder_verification.json`.

Cached pinned weights loaded on RTX 4060 Laptop GPU (8,188 MiB total according to nvidia-smi). Reported logits: 2x14; CNN activations: 2x512x14x14. All checked trainable branches had nonzero gradients. Perturbing each input modality changed logits. The script executed one disposable optimization step, LayerCAM, text attribution, metadata KernelSHAP and 16 attention links; it saved no serving checkpoint. Peak allocated CUDA memory was 1,327,502,848 bytes. Reported batch-two forward time was 1.252 seconds and single-explanation time 4.637 seconds. These are fixture timings, not clinical throughput or a full-training memory budget.

## Not established

No compatible trained diagnostic model, genuine patient end-to-end validation, partial-label training, missing-modality inference, calibrated clinical risk, external-domain performance, DAViT reproduction or learned ablation result. Frontend production build/browser interaction was not repeated because frontend source was unchanged; lint/typecheck are narrower checks. Earlier full dataset hash/integrity counts were reviewed, not recomputed. Full dependency advisory scans were not repeated; pip check establishes dependency consistency only.
