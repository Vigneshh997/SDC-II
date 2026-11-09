# Executed verification report

**Dataset follow-up:** the expanded suite now passes **43 tests**, including five new public-demo importer tests. `test_summary.json` and `backend_tests.txt` reflect this latest run. The earlier full release run and its per-command logs below remain a historical 38-test snapshot. New importer lint/format and medium/high Bandit checks also passed; `bandit_dataset_work.json` records the scan. Separate observed-data GPU evidence is in `clinical_component_verification.json` and [the dataset use report](DATASET_SEARCH_AND_USE.md). The original artificial-fixture results below are not relabeled as clinical evaluation.

Final local checks ran on 22 October 2026. **38 backend tests passed, zero failed, zero errors, zero skipped.** All inputs used by those tests were explicitly artificial software fixtures. Genuine-patient end-to-end validation was not performed.

The full recorded release run has **12 passing checks and 1 failing check**. It returns a failing overall exit code because the full development npm dependency audit remains unresolved. No failing check was disabled to obtain a passing release claim.

## Recorded checks

| Check | Result | Evidence |
|---|---|---|
| Python Ruff lint | PASS | `release/python_lint.txt` |
| Python Ruff format check | PASS | `release/python_format.txt` |
| Python compilation | PASS | `release/python_compile.txt` |
| Active GPU environment dependency consistency | PASS | `release/python_dependencies.txt` |
| Backend unit/integration/negative tests | 38 PASS | `test_summary.json`, `backend_tests.txt`, `release/backend_tests.txt` |
| Bandit medium/high-severity scan | PASS | `bandit.json`, `release/static_security.txt`; low-severity findings remain recorded |
| Python package advisory scan | PASS: no known advisories returned | `dependency_audit_final.json` |
| Authentic pretrained GPU encoder/fusion/explanation check | PASS | `encoder_gpu_verification.json`, `release/encoder_cuda.txt` |
| Frontend ESLint | PASS | `release/frontend_lint.txt` |
| Frontend TypeScript | PASS | `release/frontend_types.txt` |
| Production frontend build, Next.js 16.3.8 / Webpack | PASS | `release/frontend_build.txt` |
| Production npm dependency audit | PASS: zero advisories returned | `frontend_production_advisories.json` |
| All npm dependencies including development tools | FAIL: 5 high-severity affected packages | `frontend_all_advisories.json` |
| Default training preflight with no paired manifest | EXPECTED BLOCK: missing genuine manifest | Observed command exception and dataset negative test |
| Real local API startup/readiness/schema | PASS: HTTP 200; readiness explicitly false | `local_startup.json` |
| Real local frontend HTTP response | PASS: HTTP 200 and research disclaimer/title present | `local_startup.json` |
| Visual browser inspection | NOT EXECUTED | CUA failed to initialize its local kernel assets; no screenshot-based claim |
| Hosted CI | NOT EXECUTED | Workflow configured in `.github/workflows/verify.yml`; no remote run claimed |
| Genuine clinical end-to-end workflow | BLOCKED | No paired dataset or compatible trained checkpoint |

The corresponding exit codes and durations are in `release_checks.json`. Earlier failed attempts are retained where useful. The first Windows-sandbox API tests hung in `socketpair`, and sandbox builds encountered worker/filesystem restrictions. Successful final API tests/builds ran with the required local execution permissions. Turbopack's CSS worker also failed outside the sandbox, so the project uses the documented Webpack backend. No lint/type/build checks were suppressed.

## Backend test coverage

Seven API tests cover missing image/notes/metadata/examination confirmation, empty notes, invalid JSON and attributes, corrupted files, MIME spoofing/unsupported formats, oversize uploads, missing checkpoint readiness, and schema consistency.

Thirty-one component/integration tests cover metadata validation and training-only normalization, explicit categorical handling, preserved negations, CLAHE geometry, output shapes, all-branch gradients, frozen encoder state, each modality's effect, actual receiving-query/other-context attention construction, padding invariance, parameter updates, stable ASL, correct InfoNCE pairing and invalid contrastive batches, valid/invalid manifest linkage, patient repetition, duplicate images, label-derived note rejection, uncertain-label rejection, disabled NIH path, actual training-loop/checkpoint/resume execution on temporary fixtures, LayerCAM spatial-gradient math, original-size overlays, text attribution, signed SHAP additivity, missing-background rejection, report integration/lock release, metric behavior and legacy-checkpoint rejection.

The authentic GPU verifier loads both pinned real encoders, runs the 512-dimensional architecture, backpropagates the actual combined objective, performs one disposable software optimization step and measures input sensitivity. The final recorded output has logits `[2, 14]`, CNN maps `[2, 512, 14, 14]`, nonzero CNN/text projection/MLP/attention gradients, nonzero logit changes for each modality, available text/metadata explanations, a LayerCAM overlay and 16 attention links. Peak CUDA allocation was 1,327,502,848 bytes. Timings in the JSON are single-run software-fixture measurements, not stable clinical benchmarks. No model checkpoint from that check was saved.

## Dependency remediation and unresolved issue

CUDA PyTorch was installed in `.venv-gpu` while `.venv311` was preserved. Its setuptools requirement was resolved. Python package advisories in aiohttp, certifi, Pillow and pip were remediated in the active GPU environment, and the follow-up scan returned no known advisories. The preserved CPU environment was not upgraded and is not the recommended verified runtime.

The frontend was updated within the Next.js 16.3 release line to 16.3.8, unused Axios/Framer Motion/clsx/tailwind-merge dependencies were removed, and compatible transitive fixes were applied. The production-only npm scan is clean at the recorded time.

The full scan still reports the [braces stack-exhaustion advisory](https://github.com/advisories/GHSA-vfj7-8cjw-p6xm), propagated through micromatch, fast-glob, `@next/eslint-plugin-next` and `eslint-config-next`. These are five affected development dependency nodes for one underlying advisory. npm's proposed forced remediation downgrades the Next lint configuration to 14.2.35. That breaking downgrade was not applied to the Next 16 project. The workflow retains its full audit check and should therefore not be described as green while this advisory remains. No advisory was ignored or hidden.

Security scans are limited checks, not a proof of security or clinical/regulatory compliance. Remaining work includes a vendor-supported fix for the development advisory, hard backend timeout behavior, complete latency/resource profiling, deployment-specific authentication/rate limits/privacy review, valid clinical evaluation and explanation fidelity testing.

## Reproduce

```powershell
.\.venv-gpu\Scripts\python.exe -m backend.scripts.run_tests
.\.venv-gpu\Scripts\python.exe -m backend.scripts.release_checks
```

The release check command requires network access for advisory queries and local-loopback/worker permissions for API tests and builds. It intentionally exits nonzero for unresolved checks. GPU verification requires the pinned public encoders already cached and does not run in lightweight hosted CI.
