# X-Fusion submission audit

Reviewed 14 October 2026. Verdict: **not ready for submission as a validated clinical-performance study**. A wording revision cannot resolve the experimental issues below. No plagiarism percentage, AI-detector score, or conference acceptance guarantee is issued.

## Scope and preservation

Read the TeX extracted from `X_Fusion_IEEE_Camera_Ready_Package.zip`, extracted and visually inspected all six PDF pages, and inspected all three supplied images. Also checked relevant local model/data code, saved test predictions, evaluation JSON, and TensorBoard records. The original ZIP, TeX, PDF, images, and application code were not modified. Review copies and page renders are under `tmp/pdfs/x_fusion_review/`.

No Turnitin/iThenticate or AI-detector scan was available or performed. Limited public-web phrase searches and selected reference checks are not a comprehensive plagiarism assessment. No reverse-image provenance search was performed. No model was retrained or inference rerun; metric checks below use existing saved predictions.

## Submission blockers

### 1. The packaged TeX does not correspond to the packaged PDF

This is a substantive version mismatch, established directly from their contents:

| Item | Packaged TeX | Packaged PDF |
|---|---|---|
| Abstract opening | “X-Fusion combines chest radiographs…” | “Chest radiographs frequently exhibit coexisting…” |
| Section II | Literature Review | Related Work |
| Figures | Two native LaTeX picture environments | Three raster figures, including a radiograph |
| Image inclusion | No `includegraphics` commands or `graphicx` package | Includes all three supplied image types |
| Acknowledgment | Explicit Codex assistance disclosure | Thanks institute and NIH; no corresponding AI disclosure |
| References | More detailed DOI/URL entries | Shortened entries in a different citation order |

Do not rebuild this TeX expecting the existing six-page PDF to be reproduced. Recover the actual source of the PDF, then make content corrections in that source while retaining its class, margins, fonts, column settings, and figure/table sizing. Exact pagination can still change when text changes; a rendered comparison is necessary. This audit did not choose one conflicting version as authoritative or replace the PDF with a different layout.

### 2. Target leakage invalidates clinical-benefit claims

`backend/data/dataset.py:144` generates notes from disease labels. At line 163 and following, metadata generation reads `Finding Labels` and samples vital signs conditional on Pneumonia/Edema. The dataset calls these functions at lines 260 and 263 for its samples, including evaluation samples.

The classifier therefore receives information derived from its target. Patient-disjoint partitions do not remove this dependency. Label-derived inputs also differ materially from independently collected clinical information available at deployment. The paper correctly discloses this limitation, but disclosure does not make the comparative scores evidence of diagnostic improvement.

Required experimental repair: use independently collected, appropriately timed clinical inputs, or evaluate a clearly specified image/real-demographics task without label-derived inputs. Retrain affected models, fix validation inputs, select thresholds using validation only, and evaluate once on a held-out patient partition. Preserve image IDs, patient IDs, labels, scores, configurations, seeds, and checkpoint identifiers. Do not merely remove the leakage disclosure or replace scores with more plausible numbers.

### 3. Comparative and ablation claims remain unsupported

The package reports macro AUROC 0.9420, micro AUROC 0.9615, and mAP 0.8840 for X-Fusion and includes baseline/ablation tables sourced to a Word draft. The ZIP contains no prediction arrays for those configurations.

However, the current workspace DOES contain `outputs/xfusion/test_predictions.csv`. Therefore the TeX's broad suggestion that saved predictions are absent is outdated or too broad. It should distinguish unsupported comparative-table predictions from the available, leakage-affected test predictions.

Recalculation from 7,100 unique image rows and fourteen label/probability pairs gives:

| Quantity | Recomputed | Saved JSON, rounded |
|---|---:|---:|
| Macro AUROC | 0.9999611393 | 1.0000 |
| Macro average precision | 0.9953269859 | 0.9953 |
| Macro F1 at 0.5 | 0.9708568396 | 0.9709 |

These reproduce the rounded aggregate report; they do not validate the experimental design or establish how the predictions were originally produced. Use “macro average precision” when reporting the sklearn `average_precision_score` calculation; explain any use of “AUPRC,” since interpolated area and average precision need not be identical.

The distinct reports might represent different experiments; the numerical difference alone does not prove misconduct. Their provenance must be identified. Until original predictions and protocols support the comparison tables, remove those empirical claims from a submission or replace them with newly measured, documented results. This is a content decision, not a formatting fix.

### 4. Pneumonia threshold results need reconciliation

The reported partition has 5 validation positives and 18 test positives. For a single binary evaluation on those partitions, recall must be a multiple of 1/5 or 1/18, respectively. Table recall values such as 0.932 cannot arise from either stated partition as a single pooled confusion matrix, even after rounding to three decimals. If these are averages across folds/runs or another cohort, report that protocol and its denominators explicitly.

From the saved test predictions (18 positives), using scores >= threshold:

| Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0.20 | 0.3750 | 1.0000 | 0.5455 |
| 0.30 | 0.4146 | 0.9444 | 0.5763 |
| 0.35 | 0.4250 | 0.9444 | 0.5862 |
| 0.40 | 0.4722 | 0.9444 | 0.6296 |
| 0.50 | 0.5667 | 0.9444 | 0.7083 |

These do not reproduce the draft threshold table. They are diagnostic checks, not a recommendation to select a threshold on the test set, and they remain leakage-affected.

## Figure and PDF review

All six PDF pages are US Letter (612 x 792 points). Visual inspection found no obvious major clipping or overlapping body text. This is not an IEEE PDF eXpress validation or a venue-specific compliance certificate. No target conference, page limit, or review/anonymity requirements were supplied.

| Asset | Finding | Required correction/evidence |
|---|---|---|
| `fig1_architecture.png`, 1020 x 414 | Says “Chefer-CAM Heatmaps”; the inspected implementation averages final-block CLS attention and does not implement Chefer relevance propagation. “Cross-Attention Fusion” also obscures that `TransformerEncoder` applies self-attention across the four stacked tokens. InfoNCE is drawn inside fusion although the implementation computes it from image/text representations. Small labels are difficult at page scale. | Relabel as averaged visual attention and multimodal self-attention; identify InfoNCE as an auxiliary training loss on image/text features. Keep the original figure footprint if revising. |
| `fig2_losses.jpg`, 1600 x 654 | Correctly separates objectives and scales and discloses label-derived inputs. Fourteen scalar records corroborate displayed endpoints. | Retain caveats. A loss curve is not a generalization or localization validation. |
| `fig3_cxr.png`, 1024 x 1024 | An illustrative radiograph with no prediction overlay; PDF caption appropriately avoids diagnosis. The package does not identify its exact source image or reuse provenance. | Record dataset image ID, source, and applicable reuse terms. Cite the source in the caption where appropriate. Do not infer an image's license or originality from appearance. |

TensorBoard `lightning_logs/version_7` contains 14 `train_loss_epoch` and 14 `val_loss` records. Endpoints are 0.1544914693 -> 0.1190370172 and 0.0206883475 -> 0.0007261421, respectively, matching the displayed rounded values. The roughly 22.95% training-loss reduction is supported by the unrounded endpoints.

The main tensor dimensions and ASL/InfoNCE formulas are consistent with the inspected implementation. They largely describe established components, so the novelty claim should focus on demonstrated integration or an experimentally supported contribution, not imply that standard attention, ASL, or SHAP were invented here.

## Originality, attribution, and AI disclosure

Limited searches for the paper title and selected distinctive sentences did not identify a clear copied passage in the returned results. This is **not** a clean plagiarism verdict: coverage excludes private submissions, many full-text sources, translated/paraphrased overlap, and image provenance. Review an institutional similarity report match by match, including source attribution for methods, ideas, figures, and any reused text. A low percentage alone is not proof of originality.

Selected primary-source checks corroborated the cited [BiomedCLIP paper](https://arxiv.org/abs/2303.00915), [Boecking et al. vision-language paper](https://arxiv.org/abs/2204.09817), and [DAViT article and publication details](https://ieeexplore.ieee.org/abstract/document/11031440). This was not an exhaustive full-text comparison or verification of every bibliography field.

The TeX already records substantive Codex assistance in drafting, equations, reference checks, and figures. The PDF does not carry that disclosure. A rewrite cannot truthfully certify that such a history never occurred. IEEE's [AI-generated content policy](https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/) calls for disclosure of generated content, the system used, affected sections, and the extent of use; grammar-only editing is treated differently. Authors should verify the existing disclosure's accuracy and include the appropriate statement in the final authoritative manuscript. Do not delete it merely to seek a detector score.

## Required next steps, in order

1. Recover the source that actually builds the six-page PDF and establish a single authoritative version.
2. Resolve dataset leakage and produce documented experiments before making diagnostic-benefit claims.
3. Reconcile or remove unsupported comparison/ablation/threshold tables; correct the statement about available predictions.
4. Correct figure terminology and document radiograph provenance; retain accurate AI assistance attribution.
5. Have the authors review the research claims and obtain an institutional similarity report. Assess actual matches rather than pursuing an arbitrary score.
6. Select a specific venue/track and check its scope, anonymity, page limits, required disclosures, and submission validation requirements. Acceptance remains a reviewer/editor decision.

The present evidence can support a description of a software prototype and an analysis of leakage-affected experiments. It does not support a guaranteed conference acceptance or a verified clinical-performance claim.
