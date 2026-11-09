# Final project papers

| Package | Paper PDF | Editable source |
| --- | --- | --- |
| MIKE 2026 final (named final in the original workspace) | [X_Fusion_MIKE2026_Final.pdf](final/mike2026/X_Fusion_MIKE2026_Final.pdf) | [X_Fusion_MIKE2026_Final.tex](final/mike2026/X_Fusion_MIKE2026_Final.tex) |
| MIKE 2026 reviewed anonymous revision | [X_Fusion_MIKE2026_Anonymous.pdf](final/mike2026_reviewed/X_Fusion_MIKE2026_Anonymous.pdf) | [X_Fusion_MIKE2026_Anonymous.tex](final/mike2026_reviewed/X_Fusion_MIKE2026_Anonymous.tex) |
| Springer Nature journal format | [X_Fusion_Springer.pdf](final/springer/X_Fusion_Springer.pdf) | [X_Fusion_Springer.tex](final/springer/X_Fusion_Springer.tex) |

The reviewed anonymous revision has the most detailed revision and source-check record. These three versions are distinct manuscripts; the filename "final" does not establish submission, acceptance, experimental validity or compatibility with the current application. Read the [reviewed revision record](final/mike2026_reviewed/REVISION_AND_SOURCE_CHECKS.md) and [submission audit](review_records/X_Fusion_Submission_Audit.md) before using their results.

Each package is self-contained: preserve its local `.tex`, `.bib`, `.bbl`, `.cls`, `.bst` and figure files together. Repeated class/style/bibliography files across packages are intentional build dependencies. `per_disease_performance.pdf` is a figure used by the reviewed manuscript, not an extra paper. Its `evidence/` folder contains reproducibility scripts, predictions and metric records.

Build within the selected package directory, selecting its main `.tex` file. Use pdfLaTeX, BibTeX, pdfLaTeX, then pdfLaTeX, or upload the entire selected package to Overleaf. Scientific content and existing paper PDFs were not changed during organization.

Other materials:

- `references/`: DAViT reference paper.
- `project_figures/`: additional project figures; identical loose copies of retained final-paper figures were removed.
- `submission_records/`: CMT records and author notes, including private coauthor information. These are outside the anonymous package.
- `review_records/`: historical submission audit.

Older drafts, duplicate loose files, downloaded template archives, the old reviewed ZIP and temporary paper work are recoverable from [the cleanup archive](../backups/cleanup-2026-10-27.zip). The original-path manifest is in [the cleanup report](../audit/cleanup-2026-10-27.json).
