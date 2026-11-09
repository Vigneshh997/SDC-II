# X-Fusion Springer Nature journal paper

Main source: `X_Fusion_Springer.tex`. Compiled paper: `X_Fusion_Springer.pdf`.

Converted from a reconstructed manuscript using the supplied Springer Nature December 2024 version 3.1 journal template. During the 27 October 2026 cleanup, older drafts and the downloaded template ZIP were moved into the recovery archive. This directory retains the journal source, PDF and required local build assets. See [the paper index](../../README.md) for the separate MIKE conference versions.

## Build

Upload all files in this folder to Overleaf, choose `X_Fusion_Springer.tex` as the main document, and select pdfLaTeX. Alternatively run:

```text
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape X_Fusion_Springer.tex
bibtex X_Fusion_Springer
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape X_Fusion_Springer.tex
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape X_Fusion_Springer.tex
```

Keep the class, bibliography style, bibliography database, and three image files beside the main source. The generated `.bbl` is included for convenience.

## Conversion

- Replaced IEEEtran with the supplied unmodified `sn-jnl.cls` and its default sample choice `sn-mathphys-num`.
- Converted the title, six author entries, shared institutional affiliation, abstract, and keywords to native Springer commands.
- Preserved author order, contact details, manuscript prose, numerical results, equations, all six tables, and all three original image assets.
- Removed the Introduction subsection heading to follow the supplied sample's guidance; its paragraphs are retained.
- Reflowed the manuscript in the template's standard single-column layout, enlarged table type, and resized the illustrative radiograph.
- Converted the twelve supplied references into a BibTeX database and typeset them with the supplied Springer style. No missing bibliographic fields were invented.
- Used Springer back matter markup for acknowledgements.

## Author details to complete before submission

No specific journal was identified. This is the supplied Springer Nature journal format, not an LNCS conference template. Confirm the intended journal's reference style and submission requirements.

The source does not nominate a corresponding author. All six authors remain contributing authors; once nominated, mark the appropriate author with `\author*`.

Funding, competing interests, ethics and consent, author contributions, public code/data availability, and an accurate AI assistance disclosure require author-confirmed information. These statements have not been invented or populated with unverified “not applicable” declarations. Add the declarations required by the selected journal before submission.

This is a format conversion, not a validation of the experiments. The source's label-conditioned-input caveats and draft-reported comparison tables remain. Existing concerns about comparative metric provenance, threshold denominators, architecture-image terminology, and radiograph provenance still require author review. No model was rerun and no research results were changed.
