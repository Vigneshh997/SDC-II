# Analysis of the supplied sn directory

The directory contains one template folder, `sn-article-template`, and its `bst` subfolder: 16 files in total. These are publisher authoring resources rather than additional research-paper material.

| File | Role and conversion decision |
| --- | --- |
| sn-article.tex | Version 3.1 December 2024 example; defines native front matter, numbered sections, equations, floats, back matter, and BibTeX usage. Its active class option is `pdflatex,sn-mathphys-num`; adopted. Sample prose excluded. |
| sn-article.pdf | Twelve-page sample illustrating the template's single-column journal layout, author block, headings, captions, and references. Used as a layout reference. |
| sn-jnl.cls | Publisher class controlling journal typography, page geometry, front matter, natbib, and back matter. Copied without modification. Its internal 2019 class date differs from the December 2024 package version; preserved as supplied. |
| sn-bibliography.bib | Example entries for articles, books, proceedings, datasets, preprints, and software. Example references excluded; replaced with the twelve references in the existing paper. |
| user-manual.pdf | Instructions for native author/affiliation markup, single-source manuscripts, image placement, bibliography processing, and journal-specific requirements. Images and selected style file are placed beside the manuscript as instructed. |
| empty.eps | Dummy EPS placement asset with bounding box 0 0 115 87. Not research content; excluded. |
| fig.eps | Sample wide EPS figure with bounding box 0 0 504 84. Not research content; excluded. |
| bst/sn-basic.bst | Basic Springer Nature/Chemistry references; alternative, not selected. |
| bst/sn-mathphys-num.bst | Numbered mathematics and physical sciences style; active sample default, selected and copied unchanged. |
| bst/sn-mathphys-ay.bst | Author-year counterpart; alternative, not selected. |
| bst/sn-aps.bst | American Physical Society references; alternative, not selected. |
| bst/sn-vancouver-num.bst | Numbered Vancouver references; alternative, not selected. |
| bst/sn-vancouver-ay.bst | Author-year Vancouver references; alternative, not selected. |
| bst/sn-apacite.bst | APA references, selected via the class's `sn-apa` option; alternative, not selected. |
| bst/sn-chicago.bst | Chicago humanities references; alternative, not selected. |
| bst/sn-nature.bst | Nature Portfolio references; alternative, not selected. |

There are nine bibliography style files. The sample and manual both make the destination journal's instructions authoritative; choosing the active sample default enables a concrete conversion without assuming a journal or switching to an unrelated proceedings format.

## Source archive

`FINAL_PAPER.zip` contains seven entries: a README, a reconstruction verification JSON, the reconstructed IEEE TeX and PDF, and three figures. The README identifies the TeX as an editable reconstruction of the six-page IEEE reference. That packaged TeX is the conversion authority; similarly named loose files in the parent folder were not substituted.

The conversion retains all 12 references, 6 tables, 3 figures, equation content, numerical entries, and body paragraphs. Bibliography punctuation and ordering follow the publisher style. The original image bytes are preserved. The original six-page IEEE pagination is replaced by the natural Springer journal pagination.
