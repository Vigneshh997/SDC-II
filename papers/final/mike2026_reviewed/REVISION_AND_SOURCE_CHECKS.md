# Revision record

The review manuscript is anonymous. The separate private author record preserves the six names and order supplied by the user, including their identification of the first author as mentor. No corresponding author, ORCID, equal contribution, individual contribution, approval, funding declaration, or ethics exemption has been invented.

## Requested improvements

1. Equation 6 explicitly typesets softmax of QK-transpose divided by the square root of the key width, multiplied by V. It is checked in the rendered PDF rather than judged from extracted text.
2. Section 5.4 distinguishes the 12 heads in the final BiomedCLIP visual-backbone block from the 8 heads in each fusion encoder layer. The attention overlay is extracted before fusion. This matches the inspected implementation.
3. No new encoder/fusion training was requested after the available RTX 4060 hardware and CPU-only PyTorch installation were identified. The manuscript explicitly says a leakage-free BiomedCLIP versus X-Fusion comparison remains unperformed. No reduced image-demographics experiment is represented as a test of tri-modal fusion.
4. Table 3 prominently states “Label-conditioned pipeline diagnostics—not clinical validation” in bold. The numeric values are unchanged.
5. A new vector per-disease chart recomputes AUROC and average precision from the two genuine saved prediction files. The chart, plotting code, disease metrics, bootstrap settings, input SHA256 hashes, and uncertainty summary are included.
6. The 18 Pneumonia-positive images are verified to represent 12 patients. The manuscript explains the 1/18 recall increment and uses patient-cluster resampling for baseline uncertainty. Five hundred resamples use seed 20261015 and hold fitted models fixed. These exploratory intervals do not include training uncertainty. The paired macro AUROC difference is 0.0026897370 with a 95% percentile interval [-0.0018530481, 0.0069516497]; no reliable demographic benefit is claimed.
7. Bibliographic metadata was checked against publisher/proceedings pages, author-hosted institutional records, or arXiv. Identifying author details remain outside the review package. Submission declarations remain subject to author confirmation.

## Reference checks

These are bibliographic metadata checks, not a similarity scan, a verification of every statement in each source, or a guarantee that all referenced material is correct. Abbreviated author lists already present in the bibliography remain where appropriate; the Wang entry now names all six authors.

| Citation key | Authoritative record and action |
| --- | --- |
| wang2017 | [CVF proceedings](https://openaccess.thecvf.com/content_cvpr_2017/html/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.html): confirmed title, six authors, CVPR 2017, pp 2097–2106. Added full authors and source URL. |
| nihdataset | [Official NIH distribution](https://nihcc.app.box.com/v/ChestXray-NIHCC): retained the supplied original download URL. Automated retrieval was unavailable; no new licensing claim was inferred. |
| irvin2019 | [AAAI publisher](https://ojs.aaai.org/index.php/AAAI/article/view/3834): confirmed title, authorship, 2019, volume 33, pp 590–597; added DOI 10.1609/aaai.v33i01.3301590. Issue omitted because the proceedings BibTeX style accepts volume rather than both volume and number. |
| dosovitskiy2021 | [Author arXiv record](https://arxiv.org/abs/2010.11929) identifies the ICLR 2021 camera-ready version. Retained ICLR 2021 and added its OpenReview URL. Direct OpenReview access requested browser verification; it was not bypassed. |
| vaswani2017 | [NeurIPS proceedings](https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html): confirmed title, authors, volume 30, 2017; added source URL. |
| biomedclip2023 | [arXiv version history](https://arxiv.org/abs/2303.00915): existing title corresponds to v3, revised January 2025. Corrected the displayed year to 2025 and cited v3 explicitly, noting first submission in 2023. Added the arXiv DOI; stable citation key retained. |
| cxrbert2022 | [Springer publisher](https://link.springer.com/chapter/10.1007/978-3-031-20059-5_1): confirmed 2022, LNCS 13696, pp 1–21; added volume, pages, publisher, and DOI. |
| davit2025 | [Authors’ institutional record](https://research.monash.edu/en/publications/davit-a-domain-adapted-vision-transformer-for-automated-pneumonia/): confirmed three authors, IEEE Access 13, 103033–103044, 2025; added DOI 10.1109/ACCESS.2025.3579314. Direct IEEE access requested browser verification; it was not bypassed. |
| ridnik2021 | [CVF paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf): confirmed title and ICCV 2021; added pp 82–91 and proceedings URL. |
| oord2018 | [Author arXiv record](https://arxiv.org/abs/1807.03748): confirmed three authors, title, first submission 2018; added arXiv DOI. |
| lundberg2017 | [NeurIPS proceedings](https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html): confirmed authors, title, volume 30, 2017; added source URL. |
| chefer2021 | [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html): confirmed three authors, title, CVPR 2021, pp 782–791; added source URL. |

## Figure provenance and permissions

- Architecture: editable TikZ schematic of the project's implementation, not a reproduced publisher figure.
- Optimization traces: existing project-generated illustration of saved TensorBoard records. Earlier checks supported its endpoints. The original image is retained; author ownership and permission to publish the project material still require author confirmation.
- Disease chart: newly generated directly from included prediction CSVs using the included script. It contains no external radiograph or copied publisher figure. Its figure source and values are reproducible.
- The illustrative radiograph of unverified provenance remains excluded. No claim is made that appearance alone establishes an image's license.

## Author confirmation still required

Confirm all coauthors approve the final text and their order, the intended corresponding author in CMT, any funding and competing interests, required institutional ethics/data-use statements, data redistribution terms, and rights to the project-generated trace. Do not interpret this checklist as completed declarations. Check [Springer’s publishing policies](https://www.springernature.com/gp/policies/book-publishing-policies) and the conference's current submission form. No submission, payment, copyright transfer, or acceptance has occurred.
