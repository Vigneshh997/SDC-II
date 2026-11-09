# DAViT reproduction assessment

Reference: Fu, Tantithamthavorn and Le, IEEE Access 2025, DOI 10.1109/ACCESS.2025.3579314. Local reference under `papers/references/` was read via complete text extraction (12 pages). Official replication repository: https://github.com/awsm-research/DAViT (read 31 October 2026).

DAViT uses a DINOv2-large transformer and shallow convolutional refinement. The paper describes 1x1 (1024→2048), grouped 3x3 (32 groups), then 1x1 convolution, BN/ReLU, average pooling and a classifier. Its wording that the last layer reduces channels “back to 2048” is ambiguous and should be resolved against implementation, not silently corrected. Patch/CLS arrangement and exact backbone checkpoint also need source verification.

CXR8 pathology-versus-benign supervised training precedes downstream fine-tuning. This is adaptation, not proof of generalization to an unseen hospital. Downstream tasks are binary pneumonia detection and, on pneumonia images, binary bacterial versus viral classification. Grad-CAM is visual explanation; subtype classification is a distinct predictive task.

The Kermany cohort totals 5,856 images: 5,216 training, 16 validation, 624 testing; subtype test counts are 148 viral and 242 bacterial. Paper settings include learning rate 1e-5, clipping 1, 30 epochs, AdamW and linear scheduling with 10% warm-up, on two RTX 3090 GPUs. It selects by validation loss. The tiny validation set and single pediatric source limit robustness conclusions.

Publication-only outcomes: detection F1 96.95%, AUC 95.56%; subtype F1 80.95%, AUC 84.05%. The reported ablation changes for removing CNN are 0.92 percentage points F1 and 0.90 AUC; removing adaptation changes them by 0.24 and 0.43 points. These are not X-Fusion outcomes or evidence of superiority of its tri-modal extension.

The repository README provides separate `davit/train_detection.sh` and `train_type.sh` routes and checkpoint download scripts. Deeper repository pages/raw model source could not be retrieved through the web tool in this session. Exact code-level topology, revisions, splits and weights therefore remain unverified; the README alone does not establish faithful reproduction. No upstream weights or dataset were downloaded and no reproduction was run.

Current X-Fusion preserves the global-transformer/local-CNN principle but replaces the backbone, convolutional block, task, encoders, loss and fusion. It is an extension inspired by DAViT, not a reproduction. A faithful baseline should remain a separately identified experiment in this project, using audited original scripts/weights and the published task/splits when feasible. Preserve its provenance and evaluate it independently of tri-modal results.

Presentation diagrams visually inspected: slide 11 uses Clinical-BERT wording and broad all-modality attribution; slide 12 specifies CXR-BERT and calls BiomedCLIP a DAViT backbone. These labels are design shorthand, not architectural equivalence. Both show segmentation and report mock-ups; neither proves implemented segmentation, calibration, supported example diseases, or explanation faithfulness.
