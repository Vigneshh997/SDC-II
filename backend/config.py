"""Shared scientific schema. Label order is part of every checkpoint contract."""

DISEASE_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]
ARCHITECTURE_VERSION = "xfusion-cross-attention-v2"
BIOMEDCLIP_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BIOMEDCLIP_REVISION = "9f341de24bfb00180f1b847274256e9b65a3a32e"
CXR_BERT_ID = "microsoft/BiomedVLP-CXR-BERT-specialized"
CXR_BERT_REVISION = "5157bdba1437a3aed316dacb1a5b68edf96b9902"
DISCLAIMER = "Research prototype only. Not clinically validated or approved for diagnosis or treatment."
FEATURES = {
    "age": {"unit": "years", "min": 0, "max": 120},
    "sex": {"unit": "category", "categories": ["female", "male", "other", "unknown"]},
    "temperature": {"unit": "C", "min": 30, "max": 45},
    "heart_rate": {"unit": "beats/min", "min": 20, "max": 250},
    "resp_rate": {"unit": "breaths/min", "min": 4, "max": 80},
    "spo2": {"unit": "%", "min": 0, "max": 100},
}
