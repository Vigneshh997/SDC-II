import os
import urllib.request
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np

def create_proxy_dataset(num_samples=250):
    print(f"Setting up lightweight proxy dataset with {num_samples} samples...")
    data_root = Path("backend/data/nih_cxr14")
    images_dir = data_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate realistic fake metadata (Data_Entry_2017_v2020.csv)
    # We'll use the actual NIH label distribution for realism
    DISEASE_LABELS = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia"
    ]
    
    data = []
    patient_ids = np.random.randint(1, 1000, size=num_samples)
    
    # Generate some proxy image files (simple gradient noise to emulate x-rays)
    # To save time and disk space, we'll generate just 10 small base images and duplicate them
    base_images = []
    for i in range(10):
        img = np.random.normal(loc=128, scale=30, size=(224, 224)).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img).convert("RGB")
        base_images.append(pil_img)
        
    for i in range(num_samples):
        img_name = f"000{patient_ids[i]:05d}_{i:03d}.png"
        img_path = images_dir / img_name
        
        # Save a proxy image
        base_images[i % 10].save(img_path)
        
        # Assign random labels (mostly "No Finding", with some anomalies)
        if np.random.rand() < 0.4:
            labels = "No Finding"
        else:
            num_labels = np.random.randint(1, 3)
            labels = "|".join(np.random.choice(DISEASE_LABELS, num_labels, replace=False))
            
        data.append({
            "Image Index": img_name,
            "Finding Labels": labels,
            "Follow-up #": 0,
            "Patient ID": patient_ids[i],
            "Patient Age": np.random.randint(18, 90),
            "Patient Gender": np.random.choice(["M", "F"]),
            "View Position": "PA",
            "OriginalImage[Width": 1024,
            "Height]": 1024,
            "OriginalImagePixelSpacing[x": 0.143,
            "y]": 0.143
        })
        
    df = pd.DataFrame(data)
    csv_path = data_root / "Data_Entry_2017_v2020.csv"
    df.to_csv(csv_path, index=False)
    print(f"Created proxy CSV at {csv_path}")
    
    # 2. Create train/val and test splits
    # 80% train_val, 20% test
    unique_images = df["Image Index"].tolist()
    split_idx = int(len(unique_images) * 0.8)
    
    train_val = unique_images[:split_idx]
    test = unique_images[split_idx:]
    
    with open(data_root / "train_val_list.txt", "w") as f:
        f.write("\n".join(train_val))
        
    with open(data_root / "test_list.txt", "w") as f:
        f.write("\n".join(test))
        
    print(f"Created dataset splits: {len(train_val)} train/val, {len(test)} test.")
    print("Dataset setup complete! Ready for lightweight laptop training.")

if __name__ == "__main__":
    create_proxy_dataset()
