import os
import subprocess
from pathlib import Path

def download_via_kaggle():
    """
    Downloads the NIH Chest X-ray 14 dataset using the Kaggle API.
    This is much faster and more reliable than the Box links.
    """
    data_root = Path("backend/data/nih_cxr14")
    data_root.mkdir(parents=True, exist_ok=True)
    
    print("="*50)
    print("NIH Chest X-ray 14 Full Dataset Downloader")
    print("="*50)
    print("This script uses the official Kaggle CLI to download the 45GB dataset.")
    print("Ensure you have your kaggle.json configured (usually in ~/.kaggle/).")
    
    # Check if kaggle is installed
    try:
        import kaggle
    except ImportError:
        print("Kaggle package not found. Installing...")
        subprocess.run(["pip", "install", "kaggle"], check=True)
    
    # Run the download command
    dataset_name = "nih-chest-xrays/data"
    print(f"\nDownloading {dataset_name} into {data_root}...")
    print("This will take a while depending on your cluster's internet speed...")
    
    try:
        subprocess.run([
            "kaggle", "datasets", "download", "-d", dataset_name,
            "-p", str(data_root), "--unzip"
        ], check=True)
        print("\nDownload and extraction complete!")
    except Exception as e:
        print(f"\nError downloading from Kaggle: {e}")
        print("Please ensure your Kaggle API token is set up correctly.")

if __name__ == "__main__":
    download_via_kaggle()
